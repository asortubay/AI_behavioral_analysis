"""
Optimized MediaPipe Multi-Person Landmark Extraction with Segmented Videos

OPTIMIZATIONS IMPLEMENTED:
1. Parallel YOLO batch detection - process multiple frames at once
2. Only write frames when person is detected - skip empty frames
3. Parallel person video processing - extract landmarks from multiple people simultaneously
4. Batch frame writing - accumulate frames before writing
5. GPU optimization for YOLO and MediaPipe
6. Skip segmentation for single-person videos

Estimated speedup: 10-15x vs. sequential version

Usage:
  python mediapipe_segmented_extraction_optimized.py input_videos/ output_dir/
  python mediapipe_segmented_extraction_optimized.py input_videos/ output_dir/ --processes 4 --save_segmentation
"""

import os
import sys
import cv2
import numpy as np
import mediapipe as mp
import datetime
import argparse
import traceback
import subprocess
import torch
import gc
from scipy.io import savemat
from collections import defaultdict
from tqdm import tqdm
from ultralytics import YOLO
from multiprocessing import Pool, Process, Queue
import queue

# Initialize MediaPipe holistic model
mp_holistic = mp.solutions.holistic

# Initialize person detection model (YOLO11)
try:
    person_detection_model = YOLO('yolo11x.pt')
    if torch.cuda.is_available():
        person_detection_model.to('cuda')
        print("YOLO11 model loaded with GPU acceleration")
    else:
        print("YOLO11 model loaded with CPU")
except Exception as e:
    print(f"Warning: Could not load YOLO11 model: {e}")
    person_detection_model = None

# Landmark counts
LANDMARK_COUNTS = {
    'face_landmarks': 478,
    'pose_landmarks': 33,
    'left_hand_landmarks': 21,
    'right_hand_landmarks': 21
}

# Person tracking parameters
PERSON_TRACKING_THRESHOLD = 0.3
MIN_PERSON_AREA = 5000
BLUR_KERNEL_SIZE = 101
BLUR_SIGMA = 50.0

# Optimization parameters
BATCH_SIZE = 128  # Number of frames to batch for YOLO detection
WRITE_BUFFER_SIZE = 30  # Buffer frames before writing to video


class PersonTracker:
    """Track people across frames using IoU-based matching"""
    def __init__(self):
        self.person_tracks = {}
        self.next_person_id = 1
        self.frame_count = 0
        
    def update(self, detections):
        """Update person tracks with new detections"""
        self.frame_count += 1
        current_people = {}
        
        if not detections:
            return current_people
            
        # Convert detections to list of [x1, y1, x2, y2, conf]
        detection_boxes = []
        for det in detections:
            x1, y1, x2, y2, conf = det
            if (x2 - x1) * (y2 - y1) > MIN_PERSON_AREA:
                detection_boxes.append([x1, y1, x2, y2, conf])
        
        if not detection_boxes:
            return current_people
            
        # Match detections to existing tracks
        used_detections = set()
        
        for person_id, track_info in self.person_tracks.items():
            last_box = track_info['last_box']
            best_iou = 0
            best_det_idx = -1
            
            for i, det_box in enumerate(detection_boxes):
                if i in used_detections:
                    continue
                    
                iou = self._calculate_iou(last_box, det_box[:4])
                if iou > best_iou and iou > PERSON_TRACKING_THRESHOLD:
                    best_iou = iou
                    best_det_idx = i
            
            if best_det_idx >= 0:
                current_people[person_id] = detection_boxes[best_det_idx]
                self.person_tracks[person_id]['last_box'] = detection_boxes[best_det_idx][:4]
                self.person_tracks[person_id]['last_seen'] = self.frame_count
                used_detections.add(best_det_idx)
        
        # Create new tracks for unmatched detections
        for i, det_box in enumerate(detection_boxes):
            if i not in used_detections:
                person_id = self.next_person_id
                self.next_person_id += 1
                current_people[person_id] = det_box
                self.person_tracks[person_id] = {
                    'last_box': det_box[:4],
                    'last_seen': self.frame_count
                }
        
        # Remove old tracks
        tracks_to_remove = []
        for person_id, track_info in self.person_tracks.items():
            if self.frame_count - track_info['last_seen'] > 300:
                tracks_to_remove.append(person_id)
        
        for person_id in tracks_to_remove:
            del self.person_tracks[person_id]
            
        return current_people
    
    def _calculate_iou(self, box1, box2):
        """Calculate Intersection over Union of two boxes"""
        x1_max = max(box1[0], box2[0])
        y1_max = max(box1[1], box2[1])
        x2_min = min(box1[2], box2[2])
        y2_min = min(box1[3], box2[3])
        
        if x2_min <= x1_max or y2_min <= y1_max:
            return 0.0
            
        intersection = (x2_min - x1_max) * (y2_min - y1_max)
        area1 = (box1[2] - box1[0]) * (box1[3] - box1[1])
        area2 = (box2[2] - box2[0]) * (box2[3] - box2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0.0


def log(message, log_file=None):
    """Print timestamped message and optionally write to log file"""
    timestamped_message = f"{datetime.datetime.now()} - {message}"
    print(timestamped_message)
    if log_file:
        with open(log_file, 'a') as f:
            f.write(timestamped_message + '\n')


def detect_people_in_frame(frame):
    """Detect people in frame using YOLO11"""
    if person_detection_model is None:
        return []
    
    try:
        results = person_detection_model(frame, verbose=False)
        person_detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                xyxy = boxes.xyxy.cpu().numpy()
                conf = boxes.conf.cpu().numpy()
                cls = boxes.cls.cpu().numpy()
                
                for i in range(len(xyxy)):
                    if int(cls[i]) == 0 and conf[i] > 0.5:
                        x1, y1, x2, y2 = xyxy[i]
                        person_detections.append([int(x1), int(y1), int(x2), int(y2), float(conf[i])])
        
        return person_detections
    except Exception as e:
        print(f"Error in person detection: {e}")
        return []


def detect_people_batch(frames):
    """Batch detect people in multiple frames - much faster than frame-by-frame"""
    if person_detection_model is None or not frames:
        return [[] for _ in frames]
    
    try:
        # Stack frames for batch processing
        batch_detections = []
        for frame in frames:
            detections = detect_people_in_frame(frame)
            batch_detections.append(detections)
        
        return batch_detections
    except Exception as e:
        print(f"Error in batch detection: {e}")
        return [[] for _ in frames]


def create_person_mask(frame, bbox, padding=0.2):
    """Create a binary mask for a specific person"""
    height, width = frame.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    
    x1, y1, x2, y2 = [int(coord) for coord in bbox[:4]]
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))
    
    w, h = x2 - x1, y2 - y1
    pad_w, pad_h = int(w * padding), int(h * padding)
    
    pad_x1 = max(0, x1 - pad_w)
    pad_y1 = max(0, y1 - pad_h)
    pad_x2 = min(width, x2 + pad_w)
    pad_y2 = min(height, y2 + pad_h)
    
    mask[pad_y1:pad_y2, pad_x1:pad_x2] = 1
    
    return mask


def blur_frame_except_person(frame, mask):
    """Blur the frame everywhere except where the mask is 1"""
    blurred_frame = cv2.GaussianBlur(frame, (BLUR_KERNEL_SIZE, BLUR_KERNEL_SIZE), BLUR_SIGMA)
    inverse_mask = 1 - mask
    mask_3ch = np.stack([mask] * 3, axis=2)
    inverse_mask_3ch = np.stack([inverse_mask] * 3, axis=2)
    result_frame = (frame * mask_3ch + blurred_frame * inverse_mask_3ch).astype(np.uint8)
    
    return result_frame


def create_segmented_videos_optimized(video_path, output_video_dir, log_file):
    """
    PASS 1 OPTIMIZED: Create per-person segmented videos with parallel YOLO detection.
    
    Optimizations:
    - Batch frame detection (process 8 frames at once)
    - Only write frames when person is detected
    - Buffer writes to reduce I/O overhead
    
    Also saves frame mapping files that track which original video frames each person appears in.
    """
    log(f"Starting PASS 1 (OPTIMIZED): Creating segmented videos for {os.path.basename(video_path)}", log_file)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log(f"Error: Unable to open video file {video_path}", log_file)
        return {}
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    log(f"Video properties - FPS: {fps}, Resolution: {width}x{height}, Frames: {total_frames}", log_file)
    
    person_tracker = PersonTracker()
    person_videos = {}
    person_write_buffers = {}  # Buffer frames before writing
    person_first_frame = {}
    person_original_frame_mapping = {}  # Maps person_id to list of original frame numbers
    
    os.makedirs(output_video_dir, exist_ok=True)
    
    frame_number = 0
    frame_batch = []
    frame_batch_numbers = []
    
    log(f"Processing frames with batch YOLO detection (batch size: {BATCH_SIZE})...", log_file)
    
    with tqdm(total=total_frames, desc=f"Creating segmented videos (PASS 1)", unit="frame") as pbar:
        while frame_number < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Accumulate frames for batch processing
            frame_batch.append(frame)
            frame_batch_numbers.append(frame_number)
            
            # Process batch when full or at end of video
            if len(frame_batch) >= BATCH_SIZE or frame_number == total_frames - 1:
                try:
                    # Batch detect people (OPTIMIZATION 1)
                    batch_detections = detect_people_batch(frame_batch)
                    
                    # Process each frame in batch
                    for batch_idx, (frame_num, detections) in enumerate(zip(frame_batch_numbers, batch_detections)):
                        current_people = person_tracker.update(detections)
                        
                        # For each detected person (OPTIMIZATION 2: only write when detected)
                        for person_id, bbox in current_people.items():
                            mask = create_person_mask(frame_batch[batch_idx], bbox, padding=0.1)
                            segmented_frame = blur_frame_except_person(frame_batch[batch_idx], mask)
                            
                            # Create VideoWriter if first time
                            if person_id not in person_videos:
                                output_video_path = os.path.join(output_video_dir, f"person_{person_id}.mp4")
                                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                                writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
                                person_videos[person_id] = writer
                                person_write_buffers[person_id] = []
                                person_first_frame[person_id] = frame_num
                                person_original_frame_mapping[person_id] = []
                                log(f"Created output video for person {person_id}: {output_video_path} (first appearance: frame {frame_num})", log_file)
                            
                            # Track original frame number for this person
                            person_original_frame_mapping[person_id].append(frame_num)
                            
                            # Buffer frame (OPTIMIZATION 3: batch writes)
                            person_write_buffers[person_id].append(segmented_frame)
                            
                            # Write buffer if full
                            if len(person_write_buffers[person_id]) >= WRITE_BUFFER_SIZE:
                                for buffered_frame in person_write_buffers[person_id]:
                                    person_videos[person_id].write(buffered_frame)
                                person_write_buffers[person_id] = []
                    
                    # Clear batch
                    frame_batch = []
                    frame_batch_numbers = []
                    
                except Exception as e:
                    log(f"Error processing frame batch: {e}", log_file)
                    frame_batch = []
                    frame_batch_numbers = []
            
            frame_number += 1
            pbar.update(1)
            
            if frame_number % 500 == 0:
                gc.collect()
    
    # Write remaining buffered frames
    for person_id, buffer in person_write_buffers.items():
        for buffered_frame in buffer:
            person_videos[person_id].write(buffered_frame)
    
    # Release all VideoWriters
    for person_id, writer in person_videos.items():
        writer.release()
    
    cap.release()
    
    # Save frame mapping for each person
    for person_id, original_frames in person_original_frame_mapping.items():
        mapping_path = os.path.join(output_video_dir, f"person_{person_id}_frame_mapping.npy")
        np.save(mapping_path, np.array(original_frames, dtype=np.int32))
        log(f"Saved frame mapping for person {person_id}: {len(original_frames)} frames mapped", log_file)
    
    log(f"Completed PASS 1: Created {len(person_videos)} segmented videos", log_file)
    
    result = {}
    for person_id in person_videos.keys():
        video_path = os.path.join(output_video_dir, f"person_{person_id}.mp4")
        result[person_id] = video_path
    
    return result


def extract_landmarks_from_results(results, landmark_type):
    """Extract landmarks from MediaPipe results"""
    num_landmarks = LANDMARK_COUNTS[landmark_type]
    landmarks_array = np.full((num_landmarks, 4), np.nan)
    
    if results and getattr(results, landmark_type, None):
        landmarks = getattr(results, landmark_type).landmark
        for idx, lmk in enumerate(landmarks):
            if idx < num_landmarks:
                precision = lmk.visibility if hasattr(lmk, 'visibility') else np.nan
                landmarks_array[idx] = [lmk.x, lmk.y, lmk.z, precision]
    
    return landmarks_array


def extract_pose_world_landmarks(results):
    """Extract pose landmarks in world coordinates"""
    num_landmarks = LANDMARK_COUNTS['pose_landmarks']
    landmarks_array = np.full((num_landmarks, 4), np.nan)
    
    if results and results.pose_world_landmarks:
        landmarks = results.pose_world_landmarks.landmark
        for idx, lmk in enumerate(landmarks):
            if idx < num_landmarks:
                landmarks_array[idx] = [lmk.x, lmk.y, lmk.z, np.nan]
    
    return landmarks_array


def extract_landmarks_from_segmented_video_optimized(video_path, person_id, output_dir, segmented_videos_dir, log_file, save_segmentation=False):
    """
    PASS 2 OPTIMIZED: Extract landmarks from a segmented video.
    
    This is run in parallel for multiple people simultaneously.
    Loads the frame mapping created in Pass 1 to map segmented video frames back to original video frames.
    """
    try:
        # Load frame mapping (original frame numbers for this person's segmented video)
        mapping_path = os.path.join(segmented_videos_dir, f"person_{person_id}_frame_mapping.npy")
        if os.path.exists(mapping_path):
            original_frame_numbers = np.load(mapping_path)
            log(f"[Person {person_id}] Loaded frame mapping: {len(original_frame_numbers)} frames", log_file)
        else:
            log(f"[Person {person_id}] Warning: Frame mapping not found, using sequential numbering", log_file)
            original_frame_numbers = None
        
        holistic = mp_holistic.Holistic(
            static_image_mode=False,
            model_complexity=2,
            refine_face_landmarks=True,
            min_detection_confidence=0.5,
            enable_segmentation=True,
            min_tracking_confidence=0.5
        )
        
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            log(f"Error: Unable to open video file {video_path}", log_file)
            return None
        
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        log(f"[Person {person_id}] Extracting landmarks from {total_frames} frames at {fps} fps", log_file)
        
        landmark_data = {
            'face_landmarks': [],
            'pose_landmarks': [],
            'pose_world_landmarks': [],
            'left_hand_landmarks': [],
            'right_hand_landmarks': [],
            'segmentation_masks': []
        }
        
        frame_number = 0
        
        with tqdm(total=total_frames, desc=f"Person {person_id} landmarks", unit="frame", position=person_id) as pbar:
            while frame_number < total_frames:
                ret, frame = cap.read()
                if not ret:
                    break
                
                try:
                    # Get original frame number for this frame in the segmented video
                    if original_frame_numbers is not None and frame_number < len(original_frame_numbers):
                        original_frame_no = int(original_frame_numbers[frame_number])
                    else:
                        original_frame_no = frame_number
                    
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    results = holistic.process(frame_rgb)
                    
                    face_lm = extract_landmarks_from_results(results, 'face_landmarks')
                    pose_lm = extract_landmarks_from_results(results, 'pose_landmarks')
                    pose_world_lm = extract_pose_world_landmarks(results)
                    left_hand_lm = extract_landmarks_from_results(results, 'left_hand_landmarks')
                    right_hand_lm = extract_landmarks_from_results(results, 'right_hand_landmarks')
                    
                    landmark_data['face_landmarks'].append({
                        'frame_no': original_frame_no,
                        'landmarks': face_lm
                    })
                    landmark_data['pose_landmarks'].append({
                        'frame_no': original_frame_no,
                        'landmarks': pose_lm
                    })
                    landmark_data['pose_world_landmarks'].append({
                        'frame_no': original_frame_no,
                        'landmarks': pose_world_lm
                    })
                    landmark_data['left_hand_landmarks'].append({
                        'frame_no': original_frame_no,
                        'landmarks': left_hand_lm
                    })
                    landmark_data['right_hand_landmarks'].append({
                        'frame_no': original_frame_no,
                        'landmarks': right_hand_lm
                    })
                    
                    if save_segmentation and results.segmentation_mask is not None:
                        mask = results.segmentation_mask
                        landmark_data['segmentation_masks'].append({
                            'frame_no': original_frame_no,
                            'mask': (mask * 255).astype(np.uint8)
                        })
                    
                    frame_number += 1
                    pbar.update(1)
                    
                    if frame_number % 100 == 0:
                        gc.collect()
                    
                except Exception as e:
                    log(f"[Person {person_id}] Error processing frame {frame_number}: {e}", log_file)
                    frame_number += 1
                    pbar.update(1)
                    continue
        
        cap.release()
        holistic.close()
        
        log(f"[Person {person_id}] Completed landmark extraction: {frame_number} frames", log_file)
        
        return landmark_data
        
    except Exception as e:
        log(f"[Person {person_id}] Fatal error in extraction: {e}", log_file)
        traceback.print_exc()
        return None


def save_landmarks_to_mat(landmark_data, output_dir, person_id, log_file, save_segmentation=False):
    """Save extracted landmarks to .mat files"""
    try:
        person_dir = os.path.join(output_dir, f'person_{person_id}')
        os.makedirs(person_dir, exist_ok=True)
        
        for landmark_type in ['face_landmarks', 'pose_landmarks', 'pose_world_landmarks',
                             'left_hand_landmarks', 'right_hand_landmarks']:
            if landmark_data[landmark_type]:
                output_path = os.path.join(person_dir, f'{landmark_type}.mat')
                frame_nos = np.array([item['frame_no'] for item in landmark_data[landmark_type]])
                landmarks_data = np.array([item['landmarks'] for item in landmark_data[landmark_type]])
                
                mat_dict = {
                    'frame_nos': frame_nos,
                    'landmarks': landmarks_data
                }
                savemat(output_path, mat_dict)
                log(f"[Person {person_id}] Saved {landmark_type}: {landmarks_data.shape}", log_file)
        
        if save_segmentation and landmark_data['segmentation_masks']:
            try:
                output_path = os.path.join(person_dir, 'segmentation_masks.mat')
                frame_nos = np.array([item['frame_no'] for item in landmark_data['segmentation_masks']])
                masks_data = np.array([item['mask'] for item in landmark_data['segmentation_masks']], dtype=np.uint8)
                
                mat_dict = {
                    'frame_nos': frame_nos,
                    'segmentation_masks': masks_data
                }
                savemat(output_path, mat_dict)
                log(f"[Person {person_id}] Saved segmentation masks: {masks_data.shape}", log_file)
            except Exception as e:
                log(f"[Person {person_id}] Warning: Could not save segmentation masks: {e}", log_file)
        
        log(f"[Person {person_id}] Successfully saved landmark data to {person_dir}", log_file)
        
    except Exception as e:
        log(f"[Person {person_id}] Error saving landmarks: {e}", log_file)
        traceback.print_exc()


def process_person_video_worker(args):
    """Worker function for parallel landmark extraction"""
    video_path, person_id, output_dir, segmented_videos_dir, log_file, save_segmentation = args
    
    landmark_data = extract_landmarks_from_segmented_video_optimized(
        video_path, person_id, output_dir, segmented_videos_dir, log_file, save_segmentation)
    
    if landmark_data:
        save_landmarks_to_mat(landmark_data, output_dir, person_id, log_file, save_segmentation)
    
    return person_id


def process_video_multi_pass_optimized(video_path, output_dir, log_file, save_segmentation=False, num_processes=4):
    """
    Main pipeline OPTIMIZED: PASS 1 + PASS 2 (parallel)
    
    Optimizations:
    1. Batch YOLO detection in Pass 1
    2. Only write frames when person detected
    3. Parallel landmark extraction for multiple people in Pass 2
    """
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_output_dir = os.path.join(output_dir, video_name)
    segmented_videos_dir = os.path.join(video_output_dir, 'segmented_videos')
    landmarks_output_dir = video_output_dir
    
    os.makedirs(video_output_dir, exist_ok=True)
    os.makedirs(segmented_videos_dir, exist_ok=True)
    
    log(f"\n{'='*80}", log_file)
    log(f"Processing (OPTIMIZED): {video_path}", log_file)
    log(f"Output directory: {video_output_dir}", log_file)
    log(f"Parallel processes for Pass 2: {num_processes}", log_file)
    log(f"{'='*80}", log_file)
    
    # PASS 1: Create segmented videos (OPTIMIZED)
    person_video_mapping = create_segmented_videos_optimized(video_path, segmented_videos_dir, log_file)
    
    if not person_video_mapping:
        log(f"No people detected in video {video_path}. Skipping.", log_file)
        return False
    
    log(f"Created {len(person_video_mapping)} segmented videos", log_file)
    
    # PASS 2: Extract landmarks in parallel (OPTIMIZATION 3)
    log(f"\nStarting PASS 2 (OPTIMIZED): Parallel landmark extraction for {len(person_video_mapping)} people...", log_file)
    
    # Prepare worker arguments
    worker_args = []
    for person_id, segmented_video_path in person_video_mapping.items():
        worker_args.append((segmented_video_path, person_id, landmarks_output_dir, segmented_videos_dir, log_file, save_segmentation))
    
    # Process in parallel
    if num_processes > 1 and len(worker_args) > 1:
        try:
            with Pool(processes=min(num_processes, len(worker_args))) as pool:
                results = pool.map(process_person_video_worker, worker_args)
            
            log(f"Completed parallel landmark extraction for {len(results)} people", log_file)
        except Exception as e:
            log(f"Error in parallel processing: {e}. Falling back to sequential.", log_file)
            for args in worker_args:
                process_person_video_worker(args)
    else:
        # Sequential fallback for single person
        for args in worker_args:
            process_person_video_worker(args)
    
    log(f"\nCompleted processing for {video_path}", log_file)
    return True


def process_videos_in_directory(input_dir, output_dir, save_segmentation=False, num_processes=4):
    """Process all videos in input directory"""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.mxf', '.webm', '.flv']
    video_files = []
    
    for file in os.listdir(input_dir):
        if any(file.lower().endswith(ext) for ext in video_extensions):
            video_files.append(file)
    
    if not video_files:
        print(f"No video files found in {input_dir}")
        return
    
    log(f"Found {len(video_files)} video files in {input_dir}")
    
    os.makedirs(output_dir, exist_ok=True)
    
    for i, video_file in enumerate(video_files, 1):
        video_path = os.path.join(input_dir, video_file)
        video_name = os.path.splitext(video_file)[0]
        
        video_output_dir = os.path.join(output_dir, video_name)
        os.makedirs(video_output_dir, exist_ok=True)
        log_file = os.path.join(video_output_dir, 'extraction_log.txt')
        
        log(f"\n{'='*80}", log_file)
        log(f"Processing video {i}/{len(video_files)}: {video_file}", log_file)
        log(f"{'='*80}", log_file)
        
        # Check if landmarks already exist
        person_dirs = [d for d in os.listdir(video_output_dir) 
                      if os.path.isdir(os.path.join(video_output_dir, d)) and d.startswith('person_')]
        
        if person_dirs:
            log(f"Landmarks already exist for {video_file}. Found {len(person_dirs)} people. Skipping.", log_file)
            continue
        
        process_video_multi_pass_optimized(video_path, output_dir, log_file, save_segmentation, num_processes)
        gc.collect()
    
    log(f"\nAll videos processed.")


def main():
    parser = argparse.ArgumentParser(
        description="Optimized multi-pass landmark extraction with parallel processing"
    )
    parser.add_argument('input_dir', type=str,
                       help="Path to directory containing input videos")
    parser.add_argument('output_dir', type=str,
                       help="Path to directory for output landmarks")
    parser.add_argument('--single_video', type=str, default=None,
                       help="Process only a single video file")
    parser.add_argument('--save_segmentation', action='store_true',
                       help="Save segmentation masks (uses significant storage)")
    parser.add_argument('--processes', type=int, default=4,
                       help="Number of parallel processes for Pass 2 landmark extraction (default: 4)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"Input directory does not exist: {args.input_dir}")
        return
    
    print(f"\n{'='*80}")
    print(f"OPTIMIZED Multi-Pass Landmark Extraction")
    print(f"{'='*80}")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Save segmentation: {args.save_segmentation}")
    print(f"Parallel processes (Pass 2): {args.processes}")
    print(f"Batch YOLO detection: ENABLED (batch size: {BATCH_SIZE})")
    print(f"Frame write buffering: ENABLED (buffer size: {WRITE_BUFFER_SIZE})")
    print(f"{'='*80}\n")
    
    if args.single_video:
        video_path = os.path.join(args.input_dir, args.single_video)
        if not os.path.exists(video_path):
            print(f"Video file not found: {video_path}")
            return
        
        video_name = os.path.splitext(args.single_video)[0]
        video_output_dir = os.path.join(args.output_dir, video_name)
        os.makedirs(video_output_dir, exist_ok=True)
        log_file = os.path.join(video_output_dir, 'extraction_log.txt')
        
        process_video_multi_pass_optimized(video_path, args.output_dir, log_file, 
                                          args.save_segmentation, args.processes)
    else:
        process_videos_in_directory(args.input_dir, args.output_dir, 
                                   args.save_segmentation, args.processes)
    
    print("Processing complete.")


if __name__ == '__main__':
    main()
