"""
MediaPipe Multi-Person Landmark Extraction with Segmented Videos

This script handles multi-person landmark extraction by:
1. PASS 1: Detecting all people in the video using YOLO, creating separate video files
   where each video contains only one person (others blurred) at full frame resolution
2. PASS 2: Extracting landmarks from each segmented video independently using MediaPipe
   with segmentation enabled

Key Features:
- Creates per-person segmented videos maintaining original frame dimensions
- Enables MediaPipe segmentation by processing consistent-sized frames
- Extracts complete skeletons for each person independently
- Maintains person tracking across frames
- Gaussian blur for non-target persons
- GPU acceleration for YOLO detection

Usage:
  python mediapipe_segmented_extraction.py input_videos/ output_dir/
  python mediapipe_segmented_extraction.py input_videos/ output_dir/ --save_segmentation
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
BLUR_KERNEL_SIZE = 101  # Must be odd number for Gaussian blur
BLUR_SIGMA = 50.0  # Standard deviation for Gaussian blur
BBOX_PADDING = 0 # Padding around detected bounding box using yolo

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


def create_person_mask(frame, bbox, padding=0.2):
    """
    Create a binary mask for a specific person.
    
    Args:
        frame: Input frame
        bbox: YOLO bounding box [x1, y1, x2, y2]
        padding: Padding factor around bounding box
        
    Returns:
        Binary mask (1 for person, 0 for background)
    """
    height, width = frame.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    
    x1, y1, x2, y2 = [int(coord) for coord in bbox[:4]]
    x1 = max(0, min(x1, width - 1))
    y1 = max(0, min(y1, height - 1))
    x2 = max(x1 + 1, min(x2, width))
    y2 = max(y1 + 1, min(y2, height))
    
    # Add padding to the bounding box
    w, h = x2 - x1, y2 - y1
    pad_w, pad_h = int(w * padding), int(h * padding)
    
    pad_x1 = max(0, x1 - pad_w)
    pad_y1 = max(0, y1 - pad_h)
    pad_x2 = min(width, x2 + pad_w)
    pad_y2 = min(height, y2 + pad_h)
    
    # Set the person region to 1
    mask[pad_y1:pad_y2, pad_x1:pad_x2] = 1
    
    return mask


def blur_frame_except_person(frame, mask):
    """
    Blur the frame everywhere except where the mask is 1.
    
    Args:
        frame: Input frame (BGR)
        mask: Binary mask (1 for person, 0 for background)
        
    Returns:
        Frame with blurred background
    """
    # Apply Gaussian blur to entire frame
    blurred_frame = cv2.GaussianBlur(frame, (BLUR_KERNEL_SIZE, BLUR_KERNEL_SIZE), BLUR_SIGMA)
    
    # Create inverse mask
    inverse_mask = 1 - mask
    
    # Expand mask to 3 channels for BGR frame
    mask_3ch = np.stack([mask] * 3, axis=2)
    inverse_mask_3ch = np.stack([inverse_mask] * 3, axis=2)
    
    # Combine: keep original where mask=1, use blurred where mask=0
    result_frame = (frame * mask_3ch + blurred_frame * inverse_mask_3ch).astype(np.uint8)
    
    return result_frame


def create_segmented_videos(video_path, output_video_dir, log_file):
    """
    PASS 1: Create per-person segmented videos with blurred backgrounds.
    
    Args:
        video_path: Path to input video
        output_video_dir: Directory to save per-person videos
        log_file: Path to log file
        
    Returns:
        Dictionary mapping person_id to segmented video path
    """
    log(f"Starting PASS 1: Creating segmented videos for {os.path.basename(video_path)}", log_file)
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log(f"Error: Unable to open video file {video_path}", log_file)
        return {}
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    log(f"Video properties - FPS: {fps}, Resolution: {width}x{height}, Frames: {total_frames}", log_file)
    
    # Initialize person tracker
    person_tracker = PersonTracker()
    
    # Dictionary to store VideoWriter objects for each person
    person_videos = {}
    person_first_frame = {}
    person_original_frame_mapping = {}  # Maps person_id to list of original frame numbers
    
    os.makedirs(output_video_dir, exist_ok=True)
    
    # Process all frames
    frame_number = 0
    log(f"Processing frames to create segmented videos...", log_file)
    
    with tqdm(total=total_frames, desc=f"Creating segmented videos", unit="frame") as pbar:
        while frame_number < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            try:
                # Detect people in frame
                people_detections = detect_people_in_frame(frame)
                current_people = person_tracker.update(people_detections)
                
                # For each person, create blurred frame and write to their video
                for person_id, bbox in current_people.items():
                    # Create mask for this person
                    mask = create_person_mask(frame, bbox, padding=BBOX_PADDING)
                    
                    # Blur everything except this person
                    segmented_frame = blur_frame_except_person(frame, mask)
                    
                    # Create VideoWriter if first time seeing this person
                    if person_id not in person_videos:
                        output_video_path = os.path.join(output_video_dir, f"person_{person_id}.mp4")
                        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                        writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
                        person_videos[person_id] = writer
                        person_first_frame[person_id] = frame_number
                        person_original_frame_mapping[person_id] = []
                        log(f"Created output video for person {person_id}: {output_video_path} (first appearance: frame {frame_number})", log_file)
                    
                    # Track original frame number for this person
                    person_original_frame_mapping[person_id].append(frame_number)
                    
                    # Write frame to this person's video
                    person_videos[person_id].write(segmented_frame)
                
                frame_number += 1
                pbar.update(1)
                
            except Exception as e:
                log(f"Error processing frame {frame_number}: {e}", log_file)
                frame_number += 1
                pbar.update(1)
                continue
    
    # Release all VideoWriter objects
    for person_id, writer in person_videos.items():
        writer.release()
    
    cap.release()
    
    # Save frame mapping for each person
    for person_id, original_frames in person_original_frame_mapping.items():
        mapping_path = os.path.join(output_video_dir, f"person_{person_id}_frame_mapping.npy")
        np.save(mapping_path, np.array(original_frames, dtype=np.int32))
        log(f"Saved frame mapping for person {person_id}: {len(original_frames)} frames mapped", log_file)
    
    log(f"Completed PASS 1: Created {len(person_videos)} segmented videos", log_file)
    
    # Return mapping of person_id to video path
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


def extract_landmarks_from_segmented_video(video_path, person_id, segmented_videos_dir, log_file, save_segmentation=False):
    """
    PASS 2: Extract landmarks from a segmented video using MediaPipe.
    
    Since the video contains only one person at full frame resolution,
    MediaPipe can extract the complete skeleton with segmentation enabled.
    
    Loads frame mapping to tag landmarks with original video frame numbers.
    
    Args:
        video_path: Path to segmented video
        person_id: ID of the person in this video
        segmented_videos_dir: Directory containing frame mapping files
        log_file: Path to log file
        save_segmentation: Whether to save segmentation masks
        
    Returns:
        Dictionary with landmark data
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
    except Exception as e:
        log(f"[Person {person_id}] Error loading frame mapping: {e}", log_file)
        original_frame_numbers = None
    
    log(f"[Person {person_id}] Starting PASS 2: Extracting landmarks from {os.path.basename(video_path)}", log_file)
    
    # Initialize MediaPipe Holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=2,
        refine_face_landmarks=True,
        min_detection_confidence=0.5,
        enable_segmentation=True,
        min_tracking_confidence=0.5
    )
    
    # Open video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log(f"[Person {person_id}] Error: Unable to open video file {video_path}", log_file)
        return None
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    log(f"[Person {person_id}] Extracting landmarks from {total_frames} frames at {fps} fps", log_file)
    
    # Initialize data storage
    landmark_data = {
        'face_landmarks': [],
        'pose_landmarks': [],
        'pose_world_landmarks': [],
        'left_hand_landmarks': [],
        'right_hand_landmarks': [],
        'segmentation_masks': []
    }
    
    frame_number = 0
    
    with tqdm(total=total_frames, desc=f"Person {person_id} landmarks", unit="frame") as pbar:
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
                
                # Convert to RGB
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process with MediaPipe
                results = holistic.process(frame_rgb)
                
                # Extract all landmark types (normalized to full frame, which is what we want)
                face_lm = extract_landmarks_from_results(results, 'face_landmarks')
                pose_lm = extract_landmarks_from_results(results, 'pose_landmarks')
                pose_world_lm = extract_pose_world_landmarks(results)
                left_hand_lm = extract_landmarks_from_results(results, 'left_hand_landmarks')
                right_hand_lm = extract_landmarks_from_results(results, 'right_hand_landmarks')
                
                # Store landmarks with ORIGINAL frame number
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
                
                # Store segmentation mask if requested and available
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


def save_landmarks_to_mat(landmark_data, output_dir, person_id, log_file, save_segmentation=False):
    """Save extracted landmarks to .mat files"""
    try:
        person_dir = os.path.join(output_dir, f'person_{person_id}')
        os.makedirs(person_dir, exist_ok=True)
        
        # Save each landmark type
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
                log(f"Saved {landmark_type} for person {person_id}: {landmarks_data.shape}", log_file)
        
        # Save segmentation masks if requested
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
                log(f"Saved segmentation masks for person {person_id}: {masks_data.shape}", log_file)
            except Exception as e:
                log(f"Warning: Could not save segmentation masks for person {person_id}: {e}", log_file)
        
        log(f"Successfully saved landmark data for person {person_id} to {person_dir}", log_file)
        
    except Exception as e:
        log(f"Error saving landmarks for person {person_id}: {e}", log_file)
        traceback.print_exc()


def process_video_multi_pass(video_path, output_dir, log_file, save_segmentation=False):
    """
    Main pipeline: PASS 1 (create segmented videos) + PASS 2 (extract landmarks)
    
    Args:
        video_path: Path to input video
        output_dir: Base output directory
        log_file: Path to log file
        save_segmentation: Whether to save segmentation masks
    """
    # Create subdirectories
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_output_dir = os.path.join(output_dir, video_name)
    segmented_videos_dir = os.path.join(video_output_dir, 'segmented_videos')
    landmarks_output_dir = video_output_dir
    
    os.makedirs(video_output_dir, exist_ok=True)
    os.makedirs(segmented_videos_dir, exist_ok=True)
    
    log(f"\n{'='*80}", log_file)
    log(f"Processing: {video_path}", log_file)
    log(f"Output directory: {video_output_dir}", log_file)
    log(f"{'='*80}", log_file)
    
    # PASS 1: Create segmented videos
    person_video_mapping = create_segmented_videos(video_path, segmented_videos_dir, log_file)
    
    if not person_video_mapping:
        log(f"No people detected in video {video_path}. Skipping.", log_file)
        return False
    
    log(f"Created {len(person_video_mapping)} segmented videos", log_file)
    
    # PASS 2: Extract landmarks from each segmented video
    for person_id, segmented_video_path in person_video_mapping.items():
        log(f"\nProcessing person {person_id}...", log_file)
        
        landmark_data = extract_landmarks_from_segmented_video(
            segmented_video_path, person_id, segmented_videos_dir, log_file, save_segmentation)
        
        if landmark_data:
            save_landmarks_to_mat(landmark_data, landmarks_output_dir, person_id, log_file, save_segmentation)
            log(f"Completed processing for person {person_id}", log_file)
        else:
            log(f"Failed to extract landmarks for person {person_id}", log_file)
        
        gc.collect()
    
    log(f"\nCompleted processing for {video_path}", log_file)
    return True


def process_videos_in_directory(input_dir, output_dir, save_segmentation=False):
    """Process all videos in input directory"""
    # Find all video files
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
        
        # Create log file for this video
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
        
        process_video_multi_pass(video_path, output_dir, log_file, save_segmentation)
        gc.collect()
    
    log(f"\nAll videos processed.")


def main():
    parser = argparse.ArgumentParser(
        description="Multi-pass landmark extraction: segment videos per-person, then extract landmarks"
    )
    parser.add_argument('input_dir', type=str,
                       help="Path to directory containing input videos")
    parser.add_argument('output_dir', type=str,
                       help="Path to directory for output landmarks")
    parser.add_argument('--single_video', type=str, default=None,
                       help="Process only a single video file")
    parser.add_argument('--save_segmentation', action='store_true',
                       help="Save segmentation masks (uses significant storage)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"Input directory does not exist: {args.input_dir}")
        return
    
    print(f"Starting multi-pass landmark extraction...")
    print(f"Input directory: {args.input_dir}")
    print(f"Output directory: {args.output_dir}")
    print(f"Save segmentation: {args.save_segmentation}")
    
    if args.single_video:
        video_path = os.path.join(args.input_dir, args.single_video)
        if not os.path.exists(video_path):
            print(f"Video file not found: {video_path}")
            return
        
        video_name = os.path.splitext(args.single_video)[0]
        video_output_dir = os.path.join(args.output_dir, video_name)
        os.makedirs(video_output_dir, exist_ok=True)
        log_file = os.path.join(video_output_dir, 'extraction_log.txt')
        
        process_video_multi_pass(video_path, args.output_dir, log_file, args.save_segmentation)
    else:
        process_videos_in_directory(args.input_dir, args.output_dir, args.save_segmentation)
    
    print("Processing complete.")


if __name__ == '__main__':
    main()
