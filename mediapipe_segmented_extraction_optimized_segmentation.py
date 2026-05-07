"""
Optimized MediaPipe Multi-Person Landmark Extraction with Instance Segmentation

OPTIMIZATIONS IMPLEMENTED:
1. YOLO11-seg for pixel-perfect instance segmentation (no background noise).
2. Native BotSort algorithm for robust ID tracking across occlusions.
3. Parallel YOLO batch tracking - process multiple frames at once.
4. Only write frames when person is detected - skip empty frames.
5. Parallel person video processing - extract landmarks from multiple people simultaneously.
6. Batch frame writing - accumulate frames before writing.
7. GPU optimization for YOLO and MediaPipe.

Usage:
  python mediapipe_segmented_extraction_optimized.py input_videos/ output_dir/
  python mediapipe_segmented_extraction_optimized.py input_videos/ output_dir/ --processes 4 --save_segmentation
"""

import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'True'
import sys
import cv2
import numpy as np
import mediapipe as mp
import datetime
import argparse
import traceback
import torch
import gc
from scipy.io import savemat
from tqdm import tqdm
from ultralytics import YOLO
from multiprocessing import Pool
import csv


# Initialize MediaPipe holistic model
mp_holistic = mp.solutions.holistic

# Initialize person segmentation and tracking model (YOLO11-seg)
try:
    person_detection_model = YOLO('yolo11x-seg.pt')
    if torch.cuda.is_available():
        person_detection_model.to('cuda')
        print("YOLO11-seg model loaded with GPU acceleration")
    else:
        print("YOLO11-seg model loaded with CPU")
except Exception as e:
    print(f"Warning: Could not load YOLO11-seg model: {e}")
    person_detection_model = None

# Landmark counts
LANDMARK_COUNTS = {
    'face_landmarks': 478,
    'pose_landmarks': 33,
    'left_hand_landmarks': 21,
    'right_hand_landmarks': 21
}

# Blurring and Optimization parameters
BLUR_KERNEL_SIZE = 101
BLUR_SIGMA = 50.0
BATCH_SIZE = 16  # Adjusted for segmentation memory overhead
WRITE_BUFFER_SIZE = 30  # Buffer frames before writing to video


def log(message, log_file=None):
    """Print timestamped message and optionally write to log file"""
    timestamped_message = f"{datetime.datetime.now()} - {message}"
    print(timestamped_message)
    if log_file:
        with open(log_file, 'a') as f:
            f.write(timestamped_message + '\n')


def blur_frame_except_person_mask(frame, binary_mask):
    """Blur the frame everywhere except where the binary mask is True/1"""
    if len(binary_mask.shape) == 2:
        mask_3ch = np.stack([binary_mask] * 3, axis=-1)
    else:
        mask_3ch = binary_mask

    blurred_frame = cv2.GaussianBlur(frame, (BLUR_KERNEL_SIZE, BLUR_KERNEL_SIZE), BLUR_SIGMA)
    inverse_mask_3ch = 1 - mask_3ch
    
    result_frame = (frame * mask_3ch + blurred_frame * inverse_mask_3ch).astype(np.uint8)
    return result_frame

def mask_frame_gray_background(frame, binary_mask):
    """Replaces background with 50% gray almost instantly using NumPy broadcasting"""
    # Ensure mask is 3 channels
    if len(binary_mask.shape) == 2:
        mask_3ch = np.stack([binary_mask] * 3, axis=-1)
    else:
        mask_3ch = binary_mask

    # Create a solid gray background
    gray_bg = np.full(frame.shape, 128, dtype=np.uint8)
    
    # Fast NumPy where: if mask is 1 use frame, else use gray_bg
    return np.where(mask_3ch == 1, frame, gray_bg)

def create_segmented_videos_optimized(video_path, output_video_dir, log_file):
    """
    PASS 1 OPTIMIZED: Create per-person segmented videos using YOLO tracking + segmentation.
    Downsamples to 720p if the original video exceeds that resolution.
    """
    log(f"Starting PASS 1 (SEGMENTATION): Creating segmented videos for {os.path.basename(video_path)}", log_file)
    
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log(f"Error: Unable to open video file {video_path}", log_file)
        return {}
    
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    # --- Downsampling Logic ---
    TARGET_HEIGHT = 720
    if orig_height > TARGET_HEIGHT:
        scale = TARGET_HEIGHT / orig_height
        width = int(orig_width * scale)
        height = TARGET_HEIGHT
        
        # Video codecs generally prefer even dimensions
        width = width - (width % 2)
        height = height - (height % 2)
        
        log(f"Original resolution {orig_width}x{orig_height} exceeds 720p. Downsampling to {width}x{height}.", log_file)
        needs_resize = True
    else:
        width = orig_width
        height = orig_height
        needs_resize = False

    log(f"Processing Video properties - FPS: {fps}, Target Resolution: {width}x{height}, Frames: {total_frames}", log_file)
    
    person_videos = {}
    person_write_buffers = {}
    person_first_frame = {}
    person_original_frame_mapping = {} 
    
    os.makedirs(output_video_dir, exist_ok=True)
    
    frame_number = 0
    frame_batch = []
    frame_batch_numbers = []
    
    log(f"Processing frames with batch YOLO tracking (batch size: {BATCH_SIZE})...", log_file)
    print("Using custom tracker settings")

    with tqdm(total=total_frames, desc=f"Creating segmented videos (PASS 1)", unit="frame") as pbar:
        while frame_number < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Resize the frame immediately after reading to save batch memory
            if needs_resize:
                frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
            
            frame_batch.append(frame)
            frame_batch_numbers.append(frame_number)

            if len(frame_batch) >= BATCH_SIZE or frame_number == total_frames - 1:
                try:
                    # Native YOLO batched tracking with segmentation
                    results = person_detection_model.track(
                        source=frame_batch, 
                        persist=True, 
                        classes=[0], 
                        tracker="custom_tracker.yaml", 
                        verbose=False
                    )
                    
                    for batch_idx, (frame_num, result) in enumerate(zip(frame_batch_numbers, results)):
                        current_frame = frame_batch[batch_idx]
                        
                        # Verify tracking IDs and masks exist
                        if result.boxes is not None and result.boxes.id is not None and result.masks is not None:
                            track_ids = result.boxes.id.int().cpu().tolist()
                            masks = result.masks.data.cpu().numpy()
                            orig_shape = current_frame.shape[:2]
                            
                            for i, person_id in enumerate(track_ids):
                                # Resize mask to match the (now potentially downsampled) frame size exactly
                                mask = masks[i]
                                mask_resized = cv2.resize(mask, (orig_shape[1], orig_shape[0]), interpolation=cv2.INTER_NEAREST)
                                binary_mask = (mask_resized > 0.5).astype(np.uint8)
                                
                                segmented_frame = mask_frame_gray_background(current_frame, binary_mask)
                                
                                # Initialize writer for new persons using the target width and height
                                if person_id not in person_videos:
                                    output_video_path = os.path.join(output_video_dir, f"person_{person_id}.mp4")
                                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                                    writer = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))
                                    person_videos[person_id] = writer
                                    person_write_buffers[person_id] = []
                                    person_first_frame[person_id] = frame_num
                                    person_original_frame_mapping[person_id] = []
                                    log(f"Created output video for person {person_id}: {output_video_path} (first appearance: frame {frame_num})", log_file)
                                
                                person_original_frame_mapping[person_id].append(frame_num)
                                person_write_buffers[person_id].append(segmented_frame)
                                
                                if len(person_write_buffers[person_id]) >= WRITE_BUFFER_SIZE:
                                    for buffered_frame in person_write_buffers[person_id]:
                                        person_videos[person_id].write(buffered_frame)
                                    person_write_buffers[person_id] = []
                    
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
    
    # Flush remaining buffers
    for person_id, buffer in person_write_buffers.items():
        for buffered_frame in buffer:
            person_videos[person_id].write(buffered_frame)
    
    for person_id, writer in person_videos.items():
        writer.release()
    
    cap.release()
    
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
    """PASS 2: Extract MediaPipe landmarks from isolated tracked footage."""
    try:
        mapping_path = os.path.join(segmented_videos_dir, f"person_{person_id}_frame_mapping.npy")
        if os.path.exists(mapping_path):
            original_frame_numbers = np.load(mapping_path)
        else:
            log(f"[Person {person_id}] Warning: Frame mapping not found", log_file)
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
                    
                    landmark_data['face_landmarks'].append({'frame_no': original_frame_no, 'landmarks': face_lm})
                    landmark_data['pose_landmarks'].append({'frame_no': original_frame_no, 'landmarks': pose_lm})
                    landmark_data['pose_world_landmarks'].append({'frame_no': original_frame_no, 'landmarks': pose_world_lm})
                    landmark_data['left_hand_landmarks'].append({'frame_no': original_frame_no, 'landmarks': left_hand_lm})
                    landmark_data['right_hand_landmarks'].append({'frame_no': original_frame_no, 'landmarks': right_hand_lm})
                    
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
    """Main pipeline OPTIMIZED: PASS 1 (Tracker) + PASS 2 (MediaPipe parallel)"""
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    video_output_dir = os.path.join(output_dir, video_name)
    segmented_videos_dir = os.path.join(video_output_dir, 'segmented_videos')
    landmarks_output_dir = video_output_dir
    
    os.makedirs(video_output_dir, exist_ok=True)
    os.makedirs(segmented_videos_dir, exist_ok=True)
    
    log(f"\n{'='*80}", log_file)
    log(f"Processing (SEGMENTATION + TRACKER): {video_path}", log_file)
    
    # PASS 1: Create segmented videos
    person_video_mapping = create_segmented_videos_optimized(video_path, segmented_videos_dir, log_file)
    
    if not person_video_mapping:
        log(f"No people detected in video {video_path}. Skipping.", log_file)
        return False
    
    # PASS 2: Extract landmarks in parallel
    worker_args = []
    for person_id, segmented_video_path in person_video_mapping.items():
        worker_args.append((segmented_video_path, person_id, landmarks_output_dir, segmented_videos_dir, log_file, save_segmentation))
    
    if num_processes > 1 and len(worker_args) > 1:
        try:
            with Pool(processes=min(num_processes, len(worker_args))) as pool:
                results = pool.map(process_person_video_worker, worker_args)
        except Exception as e:
            log(f"Error in parallel processing: {e}. Falling back to sequential.", log_file)
            for args in worker_args:
                process_person_video_worker(args)
    else:
        for args in worker_args:
            process_person_video_worker(args)
    
    csv_path = os.path.join(video_output_dir, 'people_tracking_summary.csv')
    try:
        with open(csv_path, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Merged_Person_ID', 'Original_IDs', 'Label'])
            for pid in sorted(person_video_mapping.keys()):
                writer.writerow([pid, pid, ''])
    except Exception as e:
        log(f"Error creating summary CSV: {e}", log_file)

    log(f"\nCompleted processing for {video_path}", log_file)
    return True


def process_videos_in_directory(input_dir, output_dir, save_segmentation=False, num_processes=4):
    """Process all videos in input directory"""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.mxf', '.webm', '.flv']
    video_files = [f for f in os.listdir(input_dir) if any(f.lower().endswith(ext) for ext in video_extensions)]
    
    if not video_files:
        print(f"No video files found in {input_dir}")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    for i, video_file in enumerate(video_files, 1):
        video_path = os.path.join(input_dir, video_file)
        video_name = os.path.splitext(video_file)[0]
        video_output_dir = os.path.join(output_dir, video_name)
        os.makedirs(video_output_dir, exist_ok=True)
        log_file = os.path.join(video_output_dir, 'extraction_log.txt')
        
        person_dirs = [d for d in os.listdir(video_output_dir) if os.path.isdir(os.path.join(video_output_dir, d)) and d.startswith('person_')]
        if person_dirs:
            log(f"Landmarks already exist for {video_file}. Found {len(person_dirs)} people. Skipping.", log_file)
            continue
        
        process_video_multi_pass_optimized(video_path, output_dir, log_file, save_segmentation, num_processes)
        gc.collect()


def main():
    parser = argparse.ArgumentParser(description="Optimized multi-pass landmark extraction with instance segmentation tracking")
    parser.add_argument('input_dir', type=str, help="Path to directory containing input videos")
    parser.add_argument('output_dir', type=str, help="Path to directory for output landmarks")
    parser.add_argument('--single_video', type=str, default=None, help="Process only a single video file")
    parser.add_argument('--save_segmentation', action='store_true', help="Save segmentation masks (uses significant storage)")
    parser.add_argument('--processes', type=int, default=4, help="Number of parallel processes for Pass 2 (default: 4)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        print(f"Input directory does not exist: {args.input_dir}")
        return
    
    print(f"\n{'='*80}")
    print(f"OPTIMIZED Segmentation Multi-Pass Landmark Extraction")
    print(f"{'='*80}")
    print(f"Batch YOLO detection: ENABLED (batch size: {BATCH_SIZE})")
    
    if args.single_video:
        video_path = os.path.join(args.input_dir, args.single_video)
        video_name = os.path.splitext(args.single_video)[0]
        video_output_dir = os.path.join(args.output_dir, video_name)
        os.makedirs(video_output_dir, exist_ok=True)
        log_file = os.path.join(video_output_dir, 'extraction_log.txt')
        
        process_video_multi_pass_optimized(video_path, args.output_dir, log_file, args.save_segmentation, args.processes)
    else:
        process_videos_in_directory(args.input_dir, args.output_dir, args.save_segmentation, args.processes)

if __name__ == '__main__':
    main()