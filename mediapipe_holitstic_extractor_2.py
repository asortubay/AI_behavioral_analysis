import os
import cv2
import numpy as np
import json
import subprocess
from scipy.io import savemat
import argparse
from tqdm import tqdm
import mediapipe as mp
import multiprocessing as mp_process
import time
import gc
import traceback
import datetime
from collections import defaultdict
from ultralytics import YOLO

# Initialize MediaPipe holistic model in the main process
mp_holistic = mp.solutions.holistic

# Initialize person detection model (YOLO11)
try:
    # Load YOLO11 model for person detection with GPU support
    person_detection_model = YOLO('yolo11x.pt')  # nano version for speed, use 'yolo11s.pt' for better accuracy
    
    # Check if CUDA is available and set device
    import torch
    if torch.cuda.is_available():
        person_detection_model.to('cuda')
        print("YOLO11 model loaded with GPU acceleration")
    else:
        print("YOLO11 model loaded with CPU")
        
except Exception as e:
    print(f"Warning: Could not load YOLO11 model: {e}")
    person_detection_model = None

# face landmarks count is 478 including iris landmarks (468 + 10 extra iris points)
LANDMARK_COUNTS = {
    'face_landmarks': 478,
    'pose_landmarks': 33,
    'left_hand_landmarks': 21,
    'right_hand_landmarks': 21
}

# Person tracking parameters
PERSON_TRACKING_THRESHOLD = 0.3  # IoU threshold for tracking people across frames
MIN_PERSON_AREA = 5000  # Minimum area for a person detection
CROP_PADDING = 0.2  # Padding around person detection for cropping

class PersonTracker:
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
            if (x2 - x1) * (y2 - y1) > MIN_PERSON_AREA:  # Filter small detections
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
                # Update existing track
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
        
        # Remove old tracks (not seen for many frames)
        tracks_to_remove = []
        for person_id, track_info in self.person_tracks.items():
            if self.frame_count - track_info['last_seen'] > 300:  # 30 frames threshold
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

def detect_people_in_frame(frame):
    """Detect people in frame using YOLO11"""
    if person_detection_model is None:
        return []
    
    try:
        # Run inference with YOLO11
        results = person_detection_model(frame, verbose=False)
        
        person_detections = []
        
        # Extract detections from YOLO11 results
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                # Get detection data
                xyxy = boxes.xyxy.cpu().numpy()  # bounding boxes in xyxy format
                conf = boxes.conf.cpu().numpy()  # confidence scores
                cls = boxes.cls.cpu().numpy()   # class indices
                
                # Filter for person class (0) and confidence > 0.5
                for i in range(len(xyxy)):
                    if int(cls[i]) == 0 and conf[i] > 0.5:  # Person class and confidence threshold
                        x1, y1, x2, y2 = xyxy[i]
                        person_detections.append([int(x1), int(y1), int(x2), int(y2), float(conf[i])])
        
        return person_detections
    except Exception as e:
        print(f"Error in person detection: {e}")
        return []

def crop_person_region(frame, bbox, padding=0.1):
    """
    Crop frame around person with padding for optimal MediaPipe processing.
    
    Args:
        frame: Input video frame
        bbox: YOLO bounding box [x1, y1, x2, y2] in pixel coordinates
        padding: Padding factor (default 0.1 = 10% padding around bounding box)
    
    Returns:
        cropped_frame: Cropped image focused on the person
        crop_coords: (x1, y1, x2, y2) coordinates of the crop in original frame
    """
    height, width = frame.shape[:2]
    x1, y1, x2, y2 = [int(coord) for coord in bbox[:4]]
    
    # Ensure bounding box is valid
    x1 = max(0, min(x1, width-1))
    y1 = max(0, min(y1, height-1))
    x2 = max(x1+1, min(x2, width))
    y2 = max(y1+1, min(y2, height))
    
    # Calculate dimensions and padding
    w, h = x2 - x1, y2 - y1
    pad_w, pad_h = int(w * padding), int(h * padding)
    
    # Calculate crop coordinates with padding, ensuring they stay within frame bounds
    crop_x1 = max(0, x1 - pad_w)
    crop_y1 = max(0, y1 - pad_h)
    crop_x2 = min(width, x2 + pad_w)
    crop_y2 = min(height, y2 + pad_h)
    
    # Ensure minimum crop size for MediaPipe reliability
    min_size = 50
    crop_w = crop_x2 - crop_x1
    crop_h = crop_y2 - crop_y1
    
    if crop_w < min_size or crop_h < min_size:
        # Expand crop to minimum size if needed
        center_x = (crop_x1 + crop_x2) // 2
        center_y = (crop_y1 + crop_y2) // 2
        
        half_size = max(min_size // 2, max(crop_w, crop_h) // 2)
        
        crop_x1 = max(0, center_x - half_size)
        crop_y1 = max(0, center_y - half_size)
        crop_x2 = min(width, center_x + half_size)
        crop_y2 = min(height, center_y + half_size)
    
    # Perform the crop
    cropped_frame = frame[crop_y1:crop_y2, crop_x1:crop_x2]
    
    return cropped_frame, (crop_x1, crop_y1, crop_x2, crop_y2)

def adjust_landmarks_to_original_frame(landmarks_array, crop_coords, original_frame_shape):
    """
    Adjust landmarks from cropped frame back to original frame coordinates.
    
    MediaPipe returns normalized coordinates (0-1) relative to the cropped image.
    We need to convert these to normalized coordinates relative to the original frame.
    
    Args:
        landmarks_array: Array of landmarks from MediaPipe (normalized to crop)
        crop_coords: (x1, y1, x2, y2) crop coordinates in original frame
        original_frame_shape: (height, width, channels) of original frame
        
    Returns:
        Adjusted landmarks normalized to original frame
    """
    if landmarks_array is None or np.all(np.isnan(landmarks_array)):
        return landmarks_array
    
    crop_x1, crop_y1, crop_x2, crop_y2 = crop_coords
    orig_height, orig_width = original_frame_shape[:2]
    crop_width = crop_x2 - crop_x1
    crop_height = crop_y2 - crop_y1
    
    # Avoid division by zero
    if crop_width <= 0 or crop_height <= 0:
        return landmarks_array
    
    adjusted_landmarks = landmarks_array.copy()
    
    for i, landmark in enumerate(landmarks_array):
        x, y, z, precision = landmark
        if not (np.isnan(x) or np.isnan(y)):
            # Step 1: Convert from normalized coordinates in crop to pixel coordinates in crop
            pixel_x_in_crop = x * crop_width
            pixel_y_in_crop = y * crop_height
            
            # Step 2: Convert to pixel coordinates in original frame
            pixel_x_in_orig = pixel_x_in_crop + crop_x1
            pixel_y_in_orig = pixel_y_in_crop + crop_y1
            
            # Step 3: Normalize to original frame dimensions
            norm_x_in_orig = pixel_x_in_orig / orig_width
            norm_y_in_orig = pixel_y_in_orig / orig_height
            
            # Clamp to valid range [0, 1]
            norm_x_in_orig = max(0.0, min(1.0, norm_x_in_orig))
            norm_y_in_orig = max(0.0, min(1.0, norm_y_in_orig))
            
            adjusted_landmarks[i] = [norm_x_in_orig, norm_y_in_orig, z, precision]
    
    return adjusted_landmarks

def log(message, log_file=None):
    timestamped_message = f"{datetime.datetime.now()} - {message}"
    print(timestamped_message)
    if log_file:
        with open(log_file, 'a') as f:
            f.write(timestamped_message + '\n')

def convert_video_to_mp4(input_path, output_path, log_file):
    command = [
        'ffmpeg',
        '-i', input_path,
        '-c:v', 'libx264',
        '-crf', '23',
        '-preset', 'fast',
        '-c:a', 'aac',
        '-b:a', '192k',
        output_path
    ]
    log(f"Running command: {' '.join(command)}", log_file)
    subprocess.run(command, check=True)
    log(f"Conversion complete: {input_path} to {output_path}", log_file)

def validate_landmark_data(landmarks_list, log_file):
    valid = True
    for i, item in enumerate(landmarks_list):
        if not isinstance(item['landmarks'], list):
            log(f"Invalid data at index {i}: not a list", log_file)
            valid = False
        for lmk in item['landmarks']:
            if not (isinstance(lmk, list) and len(lmk) == 4 and all(isinstance(coord, (int, float)) for coord in lmk)):
                log(f"Invalid landmark data at index {i}: {lmk}", log_file)
                valid = False
            if any(isinstance(coord, float) and np.isnan(coord) for coord in lmk):
                log(f"Landmark contains NaN value at index {i}: {lmk}", log_file)
                valid = False
            if any(coord == 0 for coord in lmk):
                log(f"Landmark contains zero value at index {i}: {lmk}", log_file)
    return valid

def filter_people_by_duration(people_landmarks, fps, min_duration_seconds=5, log_file=None):
    """Filter out people with less than minimum duration of data"""
    min_frames = min_duration_seconds * fps
    filtered_people = {}
    
    log(f"Filtering people with less than {min_duration_seconds} seconds of data (minimum {min_frames} frames)", log_file)
    
    for person_id, person_data in people_landmarks.items():
        frame_count = len(person_data['face_landmarks'])
        duration_seconds = frame_count / fps
        
        if frame_count >= min_frames:
            filtered_people[person_id] = person_data
            log(f"Person {person_id}: {frame_count} frames ({duration_seconds:.2f}s) - KEPT", log_file)
        else:
            log(f"Person {person_id}: {frame_count} frames ({duration_seconds:.2f}s) - REMOVED (too short)", log_file)
    
    log(f"Kept {len(filtered_people)} out of {len(people_landmarks)} people after duration filtering", log_file)
    return filtered_people

def save_landmarks_multi_person(people_landmarks, output_subdir, log_file, save_segmentation=False):
    """Save landmarks for multiple people separately"""
    if not people_landmarks:
        log(f"No landmark data to save.", log_file)
        return

    try:
        for person_id, person_data in people_landmarks.items():
            person_dir = os.path.join(output_subdir, f'person_{person_id}')
            os.makedirs(person_dir, exist_ok=True)
            
            # Save each landmark type for this person
            for landmark_type in ['face_landmarks', 'pose_landmarks', 'pose_world_landmarks', 
                                 'left_hand_landmarks', 'right_hand_landmarks']:
                landmarks_list = person_data[landmark_type]
                if landmarks_list:
                    output_path = os.path.join(person_dir, f'{landmark_type}.mat')
                    save_landmarks(landmarks_list, landmark_type, output_path, log_file)
            
            # Save bounding boxes
            if person_data['bboxes']:
                bbox_output_path = os.path.join(person_dir, 'bboxes.mat')
                frame_nos = np.array([item['frame_no'] for item in person_data['bboxes']])
                bboxes_data = np.array([item['bbox'] for item in person_data['bboxes']])
                
                bbox_dict = {
                    'frame_nos': frame_nos,
                    'bboxes': bboxes_data
                }
                savemat(bbox_output_path, bbox_dict)
                log(f"Saved bounding boxes for person {person_id} to {bbox_output_path}", log_file)
            
            # Save segmentation masks if requested and available
            if save_segmentation and person_data.get('segmentation_masks'):
                try:
                    segmentation_output_path = os.path.join(person_dir, 'segmentation_masks.mat')
                    frame_nos = np.array([item['frame_no'] for item in person_data['segmentation_masks']])
                    
                    # Convert masks to uint8 and compress to save space
                    log(f"Processing {len(person_data['segmentation_masks'])} segmentation masks for person {person_id}...", log_file)
                    masks_data = []
                    for item in person_data['segmentation_masks']:
                        mask = item['mask']
                        if mask is not None:
                            # Convert to binary mask (0 or 255) and compress
                            binary_mask = (mask > 0.5).astype(np.uint8) * 255
                            masks_data.append(binary_mask)
                        else:
                            # Create empty mask if None
                            masks_data.append(np.zeros((480, 640), dtype=np.uint8))  # Smaller placeholder
                    
                    segmentation_dict = {
                        'frame_nos': frame_nos,
                        'segmentation_masks': np.array(masks_data, dtype=np.uint8)
                    }
                    savemat(segmentation_output_path, segmentation_dict)
                    log(f"Saved segmentation masks for person {person_id} to {segmentation_output_path}", log_file)
                except Exception as e:
                    log(f"Warning: Could not save segmentation masks for person {person_id}: {e}", log_file)
            elif person_data.get('segmentation_masks') and not save_segmentation:
                # Estimate memory usage
                num_masks = len(person_data['segmentation_masks'])
                estimated_gb = num_masks * 1920 * 1080 * 4 / (1024**3)
                log(f"Skipping segmentation masks for person {person_id} (would require ~{estimated_gb:.1f} GB, use --save_segmentation to enable)", log_file)
                
    except Exception as e:
        log(f"An error occurred while saving multi-person landmark data: {e}", log_file)
        traceback.print_exc()

def save_landmarks(landmarks, landmark_type, output_path, log_file):
    if not landmarks:
        log(f"No {landmark_type} data to save.", log_file)
        return

    try:
        log(f"Saving {landmark_type} to {output_path}", log_file)

        # Prepare data for savemat
        frame_nos = np.array([item['frame_no'] for item in landmarks])
        landmarks_data = np.array([item['landmarks'] for item in landmarks])

        # Log the shape for debugging
        log(f"frame_nos shape: {frame_nos.shape}, landmarks_data shape: {landmarks_data.shape}", log_file)

        # Create a dictionary suitable for savemat
        mat_dict = {
            'frame_nos': frame_nos,
            'landmarks': landmarks_data
        }

        # Save the dictionary to a .mat file
        savemat(output_path, mat_dict)

        log(f"Finished saving {landmark_type} to {output_path}", log_file)
    except Exception as e:
        log(f"An error occurred while saving {landmark_type} to .mat file: {e}", log_file)
        traceback.print_exc()

def init_worker():
    global holistic
    # Enable refine_face_landmarks to extract iris landmarks and enable_segmentation for person segmentation
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=2,
        refine_face_landmarks=True,
        min_detection_confidence = 0.5,
        enable_segmentation = True,
        min_tracking_confidence = 0.5
    )
    log("Holistic model initialized in worker process")

def extract_landmarks_from_results(results, landmark_type):
    num_landmarks = LANDMARK_COUNTS[landmark_type]
    # Initialize with NaNs; each landmark now has 4 values (x, y, z, precision)
    landmarks_array = np.full((num_landmarks, 4), np.nan)
    if results and getattr(results, landmark_type):
        landmarks = getattr(results, landmark_type).landmark
        for idx, lmk in enumerate(landmarks):
            if idx < num_landmarks:
                # For normalized landmarks, we use lmk.visibility if available; otherwise, NaN
                precision = lmk.visibility if hasattr(lmk, 'visibility') else np.nan
                landmarks_array[idx] = [lmk.x, lmk.y, lmk.z, precision]
    return landmarks_array

def extract_pose_world_landmarks(results):
    # Extract pose landmarks in world coordinates.
    num_landmarks = LANDMARK_COUNTS['pose_landmarks']
    # World landmarks typically don't include visibility, so we add NaN as a placeholder.
    landmarks_array = np.full((num_landmarks, 4), np.nan)
    if results and results.pose_world_landmarks:
        landmarks = results.pose_world_landmarks.landmark
        for idx, lmk in enumerate(landmarks):
            if idx < num_landmarks:
                landmarks_array[idx] = [lmk.x, lmk.y, lmk.z, np.nan]
    return landmarks_array

def extract_landmarks_frame(frame_data):
    try:
        # Initialize holistic model if not already done
        global holistic
        if 'holistic' not in globals() or holistic is None:
            holistic = mp_holistic.Holistic(
                static_image_mode=False,
                model_complexity=2,
                refine_face_landmarks=True,
                min_detection_confidence = 0.5,
                enable_segmentation = True,
                min_tracking_confidence = 0.5
            )
        
        frame_number, frame, person_tracker = frame_data
        
        # Detect people in the frame using YOLO
        people_detections = detect_people_in_frame(frame)
        current_people = person_tracker.update(people_detections)
        
        frame_results = {
            'frame_no': frame_number,
            'people': {}
        }
        
        # If no people detected, return empty results
        if not current_people:
            return frame_results
        
        # Process each person's region separately to extract their landmarks
        for person_id, bbox in current_people.items():
            try:
                # Crop frame around person with padding for better MediaPipe processing
                cropped_frame, crop_coords = crop_person_region(frame, bbox, padding=CROP_PADDING)
                
                if cropped_frame.size == 0 or cropped_frame.shape[0] < 50 or cropped_frame.shape[1] < 50:
                    continue  # Skip if crop is too small for reliable processing
                
                # Convert to RGB for MediaPipe (MediaPipe expects RGB format)
                frame_rgb = cv2.cvtColor(cropped_frame, cv2.COLOR_BGR2RGB)
                
                # Process the cropped frame with MediaPipe Holistic
                # This gives us higher precision as MediaPipe focuses only on the person
                results = holistic.process(frame_rgb)
                
                # Extract landmarks in cropped frame coordinates
                face_landmarks = extract_landmarks_from_results(results, 'face_landmarks')
                pose_landmarks = extract_landmarks_from_results(results, 'pose_landmarks')
                pose_world_landmarks = extract_pose_world_landmarks(results)
                left_hand_landmarks = extract_landmarks_from_results(results, 'left_hand_landmarks')
                right_hand_landmarks = extract_landmarks_from_results(results, 'right_hand_landmarks')
                
                # Convert landmarks from crop-relative to full-frame-relative coordinates
                # This is crucial for accurate overlay visualization
                face_landmarks = adjust_landmarks_to_original_frame(face_landmarks, crop_coords, frame.shape)
                pose_landmarks = adjust_landmarks_to_original_frame(pose_landmarks, crop_coords, frame.shape)
                left_hand_landmarks = adjust_landmarks_to_original_frame(left_hand_landmarks, crop_coords, frame.shape)
                right_hand_landmarks = adjust_landmarks_to_original_frame(right_hand_landmarks, crop_coords, frame.shape)
                # Note: pose_world_landmarks don't need coordinate adjustment as they're in world coordinates
                
                # Store results for this person
                frame_results['people'][person_id] = {
                    'bbox': bbox[:4],  # Store bounding box
                    'crop_coords': crop_coords,
                    'face_landmarks': face_landmarks,
                    'pose_landmarks': pose_landmarks,
                    'pose_world_landmarks': pose_world_landmarks,
                    'left_hand_landmarks': left_hand_landmarks,
                    'right_hand_landmarks': right_hand_landmarks
                }
                
            except Exception as e:
                log(f"Error processing person {person_id} in frame {frame_number}: {e}")
                continue
        
        return frame_results
        
    except Exception as e:
        log(f"Error in worker process: {e}")
        traceback.print_exc()
        return {'frame_no': frame_number, 'people': {}}

def process_frames_in_batches(video_path, total_frames, batch_size, num_processes, log_file, save_segmentation=False):
    cap = cv2.VideoCapture(video_path)
    frame_number = 0
    
    # Initialize person tracker
    person_tracker = PersonTracker()
    
    # Dictionary to store landmarks for each person
    def create_person_dict():
        person_dict = {
            'face_landmarks': [],
            'pose_landmarks': [],
            'pose_world_landmarks': [],
            'left_hand_landmarks': [],
            'right_hand_landmarks': [],
            'bboxes': []
        }
        if save_segmentation:
            person_dict['segmentation_masks'] = []
        return person_dict
    
    people_landmarks = defaultdict(create_person_dict)

    log("Starting sequential processing with person tracking", log_file)
    
    # Initialize holistic model for main process
    global holistic
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=2,
        refine_face_landmarks=True,
        min_detection_confidence = 0.5,
        enable_segmentation = True,
        min_tracking_confidence = 0.5
    )
    log("Holistic model initialized for main process", log_file)
    
    # Since we need to maintain person tracking state, we'll process sequentially
    with tqdm(total=total_frames, desc=f"Processing {os.path.basename(video_path)}", unit="frame") as pbar:
        while frame_number < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Process frame with person tracking
            frame_data = (frame_number, frame, person_tracker)
            frame_results = extract_landmarks_frame(frame_data)
            
            # Store results for each person
            for person_id, person_data in frame_results['people'].items():
                people_landmarks[person_id]['face_landmarks'].append({
                    'frame_no': frame_number,
                    'landmarks': person_data['face_landmarks']
                })
                people_landmarks[person_id]['pose_landmarks'].append({
                    'frame_no': frame_number,
                    'landmarks': person_data['pose_landmarks']
                })
                people_landmarks[person_id]['pose_world_landmarks'].append({
                    'frame_no': frame_number,
                    'landmarks': person_data['pose_world_landmarks']
                })
                people_landmarks[person_id]['left_hand_landmarks'].append({
                    'frame_no': frame_number,
                    'landmarks': person_data['left_hand_landmarks']
                })
                people_landmarks[person_id]['right_hand_landmarks'].append({
                    'frame_no': frame_number,
                    'landmarks': person_data['right_hand_landmarks']
                })
                people_landmarks[person_id]['bboxes'].append({
                    'frame_no': frame_number,
                    'bbox': person_data['bbox']
                })
                # Store segmentation mask if requested and available
                if save_segmentation and person_data.get('segmentation_mask') is not None:
                    people_landmarks[person_id]['segmentation_masks'].append({
                        'frame_no': frame_number,
                        'mask': person_data['segmentation_mask']
                    })
            
            frame_number += 1
            pbar.update(1)
            
            if frame_number % 100 == 0:
                gc.collect()

    cap.release()
    return people_landmarks

def extract_landmarks(video_path, output_subdir, num_processes=None, batch_size=100, save_segmentation=False):
    log_file = os.path.join(output_subdir, 'process_log.txt')
    
    # Check if any person directories already exist
    existing_people = []
    for item in os.listdir(output_subdir) if os.path.exists(output_subdir) else []:
        if os.path.isdir(os.path.join(output_subdir, item)) and item.startswith('person_'):
            existing_people.append(item)
    
    if existing_people:
        log(f"Found existing person directories: {existing_people}. Skipping processing.", log_file)
        return
    
    log(f"Opening video file {video_path}", log_file)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log(f"Error: Unable to open video file {video_path}", log_file)
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    log(f"Total frames in video: {total_frames}", log_file)
    log(f"Starting multi-person landmark extraction", log_file)
    
    # Process frames and extract landmarks for each person
    people_landmarks = process_frames_in_batches(
        video_path, total_frames, batch_size, num_processes, log_file, save_segmentation)
    
    log(f"Finished processing for {video_path}", log_file)
    log(f"Detected {len(people_landmarks)} people in the video", log_file)

    # Get video FPS for duration filtering
    cap_temp = cv2.VideoCapture(video_path)
    fps = cap_temp.get(cv2.CAP_PROP_FPS)
    cap_temp.release()
    
    # Filter out people with less than 5 seconds of data
    filtered_people_landmarks = filter_people_by_duration(people_landmarks, fps, min_duration_seconds=5, log_file=log_file)

    # Save landmarks for each person (only those that passed filtering)
    save_landmarks_multi_person(filtered_people_landmarks, output_subdir, log_file, save_segmentation)

    del people_landmarks, filtered_people_landmarks
    gc.collect()

def process_videos_in_directory(input_dir, output_dir, num_processes=None, batch_size=100, save_segmentation=False):
    files = [f for f in os.listdir(input_dir) if os.path.isfile(os.path.join(input_dir, f))]
    total_files = len(files)

    log(f"Found {total_files} files in the directory {input_dir}")

    for i, file in enumerate(files, start=1):
        log(f"Checking file {i}/{total_files}: {file}")
        file_path = os.path.join(input_dir, file)
        if is_valid_video_file(file_path):
            video_name = os.path.splitext(os.path.basename(file))[0]
            # Create a subdirectory for this video in the output directory
            output_subdir = os.path.join(output_dir, video_name)
            os.makedirs(output_subdir, exist_ok=True)

            # Check if any person directories already exist (skip if so)
            existing_people = []
            for item in os.listdir(output_subdir):
                if os.path.isdir(os.path.join(output_subdir, item)) and item.startswith('person_'):
                    existing_people.append(item)
            
            if existing_people:
                log(f"Found existing person directories for {file}: {existing_people}. Skipping processing.")
                continue

            if file.lower().endswith('.mxf'):
                mp4_path = os.path.join(output_subdir, video_name + '.mp4')
                if not os.path.exists(mp4_path):
                    log(f"Converting {file} to MP4...", log_file=os.path.join(output_subdir, 'process_log.txt'))
                    convert_video_to_mp4(file_path, mp4_path, os.path.join(output_subdir, 'process_log.txt'))
                extract_landmarks(mp4_path, output_subdir, num_processes, batch_size, save_segmentation)
            else:
                extract_landmarks(file_path, output_subdir, num_processes, batch_size, save_segmentation)
        else:
            log(f"Skipping {file} since it is not a valid video file.")

def is_valid_video_file(filepath):
    cap = cv2.VideoCapture(filepath)
    if cap.isOpened():
        cap.release()
        return True
    return False

def main():
    parser = argparse.ArgumentParser(
        description="Extract holistic landmarks from videos in a directory and save them in separate .mat files"
    )
    parser.add_argument('input_dir', type=str, help="Path to the input directory containing video files.")
    parser.add_argument('output_dir', type=str, help="Directory to save the output landmarks.")
    parser.add_argument('--processes', type=int, default=None, help="Number of processes to use for parallel processing.")
    parser.add_argument('--batch_size', type=int, default=100, help="Number of frames to process in each batch.")
    parser.add_argument('--save_segmentation', action='store_true', help="Save segmentation masks (WARNING: uses large amounts of memory)")
    args = parser.parse_args()

    log(f"Starting process with input directory: {args.input_dir} and output directory: {args.output_dir}")
    process_videos_in_directory(args.input_dir, args.output_dir, num_processes=args.processes, batch_size=args.batch_size, save_segmentation=args.save_segmentation)
    log("Processing complete.")

if __name__ == '__main__':
    main()
