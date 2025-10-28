"""
Multi-Person Landmark Overlay Tool (Compatible with Segmented Extraction)

This script overlays extracted landmarks from multiple people on the original video.
It reads the per-person landmark data created by mediapipe_segmented_extraction.py
and draws all people's skeletons on the original (unblurred) video.

Key Features:
- Combines landmarks from multiple people
- Color-coded skeleton visualization (one color per person)
- Comprehensive face mesh, pose, and hand landmark drawing
- Optional segmentation mask overlay
- Compatible with output from mediapipe_segmented_extraction.py

Usage:
  python overlay_landmarks_on_video_segmented.py input_videos/ landmarks_dir/ output_dir/
  python overlay_landmarks_on_video_segmented.py input_videos/ landmarks_dir/ output_dir/ --show_segmentation
"""

import os
import sys
import cv2
import numpy as np
import mediapipe as mp
import datetime
import argparse
import traceback
import gc
from scipy.io import loadmat
from tqdm import tqdm
from mediapipe.framework.formats import landmark_pb2

# MediaPipe drawing utilities
mp_drawing = mp.solutions.drawing_utils
mp_drawing_styles = mp.solutions.drawing_styles
mp_holistic = mp.solutions.holistic
mp_face_mesh = mp.solutions.face_mesh

# Define landmark counts
LANDMARK_COUNTS = {
    'face_landmarks': 478,
    'pose_landmarks': 33,
    'left_hand_landmarks': 21,
    'right_hand_landmarks': 21
}

# Define landmark connections
FACE_CONNECTIONS = mp_holistic.FACEMESH_CONTOURS
FACE_TESSELATION = mp_face_mesh.FACEMESH_TESSELATION
FACE_IRISES = mp_face_mesh.FACEMESH_IRISES
POSE_CONNECTIONS = mp_holistic.POSE_CONNECTIONS
HAND_CONNECTIONS = mp_holistic.HAND_CONNECTIONS

# Colors for different people (BGR format)
PERSON_COLORS = [
    (0, 255, 0),    # Green for person 1
    (255, 0, 0),    # Blue for person 2
    (0, 0, 255),    # Red for person 3
    (255, 255, 0),  # Cyan for person 4
    (255, 0, 255),  # Magenta for person 5
    (0, 255, 255),  # Yellow for person 6
]


def log(message, log_file=None):
    """Print timestamped message and optionally write to log file"""
    timestamped_message = f"{datetime.datetime.now()} - {message}"
    print(timestamped_message)
    if log_file:
        with open(log_file, 'a') as f:
            f.write(timestamped_message + '\n')


class MultiPersonLandmarkData:
    """Container for landmark data for all people"""
    def __init__(self):
        self.people_data = {}


class PersonLandmarkData:
    """Container for landmark data for a single person"""
    def __init__(self):
        self.face_landmarks = None
        self.pose_landmarks = None
        self.pose_world_landmarks = None
        self.left_hand_landmarks = None
        self.right_hand_landmarks = None
        self.segmentation_masks = None
        self.face_frame_numbers = None
        self.pose_frame_numbers = None
        self.segmentation_frame_numbers = None
        self.first_appearance_frame = None  # Track when person first appears


def load_multi_person_landmarks_from_mat(landmarks_dir, log_file):
    """Load all landmark data for multiple people from .mat files"""
    multi_person_data = MultiPersonLandmarkData()
    
    # Find all person directories
    person_dirs = []
    if os.path.exists(landmarks_dir):
        for item in os.listdir(landmarks_dir):
            item_path = os.path.join(landmarks_dir, item)
            if os.path.isdir(item_path) and item.startswith('person_'):
                person_dirs.append(item)
    
    if not person_dirs:
        log(f"No person directories found in {landmarks_dir}", log_file)
        return None
    
    log(f"Found {len(person_dirs)} people: {person_dirs}", log_file)
    
    for person_dir in person_dirs:
        try:
            # Extract person ID
            person_id = int(person_dir.split('_')[1])
            person_path = os.path.join(landmarks_dir, person_dir)
            
            # Initialize person data
            person_data = PersonLandmarkData()
            
            # File paths
            face_path = os.path.join(person_path, 'face_landmarks.mat')
            pose_path = os.path.join(person_path, 'pose_landmarks.mat')
            pose_world_path = os.path.join(person_path, 'pose_world_landmarks.mat')
            left_hand_path = os.path.join(person_path, 'left_hand_landmarks.mat')
            right_hand_path = os.path.join(person_path, 'right_hand_landmarks.mat')
            segmentation_path = os.path.join(person_path, 'segmentation_masks.mat')
            
            # Check required files
            required_files = [face_path, pose_path, pose_world_path, left_hand_path, right_hand_path]
            missing_files = [f for f in required_files if not os.path.exists(f)]
            
            if missing_files:
                log(f"Missing files for {person_dir}: {missing_files}", log_file)
                continue
            
            # Load landmark data
            face_data = loadmat(face_path)
            person_data.face_landmarks = face_data['landmarks']
            person_data.face_frame_numbers = face_data.get('frame_nos', None)
            
            # Get first appearance frame from face landmarks
            if person_data.face_frame_numbers is not None and len(person_data.face_frame_numbers) > 0:
                person_data.first_appearance_frame = int(np.min(person_data.face_frame_numbers))
                log(f"  Person {person_id} first appears at frame {person_data.first_appearance_frame}", log_file)
            
            pose_data = loadmat(pose_path)
            person_data.pose_landmarks = pose_data['landmarks']
            person_data.pose_frame_numbers = pose_data.get('frame_nos', None)
            
            pose_world_data = loadmat(pose_world_path)
            person_data.pose_world_landmarks = pose_world_data['landmarks']
            
            left_hand_data = loadmat(left_hand_path)
            person_data.left_hand_landmarks = left_hand_data['landmarks']
            
            right_hand_data = loadmat(right_hand_path)
            person_data.right_hand_landmarks = right_hand_data['landmarks']
            
            # Load segmentation masks if available
            if os.path.exists(segmentation_path):
                try:
                    segmentation_data = loadmat(segmentation_path)
                    person_data.segmentation_masks = segmentation_data['segmentation_masks']
                    person_data.segmentation_frame_numbers = segmentation_data.get('frame_nos', None)
                    log(f"  Loaded segmentation masks for person {person_id}", log_file)
                except Exception as e:
                    log(f"  Warning: Could not load segmentation masks for person {person_id}: {e}", log_file)
                    person_data.segmentation_masks = None
            
            multi_person_data.people_data[person_id] = person_data
            
            log(f"Successfully loaded data for person {person_id}", log_file)
            log(f"  Face landmarks shape: {person_data.face_landmarks.shape}", log_file)
            log(f"  Pose landmarks shape: {person_data.pose_landmarks.shape}", log_file)
            
        except Exception as e:
            log(f"Error loading data for {person_dir}: {e}", log_file)
            continue
    
    if not multi_person_data.people_data:
        log(f"No valid person data found in {landmarks_dir}", log_file)
        return None
    
    return multi_person_data


def create_landmark_list(landmarks_array):
    """Convert landmark array to MediaPipe landmark list format"""
    if landmarks_array is None or np.all(np.isnan(landmarks_array)):
        return None
    
    landmark_list = landmark_pb2.NormalizedLandmarkList()
    
    for landmark in landmarks_array:
        x, y, z, precision = landmark
        if not (np.isnan(x) or np.isnan(y)):
            lm = landmark_list.landmark.add()
            lm.x = float(x)
            lm.y = float(y)
            lm.z = float(z) if not np.isnan(z) else 0.0
            lm.visibility = float(precision) if not np.isnan(precision) else 0.5
        else:
            lm = landmark_list.landmark.add()
            lm.x = 0.0
            lm.y = 0.0
            lm.z = 0.0
            lm.visibility = 0.0
    
    return landmark_list


def draw_landmarks_on_frame_multi_person(frame, multi_person_data, frame_number, 
                                         show_segmentation=False, segmentation_alpha=0.3):
    """Draw landmarks for multiple people on a single frame"""
    if not multi_person_data or not multi_person_data.people_data:
        return frame
    
    frame_height, frame_width = frame.shape[:2]
    
    for person_id, person_data in multi_person_data.people_data.items():
        # Skip if person hasn't appeared yet
        if person_data.first_appearance_frame is not None and frame_number < person_data.first_appearance_frame:
            continue
        
        # Get color for this person
        color_idx = (person_id - 1) % len(PERSON_COLORS)
        person_color = PERSON_COLORS[color_idx]
        
        # Find landmark data for this frame
        face_landmarks = None
        pose_landmarks = None
        left_hand_landmarks = None
        right_hand_landmarks = None
        segmentation_mask = None
        
        # Use frame numbers if available
        if person_data.face_frame_numbers is not None:
            frame_indices = np.where(person_data.face_frame_numbers.flatten() == frame_number)[0]
            if len(frame_indices) > 0:
                idx = frame_indices[0]
                if idx < len(person_data.face_landmarks):
                    face_landmarks = person_data.face_landmarks[idx]
                    pose_landmarks = person_data.pose_landmarks[idx] if idx < len(person_data.pose_landmarks) else None
                    left_hand_landmarks = person_data.left_hand_landmarks[idx] if idx < len(person_data.left_hand_landmarks) else None
                    right_hand_landmarks = person_data.right_hand_landmarks[idx] if idx < len(person_data.right_hand_landmarks) else None
                    
                    # Get segmentation mask if available
                    if person_data.segmentation_masks is not None and person_data.segmentation_frame_numbers is not None:
                        seg_indices = np.where(person_data.segmentation_frame_numbers.flatten() == frame_number)[0]
                        if len(seg_indices) > 0 and seg_indices[0] < len(person_data.segmentation_masks):
                            segmentation_mask = person_data.segmentation_masks[seg_indices[0]]
        else:
            # Fall back to sequential indexing
            if frame_number < len(person_data.face_landmarks):
                face_landmarks = person_data.face_landmarks[frame_number]
                pose_landmarks = person_data.pose_landmarks[frame_number] if frame_number < len(person_data.pose_landmarks) else None
                left_hand_landmarks = person_data.left_hand_landmarks[frame_number] if frame_number < len(person_data.left_hand_landmarks) else None
                right_hand_landmarks = person_data.right_hand_landmarks[frame_number] if frame_number < len(person_data.right_hand_landmarks) else None
                
                if person_data.segmentation_masks is not None and frame_number < len(person_data.segmentation_masks):
                    segmentation_mask = person_data.segmentation_masks[frame_number]
        
        # Skip if no data for this frame
        if face_landmarks is None:
            continue
        
        # Draw segmentation mask if available and requested
        if show_segmentation and segmentation_mask is not None:
            try:
                if segmentation_mask.shape[:2] == (frame_height, frame_width):
                    colored_mask = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
                    mask_binary = (segmentation_mask > 0.5).astype(np.uint8)
                    colored_mask[mask_binary == 1] = person_color
                    frame = cv2.addWeighted(frame, 1.0, colored_mask, segmentation_alpha, 0)
                elif segmentation_mask.size > 0:
                    resized_mask = cv2.resize(segmentation_mask.astype(np.float32),
                                            (frame_width, frame_height),
                                            interpolation=cv2.INTER_LINEAR)
                    colored_mask = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)
                    mask_binary = (resized_mask > 0.5).astype(np.uint8)
                    colored_mask[mask_binary == 1] = person_color
                    frame = cv2.addWeighted(frame, 1.0, colored_mask, segmentation_alpha, 0)
            except Exception as e:
                print(f"Error drawing segmentation mask for person {person_id}: {e}")
        
        # Convert to MediaPipe format
        face_lm = create_landmark_list(face_landmarks)
        pose_lm = create_landmark_list(pose_landmarks)
        left_hand_lm = create_landmark_list(left_hand_landmarks)
        right_hand_lm = create_landmark_list(right_hand_landmarks)
        
        # Create drawing specs
        face_landmark_spec = mp_drawing.DrawingSpec(color=person_color, thickness=1, circle_radius=1)
        face_connection_spec = mp_drawing.DrawingSpec(color=person_color, thickness=1)
        face_mesh_spec = mp_drawing.DrawingSpec(color=(person_color[0]//2, person_color[1]//2, person_color[2]//2), thickness=1)
        iris_spec = mp_drawing.DrawingSpec(color=person_color, thickness=2, circle_radius=1)
        
        pose_landmark_spec = mp_drawing.DrawingSpec(color=person_color, thickness=3, circle_radius=3)
        pose_connection_spec = mp_drawing.DrawingSpec(color=person_color, thickness=3)
        
        hand_landmark_spec = mp_drawing.DrawingSpec(color=person_color, thickness=2, circle_radius=2)
        hand_connection_spec = mp_drawing.DrawingSpec(color=person_color, thickness=2)
        
        # Draw face landmarks
        if face_lm and len(face_lm.landmark) > 0:
            try:
                mp_drawing.draw_landmarks(
                    frame, face_lm, FACE_TESSELATION,
                    landmark_drawing_spec=None,
                    connection_drawing_spec=face_mesh_spec
                )
                mp_drawing.draw_landmarks(
                    frame, face_lm, FACE_CONNECTIONS,
                    landmark_drawing_spec=face_landmark_spec,
                    connection_drawing_spec=face_connection_spec
                )
                mp_drawing.draw_landmarks(
                    frame, face_lm, FACE_IRISES,
                    landmark_drawing_spec=iris_spec,
                    connection_drawing_spec=iris_spec
                )
            except Exception as e:
                print(f"Error drawing face landmarks for person {person_id}: {e}")
        
        # Draw pose landmarks
        if pose_lm and len(pose_lm.landmark) > 0:
            try:
                mp_drawing.draw_landmarks(
                    frame, pose_lm, POSE_CONNECTIONS,
                    landmark_drawing_spec=pose_landmark_spec,
                    connection_drawing_spec=pose_connection_spec
                )
            except Exception as e:
                print(f"Error drawing pose landmarks for person {person_id}: {e}")
        
        # Draw hand landmarks
        if left_hand_lm and len(left_hand_lm.landmark) > 0:
            try:
                mp_drawing.draw_landmarks(
                    frame, left_hand_lm, HAND_CONNECTIONS,
                    landmark_drawing_spec=hand_landmark_spec,
                    connection_drawing_spec=hand_connection_spec
                )
            except Exception as e:
                print(f"Error drawing left hand landmarks for person {person_id}: {e}")
        
        if right_hand_lm and len(right_hand_lm.landmark) > 0:
            try:
                mp_drawing.draw_landmarks(
                    frame, right_hand_lm, HAND_CONNECTIONS,
                    landmark_drawing_spec=hand_landmark_spec,
                    connection_drawing_spec=hand_connection_spec
                )
            except Exception as e:
                print(f"Error drawing right hand landmarks for person {person_id}: {e}")
    
    # Add indicator if segmentation is enabled
    if show_segmentation:
        cv2.putText(frame, "SEGMENTATION ON", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
    
    return frame


def overlay_landmarks_on_video(video_path, landmarks_dir, output_path, log_file,
                               show_segmentation=False, segmentation_alpha=0.3):
    """Overlay landmarks on video and save output"""
    
    # Load landmark data
    multi_person_data = load_multi_person_landmarks_from_mat(landmarks_dir, log_file)
    if multi_person_data is None:
        log(f"Failed to load landmark data for {video_path}", log_file)
        return False
    
    # Open input video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log(f"Error: Unable to open video file {video_path}", log_file)
        return False
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    log(f"Video properties - FPS: {fps}, Width: {width}, Height: {height}, Frames: {total_frames}", log_file)
    log(f"Found {len(multi_person_data.people_data)} people in landmark data", log_file)
    
    # Create VideoWriter
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    
    if not out.isOpened():
        log(f"Error: Unable to create output video writer for {output_path}", log_file)
        cap.release()
        return False
    
    frame_number = 0
    log(f"Starting to overlay landmarks on {os.path.basename(video_path)}", log_file)
    
    # Process all frames
    with tqdm(total=total_frames, desc=f"Overlaying landmarks", unit="frame") as pbar:
        while frame_number < total_frames:
            ret, frame = cap.read()
            if not ret:
                break
            
            try:
                # Overlay landmarks
                frame_with_landmarks = draw_landmarks_on_frame_multi_person(
                    frame, multi_person_data, frame_number, show_segmentation, segmentation_alpha)
                
                # Write frame
                out.write(frame_with_landmarks)
                
                frame_number += 1
                pbar.update(1)
                
            except Exception as e:
                log(f"Error processing frame {frame_number}: {e}", log_file)
                frame_number += 1
                pbar.update(1)
                continue
    
    cap.release()
    out.release()
    
    log(f"Successfully created overlay video: {output_path}", log_file)
    return True


def find_video_files(directory):
    """Find all video files in directory"""
    video_extensions = ['.mp4', '.avi', '.mov', '.mkv', '.mxf', '.webm', '.flv']
    video_files = []
    
    for file in os.listdir(directory):
        if any(file.lower().endswith(ext) for ext in video_extensions):
            video_files.append(file)
    
    return video_files


def process_videos_in_directory(input_dir, landmarks_dir, output_dir,
                                show_segmentation=False, segmentation_alpha=0.3):
    """Process all videos that have landmark data"""
    os.makedirs(output_dir, exist_ok=True)
    
    video_files = find_video_files(input_dir)
    
    if not video_files:
        log("No video files found in input directory")
        return
    
    log(f"Found {len(video_files)} video files")
    
    processed_count = 0
    skipped_count = 0
    
    for video_file in video_files:
        video_path = os.path.join(input_dir, video_file)
        video_name = os.path.splitext(video_file)[0]
        
        # Check if landmarks directory exists
        landmarks_subdir = os.path.join(landmarks_dir, video_name)
        
        if not os.path.exists(landmarks_subdir):
            log(f"No landmarks directory found for {video_file}. Skipping.")
            skipped_count += 1
            continue
        
        # Check for person directories
        person_dirs = []
        for item in os.listdir(landmarks_subdir):
            item_path = os.path.join(landmarks_subdir, item)
            if os.path.isdir(item_path) and item.startswith('person_'):
                person_dirs.append(item)
        
        if not person_dirs:
            log(f"No person directories found for {video_file}. Skipping.")
            skipped_count += 1
            continue
        
        log(f"Found {len(person_dirs)} people for {video_file}: {person_dirs}")
        
        # Define output path
        output_path = os.path.join(output_dir, f"{video_name}_with_landmarks.mp4")
        
        # Check if already exists
        if os.path.exists(output_path):
            log(f"Overlay video already exists for {video_file}. Skipping.")
            skipped_count += 1
            continue
        
        # Create log file
        log_file = os.path.join(landmarks_subdir, 'overlay_log.txt')
        
        log(f"Processing {video_file}...")
        
        # Process video
        success = overlay_landmarks_on_video(video_path, landmarks_subdir, output_path, log_file,
                                            show_segmentation, segmentation_alpha)
        
        if success:
            processed_count += 1
            log(f"Successfully processed {video_file}")
        else:
            log(f"Failed to process {video_file}")
            skipped_count += 1
        
        gc.collect()
    
    log(f"Processing complete. Processed: {processed_count}, Skipped: {skipped_count}")


def main():
    parser = argparse.ArgumentParser(
        description="Overlay extracted landmarks on original videos"
    )
    parser.add_argument('input_dir', type=str,
                       help="Path to directory containing original videos")
    parser.add_argument('landmarks_dir', type=str,
                       help="Path to directory containing landmark output subdirectories")
    parser.add_argument('output_dir', type=str,
                       help="Directory to save overlay videos")
    parser.add_argument('--single_video', type=str, default=None,
                       help="Process only a single video file")
    parser.add_argument('--show_segmentation', action='store_true',
                       help="Overlay segmentation masks (if available)")
    parser.add_argument('--segmentation_alpha', type=float, default=0.3,
                       help="Transparency of segmentation overlay (0.0 to 1.0, default: 0.3)")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.input_dir):
        log(f"Input directory does not exist: {args.input_dir}")
        return
    
    if not os.path.exists(args.landmarks_dir):
        log(f"Landmarks directory does not exist: {args.landmarks_dir}")
        return
    
    log(f"Starting overlay process...")
    log(f"Input videos directory: {args.input_dir}")
    log(f"Landmarks directory: {args.landmarks_dir}")
    log(f"Output directory: {args.output_dir}")
    
    if args.single_video:
        video_path = os.path.join(args.input_dir, args.single_video)
        video_name = os.path.splitext(args.single_video)[0]
        landmarks_subdir = os.path.join(args.landmarks_dir, video_name)
        
        os.makedirs(args.output_dir, exist_ok=True)
        output_path = os.path.join(args.output_dir, f"{video_name}_with_landmarks.mp4")
        log_file = os.path.join(landmarks_subdir, 'overlay_log.txt')
        
        overlay_landmarks_on_video(video_path, landmarks_subdir, output_path, log_file,
                                  args.show_segmentation, args.segmentation_alpha)
    else:
        process_videos_in_directory(args.input_dir, args.landmarks_dir, args.output_dir,
                                   args.show_segmentation, args.segmentation_alpha)
    
    log("Overlay process complete.")


if __name__ == '__main__':
    main()
