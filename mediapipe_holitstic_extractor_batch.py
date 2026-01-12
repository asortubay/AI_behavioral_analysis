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

# Initialize MediaPipe holistic model in the main process
mp_holistic = mp.solutions.holistic

# face landmarks count is 478 including iris landmarks (468 + 10 extra iris points)
LANDMARK_COUNTS = {
    'face_landmarks': 478,
    'pose_landmarks': 33,
    'left_hand_landmarks': 21,
    'right_hand_landmarks': 21
}

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
    # Enable refine_face_landmarks to extract iris landmarks
    holistic = mp_holistic.Holistic(
        static_image_mode=False,
        model_complexity=2,
        refine_face_landmarks=True
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
        global holistic
        frame_number, frame = frame_data
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = holistic.process(frame_rgb)

        face_landmarks = extract_landmarks_from_results(results, 'face_landmarks')
        pose_landmarks = extract_landmarks_from_results(results, 'pose_landmarks')
        # Extract world coordinates for pose landmarks
        pose_world_landmarks = extract_pose_world_landmarks(results)
        left_hand_landmarks = extract_landmarks_from_results(results, 'left_hand_landmarks')
        right_hand_landmarks = extract_landmarks_from_results(results, 'right_hand_landmarks')

        return {
            'frame_no': frame_number,
            'face_landmarks': face_landmarks,
            'pose_landmarks': pose_landmarks,
            'pose_world_landmarks': pose_world_landmarks,
            'left_hand_landmarks': left_hand_landmarks,
            'right_hand_landmarks': right_hand_landmarks
        }
    except Exception as e:
        log(f"Error in worker process: {e}")
        traceback.print_exc()
        return {}

def process_frames_in_batches(video_path, total_frames, batch_size, num_processes, log_file):
    cap = cv2.VideoCapture(video_path)
    frame_number = 0
    face_landmarks_list = []
    pose_landmarks_list = []
    pose_world_landmarks_list = []
    left_hand_landmarks_list = []
    right_hand_landmarks_list = []

    log("Starting multiprocessing pool", log_file)
    with mp_process.Pool(processes=num_processes, initializer=init_worker) as pool:
        with tqdm(total=total_frames, desc=f"Processing {os.path.basename(video_path)}", unit="frame") as pbar:
            while frame_number < total_frames:
                frames = []
                for _ in range(batch_size):
                    ret, frame = cap.read()
                    if not ret:
                        break
                    frames.append((frame_number, frame))
                    frame_number += 1

                if not frames:
                    log("No more frames to process. Breaking out of the loop.", log_file)
                    break

                batch_landmarks = pool.map(extract_landmarks_frame, frames, chunksize=1)
                
                for landmarks in batch_landmarks:
                    face_landmarks_list.append({
                        'frame_no': landmarks['frame_no'],
                        'landmarks': landmarks['face_landmarks']
                    })
                    pose_landmarks_list.append({
                        'frame_no': landmarks['frame_no'],
                        'landmarks': landmarks['pose_landmarks']
                    })
                    pose_world_landmarks_list.append({
                        'frame_no': landmarks['frame_no'],
                        'landmarks': landmarks['pose_world_landmarks']
                    })
                    left_hand_landmarks_list.append({
                        'frame_no': landmarks['frame_no'],
                        'landmarks': landmarks['left_hand_landmarks']
                    })
                    right_hand_landmarks_list.append({
                        'frame_no': landmarks['frame_no'],
                        'landmarks': landmarks['right_hand_landmarks']
                    })

                pbar.update(len(frames))
                gc.collect()

    cap.release()
    return (face_landmarks_list, pose_landmarks_list, pose_world_landmarks_list,
            left_hand_landmarks_list, right_hand_landmarks_list)

def extract_landmarks(video_path, output_subdir, num_processes=None, batch_size=100):
    log_file = os.path.join(output_subdir, 'process_log.txt')
    
    # Define paths for the separate landmark files
    face_output_path = os.path.join(output_subdir, 'face_landmarks.mat')
    pose_output_path = os.path.join(output_subdir, 'pose_landmarks.mat')
    pose_world_output_path = os.path.join(output_subdir, 'pose_world_landmarks.mat')
    left_hand_output_path = os.path.join(output_subdir, 'left_hand_landmarks.mat')
    right_hand_output_path = os.path.join(output_subdir, 'right_hand_landmarks.mat')

    # Check if all landmark files exist in the output directory
    if (os.path.exists(face_output_path) and os.path.exists(pose_output_path) and 
        os.path.exists(pose_world_output_path) and os.path.exists(left_hand_output_path) and 
        os.path.exists(right_hand_output_path)):
        log(f"All landmark files already exist for {os.path.basename(video_path)}. Skipping processing.", log_file)
        return
    
    log(f"Opening video file {video_path}", log_file)
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        log(f"Error: Unable to open video file {video_path}", log_file)
        return

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    cap.release()

    log(f"Total frames in video: {total_frames}", log_file)
    log(f"Starting multiprocessing with {num_processes} processes", log_file)
    
    (face_landmarks, pose_landmarks, pose_world_landmarks,
     left_hand_landmarks, right_hand_landmarks) = process_frames_in_batches(
         video_path, total_frames, batch_size, num_processes, log_file)
    
    log(f"Finished multiprocessing for {video_path}", log_file)

    save_landmarks(face_landmarks, 'face_landmarks', face_output_path, log_file)
    save_landmarks(pose_landmarks, 'pose_landmarks', pose_output_path, log_file)
    save_landmarks(pose_world_landmarks, 'pose_world_landmarks', pose_world_output_path, log_file)
    save_landmarks(left_hand_landmarks, 'left_hand_landmarks', left_hand_output_path, log_file)
    save_landmarks(right_hand_landmarks, 'right_hand_landmarks', right_hand_output_path, log_file)

    del face_landmarks, pose_landmarks, pose_world_landmarks, left_hand_landmarks, right_hand_landmarks
    gc.collect()

def process_videos_in_directory(input_dir, output_dir, num_processes=None, batch_size=100):
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

            face_output_path = os.path.join(output_subdir, 'face_landmarks.mat')
            pose_output_path = os.path.join(output_subdir, 'pose_landmarks.mat')
            pose_world_output_path = os.path.join(output_subdir, 'pose_world_landmarks.mat')
            left_hand_output_path = os.path.join(output_subdir, 'left_hand_landmarks.mat')
            right_hand_output_path = os.path.join(output_subdir, 'right_hand_landmarks.mat')

            if (os.path.exists(face_output_path) and os.path.exists(pose_output_path) and 
                os.path.exists(pose_world_output_path) and os.path.exists(left_hand_output_path) and 
                os.path.exists(right_hand_output_path)):
                log(f"All landmark files already exist for {file}. Skipping processing.")
                continue

            if file.lower().endswith('.mxf'):
                mp4_path = os.path.join(output_subdir, video_name + '.mp4')
                if not os.path.exists(mp4_path):
                    log(f"Converting {file} to MP4...", log_file=os.path.join(output_subdir, 'process_log.txt'))
                    convert_video_to_mp4(file_path, mp4_path, os.path.join(output_subdir, 'process_log.txt'))
                extract_landmarks(mp4_path, output_subdir, num_processes, batch_size)
            else:
                extract_landmarks(file_path, output_subdir, num_processes, batch_size)
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
    args = parser.parse_args()

    log(f"Starting process with input directory: {args.input_dir} and output directory: {args.output_dir}")
    process_videos_in_directory(args.input_dir, args.output_dir, num_processes=args.processes, batch_size=args.batch_size)
    log("Processing complete.")

if __name__ == '__main__':
    main()
