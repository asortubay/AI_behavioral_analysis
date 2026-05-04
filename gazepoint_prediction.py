from __future__ import annotations
import os
import importlib.util
import cv2
import csv
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from scipy.io import loadmat
from tqdm import tqdm

from gazelle.model import get_gazelle_model

# MediaPipe Pose landmarks for the head region
HEAD_LANDMARK_INDICES = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
BBOX_PADDING = 0.30  # Expand the tight MediaPipe face points by 30% to capture the whole head

def get_color(index):
    """Returns a distinct BGR color based on the person index."""
    colors = [
        (0, 255, 0),     # Green
        (0, 0, 255),     # Red
        (255, 255, 0),   # Cyan
        (255, 0, 255),   # Magenta
        (0, 255, 255)    # Yellow
    ]
    return colors[index % len(colors)]

def load_labels_and_valid_ids(landmarks_dir):
    """Parses the tracking CSV to get valid IDs and their string labels."""
    csv_path = os.path.join(landmarks_dir, 'people_tracking_summary.csv')
    labels_dict = {}
    valid_ids = set()
    
    if os.path.exists(csv_path):
        with open(csv_path, 'r') as f:
            reader = csv.DictReader(f)
            for row in reader:
                try:
                    primary = int(row['Merged_Person_ID'].strip())
                    valid_ids.add(primary)
                    label_text = row.get('Label', row.get('label', '')).strip()
                    if label_text:
                        labels_dict[primary] = label_text
                except (KeyError, ValueError):
                    continue
    return valid_ids, labels_dict

def load_person_pose_data(landmarks_dir, video_name, valid_ids):
    """Loads pose landmarks for all valid people into a frame-indexed dictionary."""
    video_landmarks_dir = os.path.join(landmarks_dir, video_name, 'merged_landmarks')
    frame_data = {} # Format: {frame_no: {person_id: landmarks_array}}
    
    if not os.path.exists(video_landmarks_dir):
        return frame_data

    for pid in valid_ids:
        pose_path = os.path.join(video_landmarks_dir, f'person_{pid}', 'pose_landmarks.mat')
        if os.path.exists(pose_path):
            try:
                data = loadmat(pose_path)
                frame_nos = data['frame_nos'].flatten()
                landmarks = data['landmarks']
                
                for i, frame_no in enumerate(frame_nos):
                    if frame_no not in frame_data:
                        frame_data[frame_no] = {}
                    frame_data[frame_no][pid] = landmarks[i]
            except Exception as e:
                print(f"Error loading {pose_path}: {e}")
                
    return frame_data

def get_head_bbox(landmarks, width, height):
    """Calculates a padded bounding box from MediaPipe head landmarks."""
    xs, ys = [], []
    for idx in HEAD_LANDMARK_INDICES:
        x_norm, y_norm = landmarks[idx][0], landmarks[idx][1]
        if not np.isnan(x_norm) and not np.isnan(y_norm):
            xs.append(x_norm * width)
            ys.append(y_norm * height)
            
    if not xs or not ys:
        return None
        
    xmin, xmax = min(xs), max(xs)
    ymin, ymax = min(ys), max(ys)
    
    # Apply padding
    w, h = xmax - xmin, ymax - ymin
    xmin = max(0, xmin - BBOX_PADDING * w)
    xmax = min(width, xmax + BBOX_PADDING * w)
    ymin = max(0, ymin - BBOX_PADDING * h)
    ymax = min(height, ymax + BBOX_PADDING * h)
    
    return [xmin, ymin, xmax, ymax]

def process_video_directory(input_dir, landmarks_dir, output_dir, inout_thresh=0.5):
    # 1. Setup Device and Load Models
    print("\n--- System Initialization ---")
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print(f"Device selected: {device.upper()}")

    print("Loading Gaze-LLE model from PyTorch Hub...")
    model, transform = torch.hub.load('fkryan/gazelle', 'gazelle_dinov2_vitl14_inout')
    # model, transform = get_gazelle_model("gazelle_dinov2_vitl14_inout")
    # model.load_gazelle_state_dict(torch.load("gazelle-main\gazelle_dinov2_vitl14_inout_childplay.pt", weights_only=True))

    model = model.to(device)
    model.eval()
    print("Model loaded successfully.")

    os.makedirs(output_dir, exist_ok=True)
    video_extensions = ('.mp4', '.avi', '.mov', '.mkv')
    
    # Filter for valid video files to get a total count
    video_files = [f for f in os.listdir(input_dir) if f.lower().endswith(video_extensions)]
    print(f"Found {len(video_files)} video files in input directory.")

    for i, file_name in enumerate(video_files, 1):
        video_name = os.path.splitext(file_name)[0]
        input_path = os.path.join(input_dir, file_name)
        output_path = os.path.join(output_dir, f"{video_name}_gaze.mp4")
        csv_path = os.path.join(output_dir, f"{video_name}_gaze_data.csv")
        
        print(f"\n{'='*60}")
        print(f"[{i}/{len(video_files)}] Processing Video: {file_name}")
        print(f"{'='*60}")
        
        # Load Labels and Landmark Data
        print("-> Loading tracking CSV and merging landmark data...")
        video_landmarks_base = os.path.join(landmarks_dir, video_name)
        valid_ids, labels_dict = load_labels_and_valid_ids(video_landmarks_base)
        frame_pose_data = load_person_pose_data(landmarks_dir, video_name, valid_ids)

        if not frame_pose_data:
            print(f"-> [SKIPPED] No merged landmarks found for {file_name}.")
            continue
            
        print(f"-> Tracking data loaded for {len(valid_ids)} valid people.")

        cap = cv2.VideoCapture(input_path)
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"-> Video Properties: {width}x{height} @ {fps} FPS | Total Frames: {total_frames}")
        
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        print("-> Starting Gaze Inference & Overlay Generation...")
        # Open CSV Writer
        with open(csv_path, 'w', newline='') as csvfile:
            csv_writer = csv.writer(csvfile)
            csv_writer.writerow(['Frame_No', 'Person_ID', 'Label', 'BBox_XMin', 'BBox_YMin', 'BBox_XMax', 'BBox_YMax', 'InOut_Prob', 'Gaze_X', 'Gaze_Y'])

            frame_count = 0
            
            # Wrap the frame loop in a tqdm progress bar
            with tqdm(total=total_frames, desc=f"Processing Frames", unit="frame") as pbar:
                while cap.isOpened():
                    ret, frame = cap.read()
                    if not ret:
                        break
                        
                    people_in_frame = frame_pose_data.get(frame_count, {})
                    
                    if not people_in_frame:
                        out.write(frame)
                        frame_count += 1
                        pbar.update(1)
                        continue

                    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_image = Image.fromarray(img_rgb)
                    
                    # Prepare bounding boxes for this frame
                    boxes = []
                    person_ids = []
                    norm_bboxes = []
                    
                    for pid, landmarks in people_in_frame.items():
                        bbox = get_head_bbox(landmarks, width, height)
                        if bbox:
                            boxes.append(bbox)
                            person_ids.append(pid)
                            # Normalize to [0, 1] for Gaze-LLE
                            norm_bboxes.append((
                                bbox[0] / float(width), bbox[1] / float(height),
                                bbox[2] / float(width), bbox[3] / float(height)
                            ))

                    if boxes:
                        # 4. Run Gaze-LLE Batch
                        img_tensor = transform(pil_image).unsqueeze(0).to(device)
                        model_input = {"images": img_tensor, "bboxes": [norm_bboxes]}

                        with torch.no_grad():
                            output = model(model_input)

                        predicted_heatmaps = output["heatmap"][0] 
                        predicted_inouts = output["inout"][0]

                        # 5. Draw Overlays & Write CSV
                        for idx, box in enumerate(boxes):
                            pid = person_ids[idx]
                            xmin, ymin, xmax, ymax = box
                            heatmap = predicted_heatmaps[idx]
                            inout_score = predicted_inouts[idx].item()
                            
                            color = get_color(pid - 1)
                            label = labels_dict.get(pid, "")

                            # Calculate Gaze Point
                            gaze_x, gaze_y = -1, -1
                            if inout_score > inout_thresh:
                                heatmap_np = heatmap.detach().cpu().numpy()
                                max_index = np.unravel_index(np.argmax(heatmap_np), heatmap_np.shape)
                                gaze_y = int((max_index[0] / heatmap_np.shape[0]) * height)
                                gaze_x = int((max_index[1] / heatmap_np.shape[1]) * width)
                                
                                face_cx = int((xmin + xmax) / 2)
                                face_cy = int((ymin + ymax) / 2)

                                cv2.line(frame, (face_cx, face_cy), (gaze_x, gaze_y), color, max(2, int(0.005 * min(width, height))))
                                cv2.circle(frame, (gaze_x, gaze_y), max(4, int(0.005 * min(width, height))), color, -1)

                            # Write to CSV
                            csv_writer.writerow([frame_count, pid, label, int(xmin), int(ymin), int(xmax), int(ymax), round(inout_score, 4), gaze_x, gaze_y])

                            # Draw BBox and Label
                            cv2.rectangle(frame, (int(xmin), int(ymin)), (int(xmax), int(ymax)), color, max(2, int(min(width, height) * 0.005)))
                            text = f"P{pid} InOut: {inout_score:.2f}"
                            cv2.putText(frame, text, (int(xmin), int(ymax) + int(height * 0.03)), 
                                        cv2.FONT_HERSHEY_SIMPLEX, min(width, height) * 0.001, color, 2)

                    # 6. Draw Legend
                    if labels_dict:
                        legend_y = 30
                        for pid in sorted(labels_dict.keys()):
                            label_text = f"P{pid}: {labels_dict[pid]}"
                            color = get_color(pid - 1)
                            cv2.rectangle(frame, (width - 250, legend_y - 15), (width - 230, legend_y + 5), color, -1)
                            cv2.putText(frame, label_text, (width - 220 + 2, legend_y + 2), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                            cv2.putText(frame, label_text, (width - 220, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                            legend_y += 30

                    out.write(frame)
                    frame_count += 1
                    pbar.update(1) # Update the progress bar by 1 frame

        cap.release()
        out.release()
        print(f"-> Processing complete for {video_name}.")
        print(f"   Video saved to: {output_path}")
        print(f"   Data saved to:  {csv_path}")

    print("\n--- All Videos Processed Successfully ---")

if __name__ == "__main__":
    INPUT_DIR = r"C:\Users\Aimar\Downloads\Shlomit_data\Data_Parra"
    LANDMARKS_DIR = r"C:\Users\Aimar\Downloads\Shlomit_data\Data_Parra_landmarks"
    OUTPUT_DIR = r"C:\Users\Aimar\Downloads\Shlomit_data\Data_Parra_gaze_prediction"
    
    process_video_directory(INPUT_DIR, LANDMARKS_DIR, OUTPUT_DIR, inout_thresh=0.8)