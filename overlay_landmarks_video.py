import os
import argparse
import cv2
import numpy as np
from scipy.io import loadmat
import mediapipe as mp

FACE_CONNECTIONS = mp.solutions.face_mesh.FACEMESH_TESSELATION
POSE_CONNECTIONS = mp.solutions.pose.POSE_CONNECTIONS
HAND_CONNECTIONS = mp.solutions.hands.HAND_CONNECTIONS


def load_landmark_file(path):
    if not os.path.exists(path):
        return None
    data = loadmat(path)
    frames = data.get('frame_nos')
    landmarks = data.get('landmarks')
    if frames is None or landmarks is None:
        return None
    frames = np.atleast_1d(np.squeeze(frames)).astype(int)
    landmarks = np.array(landmarks, dtype=np.float32)

    # MATLAB savemat can wrap dims; squeeze down to (frames, landmarks, 4)
    landmarks = np.squeeze(landmarks)
    if landmarks.ndim == 4 and 1 in landmarks.shape:
        landmarks = landmarks.reshape([-1 if s != 1 else 1 for s in landmarks.shape])
        landmarks = np.squeeze(landmarks)
    if landmarks.ndim == 2:
        # Single frame stored as (landmarks, 4)
        landmarks = landmarks[np.newaxis, ...]
    if landmarks.ndim != 3:
        return None
    if landmarks.shape[0] != len(frames):
        return None

    frame_to_landmarks = {int(frames[i]): landmarks[i] for i in range(len(frames))}
    return frame_to_landmarks


def visibility_color(visibility):
    if np.isnan(visibility):
        visibility = 0.0
    visibility = float(np.clip(visibility, 0.0, 1.0))
    return (0, int(visibility * 255), int((1.0 - visibility) * 255))


def draw_landmarks_set(frame, lmk_array, connections=None, min_visibility=0.5):
    if lmk_array is None:
        return
    h, w, _ = frame.shape
    pixels = []
    for lmk in lmk_array:
        x, y, z, vis = lmk
        if np.isnan(x) or np.isnan(y) or np.isnan(vis) or vis < min_visibility:
            pixels.append(None)
            continue
        px, py = int(x * w), int(y * h)
        color = visibility_color(vis)
        cv2.circle(frame, (px, py), 2, color, -1, lineType=cv2.LINE_AA)
        pixels.append((px, py))
    if connections:
        for a, b in connections:
            if a < len(pixels) and b < len(pixels):
                pa, pb = pixels[a], pixels[b]
                if pa is not None and pb is not None:
                    cv2.line(frame, pa, pb, (180, 180, 180), 1, lineType=cv2.LINE_AA)


def overlay_landmarks_on_video(video_path, landmarks_dir, output_path=None, min_visibility=0.5):
    face_map = load_landmark_file(os.path.join(landmarks_dir, 'face_landmarks.mat'))
    pose_map = load_landmark_file(os.path.join(landmarks_dir, 'pose_landmarks.mat'))
    left_hand_map = load_landmark_file(os.path.join(landmarks_dir, 'left_hand_landmarks.mat'))
    right_hand_map = load_landmark_file(os.path.join(landmarks_dir, 'right_hand_landmarks.mat'))

    if output_path is None:
        base, ext = os.path.splitext(video_path)
        output_path = f"{base}_overlay{ext or '.mp4'}"

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

    frame_idx = 0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame is None:
            frame_idx += 1
            continue
        if frame.ndim == 2:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif frame.ndim == 3 and frame.shape[2] == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        face_lmk = face_map.get(frame_idx) if face_map else None
        pose_lmk = pose_map.get(frame_idx) if pose_map else None
        left_hand_lmk = left_hand_map.get(frame_idx) if left_hand_map else None
        right_hand_lmk = right_hand_map.get(frame_idx) if right_hand_map else None

        draw_landmarks_set(frame, face_lmk, FACE_CONNECTIONS, min_visibility)
        draw_landmarks_set(frame, pose_lmk, POSE_CONNECTIONS, min_visibility)
        draw_landmarks_set(frame, left_hand_lmk, HAND_CONNECTIONS, min_visibility)
        draw_landmarks_set(frame, right_hand_lmk, HAND_CONNECTIONS, min_visibility)

        out.write(frame)
        frame_idx += 1

    cap.release()
    out.release()
    return output_path


def is_valid_video_file(filepath):
    cap = cv2.VideoCapture(filepath)
    if cap.isOpened():
        cap.release()
        return True
    return False


def process_videos_in_directory(videos_dir, landmarks_root, output_dir=None, min_visibility=0.5):
    output_dir = output_dir or videos_dir
    os.makedirs(output_dir, exist_ok=True)

    files = [f for f in os.listdir(videos_dir) if os.path.isfile(os.path.join(videos_dir, f))]
    for file in files:
        video_path = os.path.join(videos_dir, file)
        if not is_valid_video_file(video_path):
            continue

        video_name, _ = os.path.splitext(file)
        landmarks_dir = os.path.join(landmarks_root, video_name)
        if not os.path.isdir(landmarks_dir):
            continue

        out_path = os.path.join(output_dir, f"{video_name}_overlay.mp4")
        if os.path.exists(out_path):
            continue

        overlay_landmarks_on_video(video_path, landmarks_dir, out_path, min_visibility)


def main():
    parser = argparse.ArgumentParser(description='Overlay saved landmarks on one video or a directory of videos.')
    parser.add_argument('input_path', help='Path to a video file or a directory of videos')
    parser.add_argument('landmarks_root', help='Root directory containing per-video landmark subfolders')
    parser.add_argument('--output_dir', help='Where to write overlay videos (defaults to input directory)')
    parser.add_argument('--output', help='Output video path (only used when input_path is a single video)')
    parser.add_argument('--min_visibility', type=float, default=0, help='Only draw landmarks/connections with visibility >= this value')
    args = parser.parse_args()

    if os.path.isdir(args.input_path):
        process_videos_in_directory(args.input_path, args.landmarks_root, args.output_dir, args.min_visibility)
        print(f"Finished overlays in {args.output_dir or args.input_path}")
    else:
        output_path = overlay_landmarks_on_video(args.input_path, args.landmarks_root, args.output, args.min_visibility)
        print(f"Saved overlay video to {output_path}")


if __name__ == '__main__':
    main()
