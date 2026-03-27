import cv2
import mediapipe as mp
import os
import argparse
import subprocess
from tqdm import tqdm
import sys

def get_args():
    parser = argparse.ArgumentParser(description='Batch process videos to blur faces while preserving audio.')
    parser.add_argument('--input_dir', type=str, required=True, help='Directory containing input videos')
    parser.add_argument('--output_dir', type=str, required=True, help='Directory to save processed videos')
    parser.add_argument('--model_selection', type=int, default=1, help='0 for short-range (within 2 meters), 1 for full-range (within 5 meters)')
    parser.add_argument('--min_detection_confidence', type=float, default=0.01, help='Minimum confidence value ([0.0, 1.0]) for face detection to be considered successful.')
    return parser.parse_args()

def is_valid_video_file(filepath):
    """Check if the file is a valid video file using OpenCV."""
    try:
        cap = cv2.VideoCapture(filepath)
        if cap.isOpened():
            ret, _ = cap.read()
            cap.release()
            return ret
        return False
    except Exception:
        return False

def blur_face(image, detection, mp_face_detection):
    """Apply Gaussian blur to the detected face region."""
    h, w, _ = image.shape
    bboxC = detection.location_data.relative_bounding_box
    
    # Calculate pixel coordinates
    x = int(bboxC.xmin * w)
    y = int(bboxC.ymin * h)
    w_box = int(bboxC.width * w)
    h_box = int(bboxC.height * h)

    # Expand the bounding box
    expansion_factor = 0.5
    padding_x = int(w_box * expansion_factor)
    padding_y = int(h_box * expansion_factor)
    
    x1 = x - padding_x
    y1 = y - padding_y
    x2 = x + w_box + padding_x
    y2 = y + h_box + padding_y

    # Clamp coordinates to image bounds
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    
    # Recalculate x, y, width, height for ROI extraction
    x = x1
    y = y1
    width = x2 - x1
    height = y2 - y1
    
    if width > 0 and height > 0:
        # Extract the region of interest (ROI)
        roi = image[y:y+height, x:x+width]
        
        # Apply Gaussian blur
        # Kernel size (ksize) should be odd and positive. 
        # Adjust sigmaX based on the size of the ROI for stronger blur on larger faces
        ksize = (99, 99) 
        blurred_roi = cv2.GaussianBlur(roi, ksize, 30)
        
        # Place the blurred ROI back into the image
        image[y:y+height, x:x+width] = blurred_roi
        
    return image

def process_video(input_path, output_path, args):
    """Process a single video: detect faces, blur them, and save."""
    
    mp_face_detection = mp.solutions.face_detection
    
    # Initialize MediaPipe Face Detection
    with mp_face_detection.FaceDetection(
        model_selection=args.model_selection, 
        min_detection_confidence=args.min_detection_confidence) as face_detection:
        
        cap = cv2.VideoCapture(input_path)
        if not cap.isOpened():
            print(f"Error: Could not open video {input_path}")
            return

        # Get video properties
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Temporary output path for video without audio
        temp_output_path = output_path.replace('.mp4', '_temp.mp4')
        if temp_output_path == output_path:
             temp_output_path = output_path + '_temp.mp4'

        # Initialize VideoWriter
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(temp_output_path, fourcc, fps, (width, height))
        
        print(f"Processing: {os.path.basename(input_path)}")
        
        with tqdm(total=total_frames, unit="frames") as pbar:
            while cap.isOpened():
                success, image = cap.read()
                if not success:
                    break

                # Convert the BGR image to RGB
                image.flags.writeable = False
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = face_detection.process(image)

                # Draw the face detection annotations on the image.
                image.flags.writeable = True
                image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
                
                if results.detections:
                    for detection in results.detections:
                        image = blur_face(image, detection, mp_face_detection)
                
                # Write the frame
                out.write(image)
                pbar.update(1)

        cap.release()
        out.release()
        
        # Merge audio from original video
        merge_audio(input_path, temp_output_path, output_path)
        
        # Cleanup temporary file
        if os.path.exists(temp_output_path):
            os.remove(temp_output_path)

def merge_audio(input_video, video_no_audio, output_video):
    """Merge audio from input_video into video_no_audio using ffmpeg."""
    try:
        # Check if ffmpeg is installed
        subprocess.run(['ffmpeg', '-version'], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        
        # Using -c:v copy to copy video stream without re-encoding
        # Using -c:a aac to re-encode audio to AAC (or copy if compatible)
        # -map 0:a selects audio from the first input (input_video)
        # -map 1:v selects video from the second input (video_no_audio)
        # -shortest ensures output length matches the shortest stream (usually video)
        
        cmd = [
            'ffmpeg', '-y', # Overwrite output files
            '-i', input_video,
            '-i', video_no_audio,
            '-c:v', 'copy',
            '-c:a', 'aac', 
            '-map', '0:a',
            '-map', '1:v',
            '-shortest',
            output_video
        ]
        
        subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, check=True)
        # print("Audio merged successfully.")
        
    except subprocess.CalledProcessError:
        print("Warning: FFmpeg failed or not found. Output video might not have audio.")
        # Fallback: just rename the temp video if audio merge fails
        if os.path.exists(video_no_audio):
             import shutil
             shutil.copy(video_no_audio, output_video)
    except FileNotFoundError:
        print("Error: FFmpeg not found in system path. Please install FFmpeg to preserve audio.")
        if os.path.exists(video_no_audio):
             import shutil
             shutil.copy(video_no_audio, output_video)

def main():
    args = get_args()
    
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        
    files = [f for f in os.listdir(args.input_dir) if os.path.isfile(os.path.join(args.input_dir, f))]
    
    for file in files:
        input_path = os.path.join(args.input_dir, file)
        
        if is_valid_video_file(input_path):
            # Construct output filename
            filename, ext = os.path.splitext(file)
            output_filename = f"{filename}.mp4" # Force mp4 for now
            output_path = os.path.join(args.output_dir, output_filename)
            
            if os.path.exists(output_path):
                print(f"Skipping existing file: {output_filename}")
                continue

            process_video(input_path, output_path, args)
        else:
             print(f"Skipping non-video file: {file}")

if __name__ == "__main__":
    main()
