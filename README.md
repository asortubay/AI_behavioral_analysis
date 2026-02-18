
# AI Behavioral Analysis Pipeline

This repository contains scripts for processing interview audio/video recordings to extract behavioral measures for digital phenotyping research. The current codebase focuses on video synchronization and Mediapipe landmark extraction/overlay.

Note that some scripts may require modification based on specific dataset structures, interview contents (e.g., different questions), and desired output features, but the core functionality is provided.

**Paper:**
Overlap and Differences of Autism and ADHD: Digital Phenotyping of Movement and Communication During Development

Aimar Silvan, Adriana Di Martino, Michael Milham, Lucas C Parra, Jens Madsen

doi: https://doi.org/10.1101/2025.10.20.682864 


## Pipeline Overview

### 1. Video Synchronization
- **`sync_webcam_videos.py`**: Synchronizes two webcam videos using a visible and audible "clap" (sharp audio transient) recorded at the beginning of the session. The script detects the clap onset in each audio track, aligns them, and exports frame-accurately trimmed videos with identical frame counts.

### 2. Landmark Extraction
- **`mediapipe_holitstic_extractor_batch.py`**: Extracts face and body landmarks from video recordings using Google Mediapipe, batching frames for optimized processing
- **`mediapipe_holistic_extractor_smooth.py`**: Extracts face and body landmarks from video recordings using Google Mediapipe, it uses recurrent processing so landmark output is smoother, as current frame predictions use information from previous frames.

### 3. Landmark Overlay
- **`overlay_landmarks_video.py`**: Overlays extracted landmarks on the videos.

## Behavioral Measures
1.  **Landmarks**: Face and body landmarks extracted using Mediapipe.
---

## Project Setup

This guide will walk you through setting up the necessary environments to run the entire pipeline.

**System Requirements:**
*   **Python:** 3.10.x
*   **OS:** Windows, Linux, or macOS
*   **FFmpeg & FFprobe:** Must be installed and available on PATH (required for `sync_webcam_videos.py`)

### Step 1: Clone the Repository

```bash
git clone https://github.com/asortubay/AI_behavioral_analysis
cd AI_behavioral_analysis
```

### Step 1: Set Up the Mediapipe Environment

This is a separate, lightweight environment used only for running mediapipe extractor.

1.  **Create and Activate the Conda Environment:**
    ```bash
    # Create an environment named 'mediapipe_env'
    conda create --name mediapipe_env python=3.10

    # Activate the environment
    conda activate mediapipe_env
    ```

2.  **Install Required Packages:**
    ```bash
    pip install -r requirements.txt
    ```

---

## Usage

Each script is designed to be run independently. **Remember to activate the correct environment before running a script.**

### 1. Synchronize Two Webcam Videos

This step ensures your multi-camera setup is perfectly aligned in time, which is critical for behavioral analysis.

#### A. Recording Procedure
1.  Start recording on both cameras.
2.  **Wait a few seconds**, then perform a loud, sharp **CLAP** clearly visible to both cameras. This serves as the synchronization signal.
3.  Continue with your interview or session.
4.  Ensure the clap happens within the first 10 seconds of the video.

#### B. Running the Synchronization Script

1.  **Activate the environment:**
    ```bash
    conda activate mediapipe_env
    ```

2.  **Edit the script configuration:**
    Open `sync_webcam_videos.py` in your editor. Scroll to the bottom and update the file paths in the `if __name__ == "__main__":` block to point to your video files:

    ```python
    if __name__ == "__main__":
        video_1_path = r"C:\path\to\your\video_cam0.mkv"
        video_2_path = r"C:\path\to\your\video_cam1.mkv"
        
        sync_and_crop_videos(video_1_path, video_2_path)
    ```

3.  **Run the script:**
    ```bash
    python sync_webcam_videos.py
    ```

**What happens next?**
The script will extract audio, detect the clap, calculate the time offset, and re-export the videos.

**Outputs** (saved in the same folder as your input videos):
- `*_synced.avi`: The trimmed and aligned video files. They will have the **exact same number of frames**.
- `sync_check_side_by_side.avi`: A side-by-side video of the two synced outputs. Open this file to visually verify that the clap happens at the exact same moment in both views.

### 2. Run the Mediapipe Landmark Extractor

Activate the `mediapipe_env` for this step.

```bash
conda activate mediapipe_env

# Example command
python mediapipe_holitstic_extractor_batch.py path/to/video_files path/to/save_landmarks

python mediapipe_holistic_extractor_smooth.py path/to/video_files path/to/save_landmarks

python overlay_landmarks_video.py path/to/video_files path/to/save_landmarks --output_dir path/to/save_overlaid_videos


```

## Reference
If you use this code, please cite our paper:

Overlap and Differences of Autism and ADHD: Digital Phenotyping of Movement and Communication During Development

Aimar Silvan, Adriana Di Martino, Michael Milham, Lucas C Parra, Jens Madsen

doi: https://doi.org/10.1101/2025.10.20.682864 
