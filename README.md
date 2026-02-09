
# AI Behavioral Analysis Pipeline

This repository contains scripts for processing interview audio/video recordings to extract behavioral measures for digital phenotyping research. The pipeline implements the methods described in our research paper for transcription, diarization, question-answer extraction, and feature embedding.

Note that some scripts may require modification based on specific dataset structures, interview contents (e.g., different questions), and desired output features, but the core functionality is provided.

**Paper:**
Overlap and Differences of Autism and ADHD: Digital Phenotyping of Movement and Communication During Development

Aimar Silvan, Adriana Di Martino, Michael Milham, Lucas C Parra, Jens Madsen

doi: https://doi.org/10.1101/2025.10.20.682864 


## Pipeline Overview

### 1. Video Synchronization
- **`sync_webcam_videos.py`**: Synchronizes two webcam videos using a clap (sharp audio transient) in the first few seconds. Detects the clap onset independently in each audio track, refines alignment with cross-correlation, and exports frame-accurately trimmed videos with identical frame counts. Outputs are saved alongside the originals as `*_synced.avi`, plus a side-by-side verification video.

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
    pip install mediapipe==0.10.11 tqdm==4.67.1 pathlib2==2.3.7 numpy scipy
    ```

---

## Usage

Each script is designed to be run independently. **Remember to activate the correct environment before running a script.**

### 1. Synchronize Two Webcam Videos

Activate the `mediapipe_env` for this step. Both input videos must contain a clap (or sharp transient) within the first ~5 seconds.

```bash
conda activate mediapipe_env

# Edit the video paths at the bottom of the script, then run:
python sync_webcam_videos.py
```

**Outputs** (saved alongside the original files):
- `cam0_..._synced.avi` — trimmed & aligned video 1
- `cam1_..._synced.avi` — trimmed & aligned video 2
- `sync_check_side_by_side.avi` — low-res side-by-side for visual verification

Both synced videos are guaranteed to have the **exact same number of frames**.

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
