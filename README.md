
# AI Behavioral Analysis Pipeline

This repository contains scripts for processing interview audio/video recordings to extract behavioral measures for digital phenotyping research. The pipeline implements the methods described in our research paper for transcription, diarization, question-answer extraction, and feature embedding.

Note that some scripts may require modification based on specific dataset structures, interview contents (e.g., different questions), and desired output features, but the core functionality is provided.

**Paper:**
Overlap and Differences of Autism and ADHD: Digital Phenotyping of Movement and Communication During Development

Aimar Silvan, Adriana Di Martino, Michael Milham, Lucas C Parra, Jens Madsen

doi: https://doi.org/10.1101/2025.10.20.682864 


## Pipeline Overview

### 1. Transcription & Diarization
- **`batch_transcription_whisperx.py`**: Transcribes audio using WhisperX (v.3.3.0, Whisper large-v2) to generate raw transcripts with word-level timestamps.
- **`gemini_diarization_script.py`**: Performs speaker diarization using Google's Gemini-flash model to separate interviewer and participant speech.
- **`processDiarizedTranscripts_SplitAudio.py`**: Splits original audio files into separate files for each speaker based on diarization timestamps.

### 2. Question-Answer Extraction
- **`gemini_question_answer_extraction_ToM.py`**: Extracts specific questions and corresponding answers from diarized transcripts using LLM prompts. It then embeds the extracted answers using Google's `text-embedding-004` model for semantic analysis.

### 3. Feature Extraction
- **`computeSpeechFeatures_OW.py`**: Extracts acoustic and linguistic features using the Openwillis toolkit (v.3.0.5).
- **`computeVocalAcoustics_OW.py`**: Computes vocal acoustic features for speech prosody analysis using the Openwillis framework.
- **`mediapipe_holitstic_extractor.py`**: Extracts face and body landmarks from video recordings using Google Mediapipe.

## Behavioral Measures
The pipeline generates three main categories of behavioral variables:
1.  **Language & Speech Prosody**: Acoustic and linguistic features from Openwillis.
2.  **Semantic Content**: Text embeddings via Google's `text-embedding-004` model to measure semantic typicality.
3.  **Landmarks**: Face and body landmarks extracted using Mediapipe.

---

## Project Setup

This guide will walk you through setting up the necessary environments to run the entire pipeline.

**System Requirements:**
*   **Python:** 3.10.x
*   **OS:** Windows, Linux, or macOS
*   **GPU:** An NVIDIA GPU is required for `whisperx` and PyTorch acceleration.

### Important: Why Two Environments?
This project requires two separate Python environments due to a dependency conflict:
- `openwillis` requires `mediapipe==0.10.18`.
- The `mediapipe_holitstic_extractor.py` script requires `mediapipe==0.10.11`.

Following these steps will ensure both parts of the pipeline work correctly.

### Step 1: Clone the Repository

```bash
git clone https://github.com/asortubay/AI_behavioral_analysis
cd AI_behavioral_analysis
```

### Step 2: Set Up the Main Pipeline Environment

This environment will be used for all scripts **except** the Mediapipe landmark extractor. We recommend using **Conda** as it simplifies the installation of complex dependencies like PyTorch and FFmpeg.

1.  **Create and Activate the Conda Environment:**
    ```bash
    # Create an environment named 'main_env' with Python 3.10
    conda create --name main_env python=3.10

    # Activate the environment
    conda activate main_env
    ```

2.  **Install PyTorch with CUDA Support:**
    To ensure optimal performance, install the PyTorch version that matches your NVIDIA GPU's CUDA capability.
    - **Go to the [PyTorch "Get Started" page](https://pytorch.org/get-started/locally/).**
    - **Configure your build:** Select "Stable", your "OS", "Conda", and "Python".
    - **Choose the right CUDA version:** For most modern GPUs (RTX 20-series or newer), **CUDA 12.1** is the correct and stable choice.
    - Copy the generated command and run it. It will look like this:
      ```bash
      # This command is for CUDA 12.1. Verify on the PyTorch website for your specific hardware.
      conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia
      ```

3.  **Install Remaining Dependencies:**
    Use the provided `requirements.txt` file to install the rest of the packages.
    ```bash
    pip install -r requirements.txt
    ```

4.  **Install FFmpeg:**
    This is required by `whisperx` for audio processing.
    ```bash
    conda install ffmpeg -c conda-forge
    ```

### Step 3: Set Up the Mediapipe Environment

This is a separate, lightweight environment used only for running `mediapipe_holitstic_extractor.py`.

1.  **Create and Activate the Conda Environment:**
    ```bash
    # Create an environment named 'mediapipe_env'
    conda create --name mediapipe_env python=3.10

    # Activate the environment
    conda activate mediapipe_env
    ```

2.  **Install Required Packages:**
    ```bash
    pip install mediapipe==0.10.11 tqdm==4.67.1 pathlib2==2.3.7
    ```

### Step 4: Set Up API Keys

You need API keys for Google Gemini and Hugging Face. Set these as environment variables.

**On Windows (in Command Prompt or Anaconda Prompt):**
```cmd
set GEMINI_API_KEY_0="your_actual_gemini_api_key_here"
set HF_API_KEY="your_huggingface_token_here"
```

**On Linux/macOS:**
```bash
export GEMINI_API_KEY_0="your_actual_gemini_api_key_here"
export HF_API_KEY="your_huggingface_token_here"
```
*Note: You may need to set these keys each time you open a new terminal, or add them to your system's environment variables permanently.*

---

## Usage

Each script is designed to be run independently. **Remember to activate the correct environment before running a script.**

### 1. Run the Main Pipeline Scripts

Activate the `main_env` for these steps.

```bash
conda activate main_env

# Example commands (replace paths with your actual file locations)
python batch_transcription_whisperx.py path/to/audio_files path/to/save_transcripts
python gemini_diarization_script.py path/to/save_transcripts path/to/save_diarized
python processDiarizedTranscripts_SplitAudio.py path/to/save_transcripts path/to/save_diarized path/to/save_diarized_json path/to/audio_files path/to/save_split_audio
python gemini_question_answer_extraction_ToM.py path/to/save_diarized_json path/to/save_qa
python computeSpeechFeatures_OW.py path/to/save_diarized_json path/to/save_speech_features
python computeVocalAcoustics_OW.py path/to/save_split_audio path/to/save_acoustic_features
```

### 2. Run the Mediapipe Landmark Extractor

Activate the `mediapipe_env` for this step.

```bash
conda activate mediapipe_env

# Example command
python mediapipe_holitstic_extractor.py path/to/video_files path/to/save_landmarks
```

## Reference
If you use this code, please cite our paper:

Overlap and Differences of Autism and ADHD: Digital Phenotyping of Movement and Communication During Development

Aimar Silvan, Adriana Di Martino, Michael Milham, Lucas C Parra, Jens Madsen

doi: https://doi.org/10.1101/2025.10.20.682864 
