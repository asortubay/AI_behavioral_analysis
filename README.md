# AI Behavioral Analysis Pipeline

This repository contains scripts for processing interview audio/video recordings to extract behavioral measures for digital phenotyping research. The pipeline implements the methods described in our research paper for transcription, diarization, question-answer extraction, and embeddings. Note that some scripts may require modification based on specific dataset structures, interview contents (e.g., different questions), and desired output features, but the core functionality is provided.

Paper:
XXXX


## Pipeline Overview

### 1. Transcription & Diarization
- **`batch_transcription_whisperx.py`**: Transcribes audio recordings using WhisperX model (v.3.3.0, Whisper large-v2) to generate raw transcripts with word-level timestamps.
  - *Input*: Audio files (WAV, MP3, etc.)
  - *Output*: Timestamped transcripts with word alignments

- **`gemini_diarization_script.py`**: Performs speaker diarization using Google's Gemini-2.0-flash LLM with custom prompts to separate interviewer and participant speech.
  - *Input*: Raw transcripts from WhisperX
  - *Output*: Diarized transcripts with speaker labels

- **`processDiarizedTranscripts_SplitAudio.py`**: Splits original audio files into separate files for each speaker based on diarization timestamps.
  - *Input*: Original audio files and diarized transcripts
  - *Output*: Separated audio files for interviewer and participant

### 2. Question-Answer Extraction
- **`gemini_question_answer_extraction_ToM.py`**: Extracts specific protocol questions and corresponding participant answers from diarized transcripts using LLM prompts. Focuses on Theory of Mind and narrative questions, then embedds the extracted answers using Google's text-embedding-004 model for semantic analysis.
  - *Input*: Diarized transcripts
  - *Output*: Structured question-answer pairs for semantic analysis, including text embeddings

### 3. Feature Extraction
- **`computeSpeechFeatures_OW.py`**: Extracts acoustic and linguistic features using the Openwillis toolkit (v.3.0.5) for language and speech prosody measures.
  - *Input*: Audio files and transcripts
  - *Output*: Linguistic measures (lexical diversity, syntactic complexity, etc.)

- **`computeVocalAcoustics_OW.py`**: Computes vocal acoustic features for speech prosody analysis using Openwillis framework.
  - *Input*: Audio recordings
  - *Output*: Vocal acoustic measurements (fundamental frequency, jitter, shimmer, etc.)

## Behavioral Measures

The pipeline generates three main categories of behavioral variables:

1. **Language & Speech Prosody**: Acoustic and linguistic features via Openwillis
2. **Semantic Content**: Text embeddings using Google's Gecko model (text-embedding-004) to measure semantic typicality against normative baselines
3. **Movement**: Facial and body landmarks via Mediapipe Holistic with normalized 3D displacement measures

## Requirements

*Tested on Python 3.10.16*

```
whisperx==3.3.0
torch==1.10.0+cu102
torchaudio==0.10.0
transformers==4.46.3
openwillis==3.0.5
google-genai==1.0.0
tqdm==4.67.1
pathlib2==2.3.7
argparse==1.4.0
json5==0.12.0
```
See `requirements.txt` for the complete dependency list.


### 1. Python Environment Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/asortubay/AI_behavioral_analysis
    ```
2. Navigate to the project directory:
    ```bash
    cd AI_behavioral_analysis
      ```
3. Install the required dependencies:
    ```bash
    pip install -r requirements.txt
    ```

### 2. API Keys Setup
Set up API keys for Google Gemini and Hugging Face:
    - For Google Gemini, set the environment variable `GEMINI_API_KEY_0` with your API key.
    - For Hugging Face, set the environment variable `HF_API_KEY` with your token.
    ```cmd
    set GEMINI_API_KEY_0=your_actual_gemini_api_key_here
    set HF_API_KEY=your_huggingface_token_here
    ```

### 3. Model Setup
1. **WhisperX**: The script will automatically download the WhisperX model (large-v2) when first run, in the future, this model could be changed in the script if needed.
2. **Google Gemini**: Ensure you have access to the Gemini API and have set up your API key as described above, the available models in the gemini API may change, so refer to the latest documentation for model options. Other LLMs could be used in place of Gemini if desired, same for embeddings.


## Usage

Each script can be run independently, in this order:

```bash
>>python batch_transcription_whisperx.py  path/to/audio_files path/to/save_transcripts

>>python gemini_diarization_script.py path/to/save_transcripts path/to/save_diarized

>>python processDiarizedTranscripts_SplitAudio.py path/to/save_transcripts path/to/save_diarized path/to/save_diarized_json path/to/audio_files path/to/save_split_audio

>>python gemini_question_answer_extraction_ToM.py --input_diarized path/to/save_diarized_json --output_dir path/to/save_qa

>>python computeSpeechFeatures_OW.py --input_dir path/to/save_diarized_json --output_dir path/to/save_speech_features

>>python computeVocalAcoustics_OW.py path/to/save_split_audio path/to/save_acoustic_features
```


## Reference
If you use this code, please cite our paper:
XXXX
