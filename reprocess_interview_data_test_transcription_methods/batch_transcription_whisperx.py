import os
import json
import torch
import whisperx
import gc
import argparse
import re
import numpy as np
import librosa
import soundfile as sf
from collections import defaultdict
from tqdm import tqdm

from whisperx.diarize import DiarizationPipeline
# from pyannote2.audio import Pipeline



# Set device, batch size, and compute type
device = "cuda"
batch_size = 16  # Reduce if low on memory
compute_type = "float16"  # Change to "int8" if low on GPU memory (may reduce accuracy)

def parse_split_filename(filename):
    """
    Parse filename to extract base name and part information.
    Returns (base_name, part_num, total_parts) if file is split, else (base_name, None, None)
    Example: "_MRI_Speech_Language_part2of3.wav" -> ("_MRI_Speech_Language", 2, 3)
    """
    match = re.search(r'_part(\d+)of(\d+)\.wav$', filename)
    if match:
        part_num = int(match.group(1))
        total_parts = int(match.group(2))
        base_name = filename[:match.start()]
        return base_name, part_num, total_parts
    return None, None, None

def group_split_files(input_dir):
    """
    Group split audio files by their base name.
    Only includes files that are NOT already _part1of1.wav (i.e., total_parts > 1).
    Returns a dict of {base_name: [(filename, part_num, total_parts), ...]}
    """
    wav_files = [f for f in os.listdir(input_dir) if f.endswith(".wav")]
    split_groups = defaultdict(list)
    
    for filename in wav_files:
        base_name, part_num, total_parts = parse_split_filename(filename)
        # Only include files that have multiple parts (total_parts > 1)
        if base_name is not None and total_parts > 1:
            split_groups[base_name].append((filename, part_num, total_parts))
    
    return split_groups

def concatenate_split_audio_files(input_dir, base_name, file_list):
    """
    Concatenate split audio files and save as _part1of1.wav
    file_list: list of (filename, part_num, total_parts) tuples
    """
    # Sort by part number
    file_list.sort(key=lambda x: x[1])


    # output filename
    output_filename = f"{base_name}_part1of1.wav"
    output_path = os.path.join(input_dir, output_filename)
    if os.path.exists(output_path):
        return
    
    # Load and concatenate all audio files
    audio_data = []
    sr = None
    
    for filename, part_num, total_parts in file_list:
        file_path = os.path.join(input_dir, filename)   
        audio, sample_rate = librosa.load(file_path, sr=None)
        audio_data.append(audio)
        if sr is None:
            sr = sample_rate
        elif sr != sample_rate:
            # Resample if sample rates don't match
            audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=sr)
            audio_data.append(audio)
    
    # Concatenate all audio data
    concatenated_audio = np.concatenate(audio_data)
    
    # Save as _part1of1.wav
    output_filename = f"{base_name}_part1of1.wav"
    output_path = os.path.join(input_dir, output_filename)
    sf.write(output_path, concatenated_audio, sr)
    
    print(f"Concatenated {len(file_list)} parts into: {output_filename}")

def preprocess_split_audio_files(input_dir):
    """
    Find and concatenate all split audio files in the directory.
    """
    split_groups = group_split_files(input_dir)
    
    if split_groups:
        print(f"Found {len(split_groups)} groups of split audio files. Concatenating...")
        for base_name, file_list in split_groups.items():
            concatenate_split_audio_files(input_dir, base_name, file_list)
        print("Concatenation complete.\n")

def process_wav_file(audio_file, output_dir, model, model_a, diarize_model, metadata):
    # Prepare output paths
    base_name = os.path.splitext(os.path.basename(audio_file))[0]  # Get the file name without extension
    txt_file = os.path.join(output_dir, base_name + ".txt")
    json_file = os.path.join(output_dir, base_name + ".json")

    # Skip processing if both output files already exist
    if os.path.exists(txt_file) and os.path.exists(json_file):
        return

    # Load and transcribe the audio
    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=batch_size, language="en")

    # Align the transcription
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    # diarize model
    if diarize_model is not None:
        diarize_segments = diarize_model(audio_file, min_speakers=2, max_speakers=2)
        result = whisperx.assign_word_speakers(diarize_segments, result)

    # Save transcript to .txt
    with open(txt_file, "w", encoding="utf-8") as f:
        for segment in result["segments"]:
            f.write(f"{segment['start']} -> {segment['end']}: {segment['text']}\n")

    # Save transcript to .json
    with open(json_file, "w") as f:
        json.dump(result, f, indent=4)

def process_directory(input_dir, output_dir, hf_token):
    # Ensure the output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Preprocess: concatenate split audio files
    preprocess_split_audio_files(input_dir)

    # Load the model only once
    model = whisperx.load_model("large-v2", device, compute_type=compute_type,language="en")
    

    # Load the align model only once
    model_a, metadata = whisperx.load_align_model(language_code="en", device=device)

    # Load the diarization model only once
    diarize_model = None
    ## CODE TO USE THE MOST UP-TO-DATE COMMUNITY MODEL FROM HUGGINGFACE, BUT THROWS ERROR WITH WhisperX
    os.environ["PYANNOTE_SKIP_DEPENDENCY_CHECK"] = "1"
    # diarize_model = Pipeline.from_pretrained('pyannote/speaker-diarization-community-1', token=hf_token)
    # diarize_model = Pipeline.from_pretrained("/speaker_diarization")
    # diarize_model.to(torch.device(device))
    diarize_model = DiarizationPipeline(use_auth_token=hf_token, device=device)

    # Get all .wav files that end with _part1of1.wav (skip split files)
    wav_files = [f for f in os.listdir(input_dir) if f.endswith("_part1of1_holistic_landmarks_overlay.wav")]

    # Process each file with a progress bar
    for file_name in tqdm(wav_files, desc="Processing files", unit="file"):
        audio_file = os.path.join(input_dir, file_name)
        process_wav_file(audio_file, output_dir, model, model_a, diarize_model, metadata)

    # Free memory
    del model
    del model_a
    gc.collect()

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Transcribe .wav files in a directory and save results.")
    # parser.add_argument("input_dir", type=str, help="Path to the input directory containing .wav files.")
    # parser.add_argument("output_dir", type=str, help="Path to the directory where transcripts will be saved.")

    # Parse command-line arguments
    args = parser.parse_args()

    args.input_dir = r"Z:\cmi_audio_from_present_interview" # r"C:\Users\Aimar\Desktop\hbn_language_task\data\test_videos_wav"
    args.output_dir = r"C:\Users\Aimar\Desktop\hbn_language_task\data\interview_task\transcripts\00_raw_transcripts"
# r"C:\Users\Aimar\Desktop\hbn_language_task\data\test_videos_transcripts_prompted"

    # Run the processing
    process_directory(args.input_dir, args.output_dir, os.getenv("HF_API_KEY"))
