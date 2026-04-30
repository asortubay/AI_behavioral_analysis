import os
import json
import whisperx
import gc
import argparse
import subprocess
from tqdm import tqdm

# Set device, batch size, and compute type
device = "cuda"
batch_size = 16  # Reduce if low on memory
compute_type = "float16"  # Change to "int8" if low on GPU memory (may reduce accuracy)

# Formats to look for in the input directory
SUPPORTED_EXTENSIONS = {".mp4", ".mkv", ".avi", ".mov", ".flv", ".wmv", ".wav", ".mp3", ".m4a", ".flac"}

def extract_audio(input_file, audio_output_path):
    """
    Extracts audio from any media file using FFmpeg. 
    Converts to 16kHz mono WAV to preserve exact sync and optimize for WhisperX.
    """
    command = [
        "ffmpeg", "-y", "-i", input_file,
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        audio_output_path
    ]
    # Run ffmpeg quietly
    subprocess.run(command, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

def process_media_file(input_file, audio_dir, output_dir, model, model_a, metadata):
    # Prepare output paths
    base_name = os.path.splitext(os.path.basename(input_file))[0]
    audio_file = os.path.join(audio_dir, base_name + ".wav")
    txt_file = os.path.join(output_dir, base_name + ".txt")
    json_file = os.path.join(output_dir, base_name + ".json")

    # Skip processing if both transcript files already exist
    if os.path.exists(txt_file) and os.path.exists(json_file):
        return

    # 1. Extract audio if it doesn't already exist in the audio folder
    if not os.path.exists(audio_file):
        extract_audio(input_file, audio_file)

    # 2. Load and transcribe the audio
    audio = whisperx.load_audio(audio_file)
    result = model.transcribe(audio, batch_size=batch_size, language="en")

    # 3. Align the transcription (This is what gives EXACT word-level timing)
    result = whisperx.align(result["segments"], model_a, metadata, audio, device, return_char_alignments=False)

    # 4. Save transcript to .txt (Now includes exact word timings)
    with open(txt_file, "w", encoding="utf-8") as f:
        for segment in result["segments"]:
            f.write(f"[{segment['start']:.3f} -> {segment['end']:.3f}] {segment['text']}\n")
            
            # Print exact word-level timings underneath each segment
            if "words" in segment:
                for word in segment["words"]:
                    if "start" in word and "end" in word:
                        f.write(f"    {word['start']:.3f} -> {word['end']:.3f}: {word['word']}\n")

    # 5. Save transcript to .json
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4)

def process_directory(input_dir, audio_dir, output_dir):
    # Ensure the output and intermediate directories exist
    os.makedirs(audio_dir, exist_ok=True)
    os.makedirs(output_dir, exist_ok=True)

    # Load the model only once
    model = whisperx.load_model("large-v2", device, compute_type=compute_type)
    
    # Load the align model only once
    model_a, metadata = whisperx.load_align_model(language_code="en", device=device)

    # Get all supported media files in directory (ignores hidden or unsupported files)
    media_files = [f for f in os.listdir(input_dir) if os.path.splitext(f)[1].lower() in SUPPORTED_EXTENSIONS]

    # Process each file with a progress bar
    for file_name in tqdm(media_files, desc="Processing files", unit="file"):
        input_file = os.path.join(input_dir, file_name)
        process_media_file(input_file, audio_dir, output_dir, model, model_a, metadata)

    # Free memory
    del model
    del model_a
    gc.collect()

if __name__ == "__main__":
    # Set up argument parsing with your exact paths as defaults
    parser = argparse.ArgumentParser(description="Extract audio and transcribe media files with exact word timings.")
    
    # Use raw strings (r"") to handle Windows backslashes properly
    parser.add_argument("--input_dir", type=str, 
                        default=r"C:\Users\Aimar\Downloads\hbn_eeg_eyetracking_stimuli\videos", 
                        help="Path to the input directory containing media files.")
    
    parser.add_argument("--audio_dir", type=str, 
                        default=r"C:\Users\Aimar\Downloads\hbn_eeg_eyetracking_stimuli\audio", 
                        help="Path to save the intermediate audio extractions.")
    
    parser.add_argument("--output_dir", type=str, 
                        default=r"C:\Users\Aimar\Downloads\hbn_eeg_eyetracking_stimuli\transcript", 
                        help="Path to save the .txt and .json transcripts.")

    # Parse command-line arguments
    args = parser.parse_args()

    # Run the processing
    process_directory(args.input_dir, args.audio_dir, args.output_dir)