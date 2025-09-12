import os
import json
import whisperx
import gc
import argparse
from tqdm import tqdm

# Set device, batch size, and compute type
device = "cuda"
batch_size = 16  # Reduce if low on memory
compute_type = "int8"  # Change to "int8" if low on GPU memory (may reduce accuracy)

def process_wav_file(audio_file, output_dir, model, model_a, metadata):
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

    # Load the model only once
    model = whisperx.load_model("large-v2", device, compute_type=compute_type)
    
    # Load the align model only once
    model_a, metadata = whisperx.load_align_model(language_code="en", device=device)


    # Get all .wav files in directory
    wav_files = [f for f in os.listdir(input_dir) if f.endswith(".wav")]

    # Process each file with a progress bar
    for file_name in tqdm(wav_files, desc="Processing files", unit="file"):
        audio_file = os.path.join(input_dir, file_name)
        process_wav_file(audio_file, output_dir, model, model_a, metadata)

    # Free memory
    del model
    del model_a
    gc.collect()

if __name__ == "__main__":
    # Set up argument parsing
    parser = argparse.ArgumentParser(description="Transcribe .wav files in a directory and save results.")
    parser.add_argument("input_dir", type=str, help="Path to the input directory containing .wav files.")
    parser.add_argument("output_dir", type=str, help="Path to the directory where transcripts will be saved.")

    # Parse command-line arguments
    args = parser.parse_args()

    # Run the processing
    process_directory(args.input_dir, args.output_dir, os.getenv("HF_API_KEY"))
