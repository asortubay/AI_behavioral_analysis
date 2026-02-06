import os
import json
import torch
import whisperx
import gc
import argparse
from pathlib import Path
from tqdm import tqdm
from whisperx.diarize import DiarizationPipeline

### this file deletes the diarization applied by gemini and re-runs diarization based on pyannote so we can compare both approaches.


# Set device, batch size, and compute type
device = "cuda"
compute_type = "float16"  # Change to "int8" if low on GPU memory

def find_audio_file(base_name, audio_dirs):
    """
    Find the audio file corresponding to a transcript base name.
    Searches through provided audio directories.
    """
    # Common audio file patterns
    patterns = [
        f"{base_name}.wav",
        f"{base_name}_part1of1.wav",
        f"{base_name}_part1of1_holistic_landmarks_overlay.wav",
    ]
    
    for audio_dir in audio_dirs:
        for pattern in patterns:
            audio_path = os.path.join(audio_dir, pattern)
            if os.path.exists(audio_path):
                return audio_path
    
    return None

def process_json_file(json_file, output_dir, diarize_model, audio_dirs, model_a=None, metadata=None):
    """
    Load JSON transcript, apply diarization, and save diarized result.
    """
    base_name = os.path.splitext(os.path.basename(json_file))[0]
    output_json_path = os.path.join(output_dir, base_name + ".json")
    
    # Skip if output already exists
    if os.path.exists(output_json_path):
        return True
    
    # Find corresponding audio file
    audio_file = find_audio_file(base_name, audio_dirs)
    if audio_file is None:
        print(f"Warning: Could not find audio file for {base_name}")
        return False
    
    try:
        # Load the JSON result
        with open(json_file, 'r') as f:
            result = json.load(f)
        
        # Remove existing speaker assignments to overwrite with new diarization
        for segment in result.get("segments", []):
            if "speaker" in segment:
                del segment["speaker"]
            for word in segment["words"]:
                if "speaker" in word:
                    del word["speaker"]

        for word in result.get("word_segments",[]):
            if "speaker" in word:
                del word["speaker"]


        # Run diarization on the audio file
        diarize_segments = diarize_model(audio_file, min_speakers=2, max_speakers=2)
        
        # Assign speakers to words
        result = whisperx.assign_word_speakers(diarize_segments, result)
        
        # Save diarized transcript to JSON
        with open(output_json_path, 'w') as f:
            json.dump(result, f, indent=4)
        
        # Optionally save a text version
        output_txt_path = os.path.join(output_dir, base_name + ".txt")
        with open(output_txt_path, "w", encoding="utf-8") as f:
            for segment in result["segments"]:
                speaker = segment.get("speaker", "Unknown")
                f.write(f"{segment['start']} -> {segment['end']} [{speaker}]: {segment['text']}\n")
        
        return True
    
    except Exception as e:
        print(f"Error processing {base_name}: {str(e)}")
        return False

def batch_diarize(input_dir, output_dir, audio_dirs, hf_token):
    """
    Apply diarization to all JSON transcripts in input directory.
    """
    # Ensure output directory exists
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Load the diarization model
    print("Loading diarization model...")
    os.environ["PYANNOTE_SKIP_DEPENDENCY_CHECK"] = "1"
    diarize_model = DiarizationPipeline(use_auth_token=hf_token, device=device)
    
    # Optionally load alignment model if needed for re-alignment
    # model_a, metadata = whisperx.load_align_model(language_code="en", device=device)
    
    # Get all .json files from input directory
    json_files = [f for f in os.listdir(input_dir) if f.endswith(".json")]
    
    if not json_files:
        print(f"No JSON files found in {input_dir}")
        return
    
    print(f"Found {len(json_files)} JSON files to diarize.")
    
    # Process each file with progress bar
    successful = 0
    failed = 0
    
    for json_file_name in tqdm(json_files, desc="Diarizing transcripts", unit="file"):
        json_file_path = os.path.join(input_dir, json_file_name)
        if process_json_file(json_file_path, output_dir, diarize_model, audio_dirs, None, None):
            successful += 1
        else:
            failed += 1
    
    print(f"\nDiarization complete.")
    print(f"Successfully processed: {successful}")
    print(f"Failed: {failed}")
    
    # Free memory
    del diarize_model
    gc.collect()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Apply post-hoc diarization to transcripts.")
    parser.add_argument("--input-dir", type=str, default=r"Z:\cmi_transcriptions\diarized",
                        help="Path to directory containing JSON transcripts (non-diarized).")
    parser.add_argument("--output-dir", type=str, default=r"Z:\cmi_transcriptions\pyannote_diarized",
                        help="Path to directory where diarized transcripts will be saved.")
    parser.add_argument("--audio-dirs", type=str, nargs='+', 
                        default=[r"Z:\cmi_audio_from_present_interview"],
                        help="Path(s) to directory/directories containing audio files.")
    
    args = parser.parse_args()
    
    # Run the batch diarization
    batch_diarize(args.input_dir, args.output_dir, args.audio_dirs, os.getenv("HF_API_KEY"))
