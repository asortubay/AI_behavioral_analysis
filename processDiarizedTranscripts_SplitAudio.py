import json
import copy
import os
import re
from tqdm import tqdm
import openwillis as ow
import argparse

def diarize_transcript(raw_json_path, diarized_txt_path, output_json_path):
    """
    Combines a raw JSON transcript with a diarized TXT transcript to produce
    a new JSON transcript with speaker labels.

    Args:
        raw_json_path: Path to the original JSON transcript.
        diarized_txt_path: Path to the diarized TXT transcript.
        output_json_path: Path to save the new diarized JSON transcript.
    """

    try:
        with open(raw_json_path, 'r', encoding='utf-8') as f:
            raw_json = json.load(f)
        with open(diarized_txt_path, 'r', encoding='utf-8') as f:
            diarized_lines = f.readlines()
    except FileNotFoundError as e:
        print(f"Error: File not found: {e}")
        return
    except json.JSONDecodeError as e:
        print(f"Error: JSON decoding error: {e}")
        return
    except Exception as e:
        print(f"An unexpected error occurred: {e}")
        return


    # Create a list of (speaker, text) tuples from the diarized text.
    diarized_data = []
    for line in diarized_lines:
        line = line.strip()
        if line:  # Skip empty lines
            try:
                speaker, text = line.split(":", 1)  # Split only on the first colon
                diarized_data.append((speaker.strip(), text.strip()))
            except ValueError:
                print(f"Warning: Skipping malformed line: '{line}' in {diarized_txt_path}")
                continue

    # Iterate through the raw JSON segments and add speaker labels.
    diarized_segments = []
    diarized_data_index = 0
    text_buffer = ""
    text_buffer_previous = ""  # Store the previous text buffer to check for misassigned speakers
    for segment in raw_json['segments']:
        segment_text = segment['text'].strip()
        
        # Add segment's full text to the buffer.
        text_buffer += segment_text + " "

        # Check if the accumulated text matches the current diarized segment
        if diarized_data_index < len(diarized_data):
            current_speaker, current_diarized_text = diarized_data[diarized_data_index]
            if current_speaker == "clip":  # Skip "clip" segments
                diarized_data_index += 1
                continue            
                
            # Check for the presence of complete diarized segment
            if current_diarized_text in text_buffer:
                # Create a new segment with the current speaker
                new_segment = segment.copy()  # Copy existing segment data.
                new_segment['speaker'] = current_speaker

                # Add the speaker to each word in the segment.
                for word_data in new_segment['words']:
                    word_data['speaker'] = current_speaker
                diarized_segments.append(new_segment)
                
                #check if the next diarized segment is in the buffer
                if diarized_data_index+1 < len(diarized_data):
                    next_speaker, next_diarized_text = diarized_data[diarized_data_index+1]
                    if next_diarized_text in text_buffer and next_speaker != 'clip':
                        diarized_segments[-1]['speaker'] = next_speaker
                        for word_data in diarized_segments[-1]['words']:
                            word_data['speaker'] = next_speaker
                        diarized_data_index += 1
                
                #update index and clear buffer
                diarized_data_index += 1
                text_buffer = ""

            # bla = 1


    # Handle any remaining diarized segments (shouldn't normally happen, but good to be safe)
    while diarized_data_index < len(diarized_data):
        print(f"Warning: Unmatched diarized segment: {diarized_data[diarized_data_index]} in {diarized_txt_path}")
        diarized_data_index += 1



    # Create the new diarized JSON structure.
    diarized_json = {
        'segments': diarized_segments,
        'word_segments': []  # We'll populate this next
    }

    # Add speaker information to 'word_segments'.  Iterate through segments
    # to preserve original order and ensure correct speaker.
    for segment in diarized_segments:
        for word_data in segment['words']:
            # Create a new word segment with only the fields that exist in word_data
            new_word_segment = {'speaker': word_data['speaker']}
            
            # Copy all existing fields from word_data
            for key in word_data:
                if key != 'speaker':  # We already added speaker
                    new_word_segment[key] = word_data[key]
                    
            diarized_json['word_segments'].append(new_word_segment)

    # Save the new JSON file.
    try:
        with open(output_json_path, 'w', encoding='utf-8') as outfile:
            json.dump(diarized_json, outfile, indent=4)
        print(f"Successfully created diarized JSON: {output_json_path}")
    except Exception as e:
        print(f"Error saving JSON file: {e}")



def process_transcripts(raw_folder, diarized_folder, output_folder):
    """
    Processes all transcripts in the given folders.

    Args:
        raw_folder: Folder containing the original JSON transcripts.
        diarized_folder: Folder containing the diarized TXT transcripts.
        output_folder: Folder to save the new diarized JSON transcripts.
    """

    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    # Get all raw JSON files
    for filename in tqdm(os.listdir(raw_folder)):
        if filename.endswith(".json"):
            base_filename = filename[:-5]  # Remove ".json"
            subject_id = base_filename.split("_MRI")[0]

            # Construct file paths
            raw_json_path = os.path.join(raw_folder, filename)
            diarized_txt_filename = base_filename + ".txt"
            diarized_txt_path = os.path.join(diarized_folder, diarized_txt_filename)
            output_json_path = os.path.join(output_folder, diarized_txt_filename[:-4] + ".json")


            # Check if the corresponding TXT file exists before processing
            if os.path.exists(output_json_path):
                print(f"Skipping {filename}: Output file already exists.")
                continue
            if os.path.exists(diarized_txt_path) and os.path.exists(raw_json_path):
                diarize_transcript(raw_json_path, diarized_txt_path, output_json_path)
            else:
                print(f"Warning: No matching diarized TXT file found for {filename}. Skipping.")

def replace_speaker_labels(obj): #replace speaker labels in the json file to speaker0 and speaker1, so we can later segment the audio
    if isinstance(obj, dict):
        return {k: replace_speaker_labels(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [replace_speaker_labels(item) for item in obj]
    elif isinstance(obj, str):
        return obj.replace("Interviewer", "speaker0").replace("Subject", "speaker1")
    else:
        return obj

def split_audio_files(diarized_output_folder, split_audio_folder, audio_source_folder):
    """
    Processes all diarized JSON transcripts and splits audio by speaker.
    
    Args:
        diarized_output_folder: Folder containing the diarized JSON transcripts.
        split_audio_folder: Folder to save the separated audio files.
        audio_source_folder: Folder containing the source audio files.
    """
    
    # Ensure the output directory exists
    if not os.path.exists(split_audio_folder):
        os.makedirs(split_audio_folder)
    
    # Process each JSON file
    for filename in tqdm(os.listdir(diarized_output_folder)):
        if filename.endswith(".json"):
            json_filepath = os.path.join(diarized_output_folder, filename)
            
            # Determine the corresponding audio file name
            # Assuming format: NDARXXXXXX_MRI_Present_Interview_partXofX_holistic_landmarks_overlay.json
            base_filename = filename[:-5]  # Remove ".json"
            audio_filename = f"{base_filename}.wav"
            audio_filepath = os.path.join(audio_source_folder, audio_filename)
            
            # Skip if the split audio file already exists
            split_audio_filename = f"{base_filename}_speaker1.wav"
            split_audio_filepath = os.path.join(split_audio_folder, split_audio_filename)
            if os.path.exists(split_audio_filepath):
                # print(f"Skipping {filename}: Split audio file already exists.")
                continue
            
            # Skip if the audio file doesn't exist
            if not os.path.exists(audio_filepath):
                # print(f"Warning: No matching audio file found for {filename}. Skipping.")
                continue
                
            try:
                # Load the JSON file
                with open(json_filepath, 'r', encoding='utf-8') as f:
                    transcript_json = json.load(f)              
                
                # Check if the segments list is empty
                if not transcript_json.get('segments') or len(transcript_json['segments']) == 0:
                    # print(f"Skipping {filename}: No segments found in the transcript.")
                    continue
                
                # Create a deep copy to avoid modifying the original unexpectedly
                modified_json = copy.deepcopy(transcript_json)
                
                # Apply the replacements throughout the JSON structure
                modified_json = replace_speaker_labels(modified_json)
                
                # print(f"Processing: {base_filename}")
                
                # Generate speaker separation and create audio files
                speaker_dict = ow.speaker_separation_labels(
                    filepath=audio_filepath, 
                    transcript_json=modified_json, 
                    volume_normalization=''
                )
                
                ow.to_audio(
                    filepath=audio_filepath, 
                    speaker_dict=speaker_dict, 
                    output_dir=split_audio_folder
                )
                
                # delete the audio file that corresponds to the interviewer (to save space, speaker0 is always the interviewer)
                interviwer_audio_filename = f"{base_filename}_speaker0.wav"
                interviwer_audio_filepath = os.path.join(split_audio_folder, interviwer_audio_filename)
                os.remove(interviwer_audio_filepath)
                
                
                # print(f"Successfully processed {filename}")
                
            except Exception as e:
                print(f"Error processing {filename}: {e}")
                continue
    
    print("Finished processing all files.")
    
    
    
# --- Main execution ---
if __name__ == "__main__":
        
    parser = argparse.ArgumentParser(description="Process diarized transcripts and split audio files.")
    parser.add_argument("raw_folder", type=str, help="Path to the directory containing raw JSON transcript files.")
    parser.add_argument("diarized_folder", type=str, help="Path to the directory containing diarized TXT transcript files.")
    parser.add_argument("diarized_output_folder", type=str, help="Path to the directory where diarized JSON transcripts will be saved.")
    parser.add_argument("original_audio_folder", type=str, help="Path to the directory containing original audio files.")
    parser.add_argument("split_audio_folder", type=str, help="Path to the directory where split audio files will be saved.")

    args = parser.parse_args()
    
    raw_folder = args.raw_folder
    diarized_folder = args.diarized_folder
    diarized_output_folder = args.diarized_output_folder
    original_audio_folder = args.original_audio_folder
    split_audio_folder = args.split_audio_folder
    process_transcripts(raw_folder, diarized_folder, diarized_output_folder)
    print("Finished processing transcripts.")
    
    split_audio_files(diarized_output_folder, split_audio_folder, original_audio_folder)
    print("All processing completed successfully.")
    
    
    
    