import os
import argparse
import time
from google import genai
from google.genai import types

from tqdm import tqdm
import re
import json
import csv

# Gemini model parameters
MODEL_NAME = "gemini-3-flash-preview"
MAX_RETRIES = 3  # Number of retries if the API fails
DELAY = 2  # Seconds to wait between retries

# System prompt for Gemini
SYSTEM_PROMPT = """
Role: You are an expert clinical transcriptionist. Your task is to diarize a transcript of a voice and speech assessment by labeling speakers as "Interviewer:" (the clinician) or "Subject:" (the participant).

### 1. Core Directives (Strict Compliance Required):
*   STRICT FIDELITY: Do not add, remove, change, or "fix" any words in the transcript. If the text is a fragment or contains errors, leave it exactly as it is.
*   NO HALLUCINATION: Do not make up dialogue to fill gaps. Use only the words provided in the source transcript.
*   REMOVE TIMESTAMPS: Strip all time codes (e.g., `12.3 -> 14.5:`) but keep the text structure intact.
*   CONTEXTUAL DIARIZATION: Use the sequence of the clinical protocol to determine the speaker.

### 2. Clinical Protocol Logic:
*   Reading Passages: During the stories listed below, the Subject is the primary speaker. However, the Interviewer may interject to provide a word if the subject gets stuck, or to provide encouragement (e.g., "You're doing great"), the interviewer may also read before the participant to provide guidance.

### 3. Reading Passage Reference (Story Text):
Use these texts to identify when the Subject is reading. If the transcript text matches these stories, label it as Subject, unless the clinician is clearly interrupting to help or encourage.
Story 1: Peggy Babcock
> "It was the first day of school. It was a tough day for all the kids. One girl had a really hard time because nobody could say her name. Her name was Peggy Babcock. Go ahead, try and say it three times quickly. Peggy Babcock, Peggy Babcock, Peggy Babcock. Not easy going, right? She was afraid to say hello to any of the other kids on the playland. One boy walked up to her and asked what her name was. She said, when you hear my name, it sounds simple, but no one can say it. It is Peggy Babcock. He laughed and said, your name is tricky, but mine is better. It sounds simple, but no one can remember it. It is Jonas Norvin Sven Arthur Schwinn Bart Winston Ulysses M. Peggy laughed and said, easy, your name sounds like Joan is nervous when others win. But you win some, you lose some. How do you like my version? Jonas was so happy that he said, let's be friends. I will call you PB. A pair of them stuck so close to each other that everyone at school called them PB&J."

Story 2: The Phonetic Kingdom
> "Some time ago, in a place neither near nor far, there lived a king who didn't know how to count, not even to zero. Some say this is the reason he would always wish for more. More food, more gold, more land. He simply didn't realize how much he already owned. Everyone in his kingdom could do the math and tally bushels of corn, loaves of bread, and urns of gold. But how would they measure the height of his castle or the stretch of his kingdom? You might think, ah, ooh, easy, just measure it in meters. But in those days, the useless unit of measure was based on stained splatter along the king's cloak while drinking shrub juice. The kingdom needed a new way of counting distance. A kingdom without a proper ruler, proclaimed the king, is like riches without measure. He launched a challenge amid trumpets, drums, flags, and cannons. The person who creates a unit of measure fit for a ruler will be rewarded beyond measure. A tall order indeed. The first person to come forward was a bulky locksmith with a stiff jaw. He approached the king with an air of secrecy and whispered, I have the key to measure the kingdom, but only I can wield it. He then rubbed his beard and pulled the key from his locks of oily hair. The key turned out to be a hair itself. Judge the reach of my vast kingdom with a hair's width, laughed the king. What a poor idea. That would take forever or longer. The second person eager for the prize was a fidgety boy who knew all numbers, including zero. He produced a curious object from one of his many pockets. It was a complex shape that seemed to change proportions depending on which direction you gazed upon. The boy said in a measured voice, this polyhedron has many edges, with each edge of a different length. Only a kid could be counted on to use it justly. He gave the king an awful earful of an explanation that went on and on. The long and the short of it was that the king could make no more use of it than a puddle of spilled oatmeal. Finally, a little girl with a big idea tugged on the mismeasured cloak of the king. The king sized up the little girl with the big idea and said, I don't have time for this, and for that matter, I have no concept of space either. The girl looked up, then down, then spun around and blurted out, Aren't you able to solve this puzzle yourself? Why must you break up your kingdom into tiny pieces when everything around you is humpty-dumpty together again? Your kingdom is a unit, and you are the ruler. The king, startled, befuddled, and bemused, found the words wise. He aimed to be satisfied with all around him, big or small, or somewhere in between."

### 4. Formatting:
*   Label every turn as Interviewer: or Subject:
*   Maintain the original line breaks and spacing of the transcript, only removing timestamps.
*   Make sure to keep the original text intact, only removing timestamps

### These specific lines are part of the story, so it is likely the subject is saying them:
1. "Go ahead, try and say it three times quickly."
2. "That's easy going, right?"


---

### Transcript to Process:
"""
verbose = True

def load_reading_analysis_timestamps(analysis_json_file):
    """
    Load timestamps from gemini_reading_text_analysis output.
    Returns a dict with start_s and end_s for each story (if available).
    Returns None if file doesn't exist.
    """
    if not os.path.exists(analysis_json_file):
        return None
    
    try:
        with open(analysis_json_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        if verbose:
            print(f"Error loading analysis file {analysis_json_file}: {e}")
        return None

def filter_transcript_by_timestamp(input_txt_file, input_json_file, start_s, end_s, output_dir, base_name):
    """
    Filter transcript files to only include segments within the timestamp range [start_s, end_s].
    Saves filtered txt and json files.
    """
    # Load the JSON transcript
    with open(input_json_file, 'r', encoding='utf-8') as f:
        full_transcript = json.load(f)
    
    # Filter segments
    filtered_segments = [
        seg for seg in full_transcript.get('segments', [])
        if seg['start'] >= start_s and seg['end'] <= end_s
    ]
    
    # Create filtered transcript data
    filtered_transcript = full_transcript.copy()
    filtered_transcript['segments'] = filtered_segments
    
    # Save filtered JSON only if it doesn't exist
    output_json_file = os.path.join(output_dir, base_name + ".json")
    if not os.path.exists(output_json_file):
        with open(output_json_file, 'w', encoding='utf-8') as f:
            json.dump(filtered_transcript, f, indent=4)
    
    # Save filtered TXT only if it doesn't exist
    output_txt_file = os.path.join(output_dir, base_name + ".txt")
    if not os.path.exists(output_txt_file):
        with open(output_txt_file, 'w', encoding='utf-8') as f:
            for segment in filtered_segments:
                f.write(f"{segment['start']} -> {segment['end']}: {segment['text']}\n")
    
        print(f"Saved filtered transcript: {base_name} ({len(filtered_segments)} segments)")

def preprocess_transcripts_with_reading_analysis(input_dir, output_dir, analysis_dir):
    """
    Preprocess transcripts by filtering them based on reading analysis timestamps.
    Finds the earliest start_s from all stories and cuts the transcript from that point to the end.
    Looks for matching .json files in analysis_dir for each transcript.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Get all .txt files in input directory
    txt_files = [f for f in os.listdir(input_dir) if f.endswith(".txt")]
    
    processed_count = 0
    skipped_count = 0
    
    for txt_file in txt_files:
        base_name = os.path.splitext(txt_file)[0]
        input_txt_path = os.path.join(input_dir, txt_file)
        input_json_path = os.path.join(input_dir, base_name + ".json")
        
        # Check if corresponding JSON exists
        if not os.path.exists(input_json_path):
            if verbose:
                print(f"Skipping {txt_file}: corresponding JSON file not found.")
            skipped_count += 1
            continue
        
        # Look for analysis file in analysis_dir
        analysis_json_file = os.path.join(analysis_dir, base_name + ".txt")  # Note: analysis output is .txt format but contains JSON
        
        timestamps_data = load_reading_analysis_timestamps(analysis_json_file)
        
        if timestamps_data is None:
            if verbose:
                print(f"Skipping {txt_file}: no reading analysis found.")
            skipped_count += 1
            continue
        
        # Find the earliest start timestamp across all stories
        earliest_start = None
        for story_key in ['peggy_babcock', 'phonetic_kingdom']:
            if story_key in timestamps_data:
                story_data = timestamps_data[story_key]
                start_s = story_data.get('start_s')
                if start_s is not None:
                    if earliest_start is None or start_s < earliest_start:
                        earliest_start = start_s
        
        if earliest_start is None:
            if verbose:
                print(f"Skipping {txt_file}: no valid start timestamps found in analysis.")
            skipped_count += 1
            continue
        
        # Cut transcript from earliest_start to the end
        filter_transcript_by_timestamp(
            input_txt_path, input_json_path, earliest_start, float('inf'), output_dir, base_name
        )
        processed_count += 1
    
    if verbose:
        print(f"\nPreprocessing complete: {processed_count} cut transcripts created, {skipped_count} skipped.\n")
    
    return processed_count, skipped_count

def generate_reading_analysis_csv(analysis_dir, output_dir, output_filename="reading_analysis_summary.csv"):
    """
    Parse reading analysis JSON files and generate a CSV with participant data.
    CSV will contain one row per participant with separate columns for each story.
    Columns: participant_id, peggy_babcock_start_s, peggy_babcock_end_s, peggy_babcock_overall_read_score,
             phonetic_kingdom_start_s, phonetic_kingdom_end_s, phonetic_kingdom_overall_read_score
    Missing data will be left empty.
    """
    if not os.path.exists(analysis_dir):
        if verbose:
            print(f"Analysis directory not found: {analysis_dir}")
        return
    
    rows = {}
    analysis_files = [f for f in os.listdir(analysis_dir) if f.endswith(".txt")]
    
    for analysis_file in analysis_files:
        file_path = os.path.join(analysis_dir, analysis_file)
        participant_id = os.path.splitext(analysis_file)[0]
        
        # Initialize row for this participant
        if participant_id not in rows:
            rows[participant_id] = {'participant_id': participant_id.split('_')[0]}
        
        timestamps_data = load_reading_analysis_timestamps(file_path)
        if timestamps_data is None:
            continue
        
        # Process each story
        for story_key in ['peggy_babcock', 'phonetic_kingdom']:
            if story_key in timestamps_data:
                story_data = timestamps_data[story_key]
                start_s = story_data.get('start_s', '')
                end_s = story_data.get('end_s', '')
                overall_read_score = story_data.get('overall_read_score', '')
                
                rows[participant_id][f'{story_key}_start_s'] = start_s
                rows[participant_id][f'{story_key}_end_s'] = end_s
                rows[participant_id][f'{story_key}_overall_read_score'] = overall_read_score
    
    # Write CSV
    if rows:
        csv_path = os.path.join(output_dir, output_filename)
        fieldnames = [
            'participant_id',
            'peggy_babcock_start_s', 'peggy_babcock_end_s', 'peggy_babcock_overall_read_score',
            'phonetic_kingdom_start_s', 'phonetic_kingdom_end_s', 'phonetic_kingdom_overall_read_score'
        ]
        
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames, restval='')
            writer.writeheader()
            writer.writerows(rows.values())
        
        if verbose:
            print(f"Reading analysis CSV saved: {csv_path}")
    else:
        if verbose:
            print("No reading analysis data found to create CSV.")


def call_gemini_api(transcript, api_key):
    # Calls the Gemini API to perform diarization.
    client = genai.Client(api_key=api_key)
    generation_config = types.GenerateContentConfig(
        temperature=0.0,
    )
    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME, 
                contents=SYSTEM_PROMPT + "\n\n" + transcript,
                config=generation_config
            )
            return response.text
        except Exception as e:
            if verbose:
                print(f"API error: {e}, retrying ({attempt + 1}/{MAX_RETRIES})...")
            time.sleep(DELAY)
    if verbose:
        print("Max retries reached. Skipping file.")
    return None

def process_transcripts(input_dir, output_dir, api_key, reading_analysis_dir=None, cut_transcripts_dir=None):
    """Processes all transcript files in the input directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    # Step 1: Preprocess transcripts with reading analysis timestamps if analysis_dir is provided
    if reading_analysis_dir and os.path.exists(reading_analysis_dir):
        if verbose:
            print("Step 1: Cutting transcripts based on reading analysis timestamps...\n")
        if cut_transcripts_dir is None:
            cut_transcripts_dir = output_dir
        preprocess_transcripts_with_reading_analysis(input_dir, cut_transcripts_dir, reading_analysis_dir)
        
        # Step 2: Generate reading analysis CSV
        if verbose:
            print("Step 2: Generating reading analysis CSV...\n")
        generate_reading_analysis_csv(reading_analysis_dir, cut_transcripts_dir)
        
        # After cutting, process the cut_transcripts_dir instead of input_dir
        process_input_dir = cut_transcripts_dir
    else:
        if verbose:
            print("No reading analysis directory provided. Processing all transcripts...\n")
        process_input_dir = input_dir
        
    # Get all .txt files in directory to process
    txt_files = [f for f in os.listdir(process_input_dir) if f.endswith(".txt")]
    
    # set a counter to see if API is working
    errorcounter = 0
    total_errors = 0
    
    # Process each file with a progress bar
    for file_name in tqdm(txt_files, desc="Processing files", unit="file"):
        if file_name.endswith(".txt"):
            input_path = os.path.join(process_input_dir, file_name)
            output_path = os.path.join(output_dir, file_name)

            # Skip if already processed (only if we're processing input_dir directly)
            if os.path.exists(output_path):
                if verbose:
                    print(f"Skipping {file_name}, already processed.")
                continue

            # Read transcript
            with open(input_path, "r", encoding="utf-8") as f:
                transcript = f.read().strip()
            if verbose:
                print(f"Processing {file_name}...")

            # Call Gemini API
            diarized_transcript = call_gemini_api(transcript, api_key)
            if diarized_transcript:
                # Save new transcript
                with open(output_path, "w", encoding="utf-8") as f:
                    f.write(diarized_transcript)
                if verbose:
                    print(f"Saved: {output_path}")
                errorcounter = 0
            else:
                errorcounter += 1
                total_errors += 1
        if errorcounter>5:
            print("API is not working, check API usage or other bugs. Breaking execution...")
            break
    print(f"Runtime finished, total errors: {total_errors}")
    
    ### some interviews are cut into two parts, we need to combine them
    pattern = re.compile(r'^(.*?)_MRI_Speech_Language_part(\d+)of(\d+)\.txt$')
    files_by_id = {}

    for fname in os.listdir(output_dir):
        match = pattern.match(fname)
        if match:
            subject_id, part_num, total_parts = match.groups()
            files_by_id.setdefault(subject_id, {}).setdefault(total_parts, []).append(fname)

    for subject_id, parts_dict in files_by_id.items():
        if '2' in parts_dict and len(parts_dict['2']) == 2:
            combined = []
            parts_dict['2'].sort()
            for f in parts_dict['2']:
                with open(os.path.join(output_dir, f), 'r', encoding='utf-8') as fr:    
                    combined.append(fr.read())
            new_fname = f"{subject_id}_MRI_Speech_Language.txt"
            with open(os.path.join(output_dir, new_fname), 'w', encoding='utf-8') as fw:
                fw.write('\n'.join(combined))
            for f in parts_dict['2']:
                os.remove(os.path.join(output_dir, f))
            print(subject_id, new_fname)
    
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diarize interview transcripts using Gemini API.")
    # parser.add_argument("input_dir", type=str, help="Path to the directory containing transcript .txt files.")
    # parser.add_argument("output_dir", type=str, help="Path to the directory where diarized transcripts will be saved.")

    args = parser.parse_args()
    args.input_dir = r"C:\Users\Aimar\Desktop\hbn_language_task\data\test_videos_transcripts_prompted"
    args.reading_analysis_dir = r"C:\Users\Aimar\Desktop\hbn_language_task\data\test_videos_reading_text_analysis"
    args.cut_transcripts_dir = r"C:\Users\Aimar\Desktop\hbn_language_task\data\test_videos_transcripts_prompted_reading_task_only"
    args.output_dir = r"C:\Users\Aimar\Desktop\hbn_language_task\data\test_videos_transcripts_prompted_reading_task_only_diarized"

    process_transcripts(args.input_dir, args.output_dir, os.getenv("GEMINI_API_KEY_5" \
    ""), args.reading_analysis_dir, args.cut_transcripts_dir)
