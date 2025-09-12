import os
import argparse
import time
from google import genai
from tqdm import tqdm
import re

# Gemini model parameters
MODEL_NAME = "gemini-2.0-flash"
MAX_RETRIES = 3  # Number of retries if the API fails
DELAY = 2  # Seconds to wait between retries

# System prompt for Gemini
SYSTEM_PROMPT = """You are given a transcript of an interview between a clinician (interviewer) and a subject. The interviewer asks questions from a predefined list, though slight variations may occur. Your task is to label the dialogue as either "Interviewer:" or "Subject:", removing timestamps while preserving the dialogue structure.

### Instructions:
1. Identify lines spoken by the **interviewer** based on a predefined question list. These questions may have slight variations but follow the same general intent.
2. Label all other responses as **"Subject:"**
3. Remove timestamps but keep the text structure intact.
4. Ensure proper formatting with a new line between each labeled turn.

### Example Input:

0.709 -> 2.83: I hope you enjoyed the last movie about the puppy.  
2.95 -> 3.851: Have you seen it before?  
4.451 -> 4.751: No.  
5.231 -> 6.792: Can you tell me what happened in the movie?  
6.852 -> 8.013: Try to tell the whole story.  
8.093 -> 11.515: Remember that stories have a beginning, things that happen, and an ending.  
12.895 -> 15.877: I didn't watch the whole thing though.  
16.637 -> 18.478: Can you tell me about the part that you did watch?  
19.139 -> 19.639: Okay.  

### Example Output:

Interviewer: I hope you enjoyed the last movie about the puppy.  
Interviewer: Have you seen it before?  
Subject: No.  
Interviewer: Can you tell me what happened in the movie?  
Interviewer: Try to tell the whole story.  
Interviewer: Remember that stories have a beginning, things that happen, and an ending.  
Subject: I didn't watch the whole thing though.  
Interviewer: Can you tell me about the part that you did watch?  
Subject: Okay.

### If the transcript contains these specific sentences, you will label them as "clip" instead of interviewer or subject:

Whoa! Cool!
You gotta be kidding me
Get lost
Mom, I'll/we'll be outside

### The interviewer question list:

So I hope you enjoyed the last movie. Have you seen it before?
So can you tell me what happened in the movie? Try to tell the whole story. Remember that stories have a beginning, things that happen, and an ending.
Do you remember anything else from the story?
So what are some of the things you liked about the movie?
What are some of the things you didn't like about the movie?
Who gave the boy a box?
What was in the box?
What was the boy doing before he got the box?
What was the puppy playing with?
How are the puppy and the boy the same?
So in the movie, who is missing a leg? The boy, the puppy, both the boy and the puppy, or no one?
So we're going to watch a short clip from the movie and then we'll talk about it.
How do you think the puppy was feeling?
How do you think the boy was feeling?
And how did you feel while you were watching that part?
How do you think the puppy was feeling?
How do you think the boy was feeling?
And how did you feel while you were watching that part?
How do you think the puppy was feeling?
How do you think the boy was feeling?
And how did you feel while you were watching that part?
How do you think the puppy was feeling?
How do you think the boy was feeling?
And how did you feel while you were watching this part?
Great, thank you

### In some cases, the interviewer may ask additional questions or make comments. You should label these as "Interviewer:" as well using their specific context.

### The transcript:
"""
verbose = False
def call_gemini_api(transcript, api_key):
    """Calls the Gemini API to perform diarization."""
    client = genai.Client(api_key=api_key)
    for attempt in range(MAX_RETRIES):
        try:
            response = client.models.generate_content(
                model=MODEL_NAME, 
                contents=SYSTEM_PROMPT + "\n\n" + transcript
            )
            return response.text
        except Exception as e:
            if verbose:
                print(f"API error: {e}, retrying ({attempt + 1}/{MAX_RETRIES})...")
            time.sleep(DELAY)
    if verbose:
        print("Max retries reached. Skipping file.")
    return None

def process_transcripts(input_dir, output_dir, api_key):
    """Processes all transcript files in the input directory."""
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
        
    # Get all .txt files in directory
    txt_files = [f for f in os.listdir(input_dir) if f.endswith(".txt")]
    
    # set a counter to see if API is working
    errorcounter = 0
    total_errors = 0
    
    # Process each file with a progress bar
    for file_name in tqdm(txt_files, desc="Processing files", unit="file"):
        if file_name.endswith(".txt"):
            input_path = os.path.join(input_dir, file_name)
            output_path = os.path.join(output_dir, file_name)

            # Skip if already processed
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
    pattern = re.compile(r'^(.*?)_MRI_Present_Interview_part(\d+)of(\d+)_holistic_landmarks_overlay\.txt$')
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
            new_fname = f"{subject_id}_MRI_Present_Interview_part1of1_holistic_landmarks_overlay.txt"
            with open(os.path.join(output_dir, new_fname), 'w', encoding='utf-8') as fw:
                fw.write('\n'.join(combined))
            for f in parts_dict['2']:
                os.remove(os.path.join(output_dir, f))
            print(subject_id, new_fname)
    
        

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Diarize interview transcripts using Gemini API.")
    parser.add_argument("input_dir", type=str, help="Path to the directory containing transcript .txt files.")
    parser.add_argument("output_dir", type=str, help="Path to the directory where diarized transcripts will be saved.")

    args = parser.parse_args()

    process_transcripts(args.input_dir, args.output_dir,os.getenv("GEMINI_API_KEY_0"))
