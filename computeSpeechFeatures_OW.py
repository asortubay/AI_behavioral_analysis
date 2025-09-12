import os
import json
import pandas as pd
import openwillis as ow
from tqdm import tqdm
import numpy as np
from pathlib import Path
import argparse

def extract_subject_id(filename):
    """Extract subject ID from the filename."""
    return filename.split('_MRI')[0]

def process_subject_transcripts(input_dir, output_dir):
    """
    Process all json files in the input directory.
    
    Args:
        input_dir: Directory containing the diarized json files
        output_dir: Directory to save the results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all json transcripts
    all_files = [f for f in os.listdir(input_dir) if f.endswith('.json')]
    
    # Track processed subjects for final summary
    all_subjects_summary = {}
    processed_count = 0
    skipped_count = 0
    
    # Process each file
    for json_file in tqdm(all_files, desc="Processing json transcripts"):
        try:
            subject_id = extract_subject_id(json_file)
            json_path = os.path.join(input_dir, json_file)
            
            # Create subject directory
            subject_dir = os.path.join(output_dir, subject_id)
            os.makedirs(subject_dir, exist_ok=True)
            
            # Check if results already exist
            words_path = os.path.join(subject_dir, 'words.csv')
            turns_path = os.path.join(subject_dir, 'turns.csv')
            summary_path = os.path.join(subject_dir, 'summary.csv')
            
            if os.path.exists(words_path) and os.path.exists(summary_path) and os.path.exists(turns_path):
                print(f"Skipping {subject_id}: Results already exist")
                # Load existing summary for final compilation
                summary_df = pd.read_csv(summary_path)
                all_subjects_summary[subject_id] = summary_df.to_dict(orient='records')[0]
                skipped_count += 1
                continue
            
            print(f"Processing {subject_id} from {json_file}")
            
            transcript_json = json.load(open(json_path))
            # Check if the segments list is empty
            if not transcript_json.get('segments') or len(transcript_json['segments']) == 0:
                # print(f"Skipping {filename}: No segments found in the transcript.")
                continue
            
            # Extract speech features
            words, turns, summary = ow.speech_characteristics(json_conf = transcript_json, speaker_label = 'Subject', min_turn_length = 5, min_coherence_turn_length = 5, option = 'coherence')
           
            # Convert words data to DataFrame
            if not words.empty:
                words_df = pd.DataFrame(words)
                # Save words data
                words_df.to_csv(words_path, index=False)
            
            # Convert turns to DataFrame and save
            if not turns.empty:
                turns_df = pd.DataFrame(turns)
                turns_df.to_csv(turns_path, index=False)
                
            # Convert summary to DataFrame and save
            if not summary.empty:
                summary_df = pd.DataFrame(summary)
                summary_df.to_csv(summary_path, index=False)
                all_subjects_summary[subject_id] = summary
            
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing {json_file}: {str(e)}")
    
    # Create final summary dataframe with all subjects
    print("Creating final summary table...")
    final_summary = []
    
    for subject_id, summary_data in all_subjects_summary.items():
        if summary_data and len(summary_data) > 0:
            row = {'subject_id': subject_id}
            row.update(summary_data)
            final_summary.append(row)
    
    # Create final dataframe with all subjects we found
    if final_summary:  # Check if list is not empty
        summary_df = pd.DataFrame(final_summary)
        
        # Find all expected subjects (to include missing ones)
        all_subjects = set([extract_subject_id(f) for f in os.listdir(input_dir) 
                           if f.endswith('.json')])
        
        # Add missing subjects with NaN values
        existing_subjects = set(summary_df['subject_id'])
        missing_subjects = all_subjects - existing_subjects
        
        for subject in missing_subjects:
            missing_row = {'subject_id': subject}
            # Add NaN for all other columns
            for col in summary_df.columns:
                if col != 'subject_id':
                    missing_row[col] = np.nan
            summary_df = pd.concat([summary_df, pd.DataFrame([missing_row])], ignore_index=True)
        
        # Save the final summary table
        summary_df.to_csv(os.path.join(output_dir, "T_speech_features.csv"), index=False)
    
    return processed_count, skipped_count

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Compute speech features using OpenWillis.")
    parser.add_argument("input_dir", type=str, help="Path to the directory containing diarized transcripts.")
    parser.add_argument("output_dir", type=str, help="Path to the directory where speech features will be saved.")

    args = parser.parse_args()
    
    # Process the transcripts
    processed, skipped = process_subject_transcripts(args.input_dir, args.output_dir)
    
    print(f"Processing complete. Results saved to {args.output_dir}")
    print(f"Processed: {processed} files | Skipped (already existed): {skipped} files")
    print(f"Final summary table saved as: {os.path.join(args.output_dir, 'T_vocal_features.csv')}")