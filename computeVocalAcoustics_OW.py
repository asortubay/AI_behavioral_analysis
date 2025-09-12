import os
import argparse
import json
import pandas as pd
import openwillis as ow
from tqdm import tqdm
import numpy as np
from pathlib import Path

def extract_subject_id(filename):
    """Extract subject ID from the filename."""
    return filename.split('_MRI')[0]

def process_speaker1_files(input_dir, output_dir):
    """
    Process all speaker1 audio files in the input directory.
    
    Args:
        input_dir: Directory containing the audio files
        output_dir: Directory to save the results
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Find all speaker1 audio files
    all_files = [f for f in os.listdir(input_dir) if f.endswith('speaker1.wav')]
    
    # Track processed subjects for final summary
    all_subjects_summary = {}
    processed_count = 0
    skipped_count = 0
    
    # Process each file
    for audio_file in tqdm(all_files, desc="Processing audio files"):
        try:
            subject_id = extract_subject_id(audio_file)
            audio_path = os.path.join(input_dir, audio_file)
            
            # Create subject directory
            subject_dir = os.path.join(output_dir, subject_id)
            os.makedirs(subject_dir, exist_ok=True)
            
            # Check if results already exist
            framewise_path = os.path.join(subject_dir, 'framewise.csv')
            summary_path = os.path.join(subject_dir, 'summary.csv')
            
            if os.path.exists(framewise_path) and os.path.exists(summary_path):
                print(f"Skipping {subject_id}: Results already exist")
                # Load existing summary for final compilation
                summary_df = pd.read_csv(summary_path)
                all_subjects_summary[subject_id] = summary_df.to_dict(orient='records')[0]
                skipped_count += 1
                continue
            
            print(f"Processing {subject_id} from {audio_file}")
            
            # Extract vocal acoustics features
            framewise, summary = ow.vocal_acoustics(audio_path=audio_path, voiced_segments=True)
            
            # Convert framewise data to DataFrame
            if not framewise.empty:
                framewise_df = pd.DataFrame(framewise)
                # Save framewise data
                framewise_df.to_csv(framewise_path, index=False)
            
            # Convert summary to DataFrame and save
            if not summary.empty:
                summary_df = pd.DataFrame(summary)
                summary_df.to_csv(summary_path, index=False)
                all_subjects_summary[subject_id] = summary
            
            processed_count += 1
            
        except Exception as e:
            print(f"Error processing {audio_file}: {str(e)}")
    
    # Create final summary dataframe with all subjects
    print("Creating final summary table...")
    final_summary = []
    
    for subject_id, summary_data in all_subjects_summary.items():
        if summary_data and len(summary_data) > 0:
            row = {'subject_id': subject_id}
            row.update(summary_data)
            final_summary.append(row)
    
    # Create final dataframe with all subjects we found
    if final_summary:
        summary_df = pd.DataFrame(final_summary)
        
        # Find all expected subjects (to include missing ones)
        all_subjects = set([extract_subject_id(f) for f in os.listdir(input_dir) 
                           if f.endswith('speaker1.wav')])
        
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
        summary_df.to_csv(os.path.join(output_dir, "T_vocal_acoustics.csv"), index=False)
    
    return processed_count, skipped_count

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="Extract Vocal Acoustics using OpenWillis.")
    parser.add_argument("input_dir", type=str, help="Path to the directory containing separated audio files.")
    parser.add_argument("output_dir", type=str, help="Path to the directory where extracted vocal acoustics will be saved.")

    args = parser.parse_args()

    input_directory = args.input_dir
    output_directory = args.output_dir

    # Process all speaker1 files
    processed, skipped = process_speaker1_files(input_directory, output_directory)
    
    print(f"Processing complete. Results saved to {output_directory}")
    print(f"Processed: {processed} files | Skipped (already existed): {skipped} files")
    print(f"Final summary table saved as: {os.path.join(output_directory, 'T_vocal_features.csv')}")