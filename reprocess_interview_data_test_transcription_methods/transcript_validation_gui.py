"""
Transcript Validation and Metrics Benchmarking GUI

Loads Method 1 (raw transcripts) and Method 2 (Gemini diarization) for comparison,
allows manual editing and speaker correction, calculates WER/DER metrics.
"""

import json
import os
import subprocess
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
from pathlib import Path
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
import numpy as np
import librosa
import copy
import csv
import string

# Metrics
from jiwer import wer
try:
    from pyannote.metrics.diarization import DiarizationErrorRate
    PYANNOTE_AVAILABLE = True
except ImportError:
    PYANNOTE_AVAILABLE = False

# Default directories
DEFAULT_METHOD1_DIR = r"Z:\cmi_transcriptions\pyannote_diarized"
DEFAULT_METHOD2_DIR = r"Z:\cmi_transcriptions\diarized"
DEFAULT_AUDIO_DIR = r"Z:\cmi_audio_from_present_interview"
DEFAULT_OUTPUT_DIR = r"C:\Users\Aimar\Desktop\hbn_language_task\data\interview_task\transcripts\01_text_analysis"


@dataclass
class TranscriptData:
    """Container for transcript JSON data."""
    segments: List[Dict]
    raw_json: Dict
    
    @property
    def full_text(self) -> str:
        """Get concatenated text from all segments."""
        return " ".join(seg.get("text", "").strip() for seg in self.segments).strip()
    
    @property
    def speakers_text(self) -> Dict[str, str]:
        """Get concatenated text by speaker."""
        speakers = {}
        for seg in self.segments:
            speaker = seg.get("speaker", "Unknown")
            text = seg.get("text", "").strip()
            if text:
                speakers.setdefault(speaker, []).append(text)
        return {speaker: " ".join(texts) for speaker, texts in speakers.items()}


class DataManager:
    """Manages loading and processing transcript data."""
    
    def __init__(self, method1_dir: str, method2_dir: str, audio_dir: str):
        self.method1_dir = Path(method1_dir)
        self.method2_dir = Path(method2_dir)
        self.audio_dir = Path(audio_dir)
        self.cache = {}
    
    def find_matching_files(self) -> Dict[str, Dict[str, Path]]:
        """
        Find matching Method 1 and Method 2 files by participant ID.
        Returns dict: {participant_id: {"method1": path, "method2": path, "audio": path}}
        """
        matches = {}
        
        # Get all Method 1 files
        method1_files = {f.stem: f for f in self.method1_dir.glob("*.json")}
        method2_files = {f.stem: f for f in self.method2_dir.glob("*.json")}
        
        # Extract participant IDs and match files
        for stem, method1_path in method1_files.items():
            # Extract participant ID (e.g., "NDARAA306NT2" from "NDARAA306NT2_MRI_Present_Interview_part1of1_holistic_landmarks_overlay")
            participant_id = stem.split("_")[0]
            
            # Find corresponding Method 2 file with same participant ID
            method2_path = None
            for method2_stem, m2_path in method2_files.items():
                if method2_stem.startswith(participant_id):
                    method2_path = m2_path
                    break
            
            # Find audio file with same participant ID
            audio_path = None
            for audio_file in self.audio_dir.glob("*.wav"):
                if audio_file.stem.startswith(participant_id):
                    audio_path = audio_file
                    break
            
            if method2_path or audio_path:  # At least one should exist
                matches[participant_id] = {
                    "method1": method1_path,
                    "method2": method2_path,
                    "audio": audio_path
                }
        
        return matches
    
    def load_transcript(self, file_path: Path) -> Optional[TranscriptData]:
        """Load and parse transcript JSON file."""
        if file_path is None:
            return None
        
        if file_path in self.cache:
            return self.cache[file_path]
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            transcript = TranscriptData(
                segments=data.get("segments", []),
                raw_json=data
            )
            self.cache[file_path] = transcript
            return transcript
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None
    
    def save_transcript(self, file_path: Path, transcript: TranscriptData) -> bool:
        """Save transcript back to JSON file, preserving original structure."""
        try:
            # Update segments in the raw JSON
            transcript.raw_json["segments"] = transcript.segments
            
            with open(file_path, 'w', encoding='utf-8') as f:
                json.dump(transcript.raw_json, f, indent=2)
            return True
        except Exception as e:
            print(f"Error saving {file_path}: {e}")
            return False
    
    def load_audio(self, file_path: Path) -> Tuple[np.ndarray, int]:
        """Load audio file and return (samples, sr)."""
        if file_path is None or not file_path.exists():
            return None, None
        
        try:
            y, sr = librosa.load(file_path, sr=None)
            return y, sr
        except Exception as e:
            print(f"Error loading audio {file_path}: {e}")
            return None, None


class MetricsCalculator:
    """Calculate WER and DER metrics."""
    
    @staticmethod
    def calculate_wer(reference: str, hypothesis: str) -> float:
        """
        Calculate Word Error Rate between full concatenated texts.
        Returns float between 0 and 1 (or > 1 for very different texts).
        """
        if not reference or not hypothesis:
            return None
        
        try:
            error_rate = wer(reference, hypothesis)
            return error_rate
        except Exception as e:
            print(f"Error calculating WER: {e}")
            return None
    
    @staticmethod
    def calculate_der(reference_transcript: TranscriptData, 
                     hypothesis_transcript: TranscriptData) -> float:
        """
        Calculate Diarization Error Rate using pyannote.metrics.
        Compares speaker assignments at word level.
        """
        if not PYANNOTE_AVAILABLE:
            return None
        
        try:
            # Build timeline annotations from transcripts
            ref_annotations = _build_annotations(reference_transcript)
            hyp_annotations = _build_annotations(hypothesis_transcript)
            
            der_metric = DiarizationErrorRate()
            der = der_metric(ref_annotations, hyp_annotations)
            return der
        except Exception as e:
            print(f"Error calculating DER: {e}")
            return None
    
    @staticmethod
    def compare_speakers(transcript1: TranscriptData, 
                        transcript2: TranscriptData) -> Dict[str, any]:
        """
        Compare speaker assignments between two transcripts at segment level.
        Returns statistics on agreement/disagreement.
        """
        agreement = 0
        total = 0
        disagreements = []
        
        for i, (seg1, seg2) in enumerate(zip(transcript1.segments, transcript2.segments)):
            speaker1 = seg1.get("speaker", "Unknown")
            speaker2 = seg2.get("speaker", "Unknown")
            
            if speaker1 == speaker2:
                agreement += 1
            else:
                disagreements.append({
                    "segment": i,
                    "text": seg1.get("text", ""),
                    "method1": speaker1,
                    "method2": speaker2
                })
            total += 1
        
        return {
            "total_segments": total,
            "agreed_segments": agreement,
            "agreement_rate": agreement / total if total > 0 else 0,
            "disagreements": disagreements
        }
    
    @staticmethod
    def compute_all_metrics(corrected: TranscriptData, method1: TranscriptData, method2: TranscriptData) -> Dict[str, float]:
        """
        Compute all metrics at once and return as a dictionary.
        Returns dict with keys: wer_m1, wer_m2, der_m1, der_m2, speaker_agree_m1, speaker_agree_m2
        """
        results = {}
        
        # WER calculations
        if method1 and corrected:
            wer_m1 = MetricsCalculator.calculate_wer(corrected.full_text, method1.full_text)
            results["wer_m1"] = wer_m1 if wer_m1 is not None else None
        else:
            results["wer_m1"] = None
        
        if method2 and corrected:
            wer_m2 = MetricsCalculator.calculate_wer(corrected.full_text, method2.full_text)
            results["wer_m2"] = wer_m2 if wer_m2 is not None else None
        else:
            results["wer_m2"] = None
        
        # DER calculations
        if method1 and corrected:
            der_m1 = MetricsCalculator.calculate_der(corrected, method1)
            results["der_m1"] = der_m1 if der_m1 is not None else None
        else:
            results["der_m1"] = None
        
        if method2 and corrected:
            der_m2 = MetricsCalculator.calculate_der(corrected, method2)
            results["der_m2"] = der_m2 if der_m2 is not None else None
        else:
            results["der_m2"] = None
        
        # Speaker agreement
        if method1 and corrected:
            speaker_m1 = MetricsCalculator.compare_speakers(corrected, method1)
            results["speaker_agree_m1"] = speaker_m1["agreement_rate"]
        else:
            results["speaker_agree_m1"] = None
        
        if method2 and corrected:
            speaker_m2 = MetricsCalculator.compare_speakers(corrected, method2)
            results["speaker_agree_m2"] = speaker_m2["agreement_rate"]
        else:
            results["speaker_agree_m2"] = None
        
        return results


def _build_annotations(transcript: TranscriptData):
    """Helper to build pyannote Annotation object from transcript."""
    from pyannote.core import Annotation, Segment as PyannotteSegment
    
    annotation = Annotation()
    for seg in transcript.segments:
        speaker = seg.get("speaker", "Unknown")
        start = seg.get("start", 0)
        end = seg.get("end", 0)
        annotation[PyannotteSegment(start, end)] = speaker
    
    return annotation


class TranscriptEditorPanel(ttk.Frame):
    """Panel for viewing and editing transcript text."""
    
    def __init__(self, parent, title: str, read_only: bool = False, **kwargs):
        super().__init__(parent, **kwargs)
        
        self.title = title
        self.transcript_data: Optional[TranscriptData] = None
        self.current_segment_index = 0
        self.comparison_transcript: Optional[TranscriptData] = None
        self.read_only = read_only
        
        # Title
        title_label = ttk.Label(self, text=title, font=("Arial", 12, "bold"))
        title_label.pack(fill="x", padx=5, pady=5)
        
        # Speaker display frame (visible for all panels)
        speaker_frame = ttk.Frame(self)
        speaker_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Label(speaker_frame, text="Speaker:").pack(side="left")
        
        if not read_only:
            # Editable panel: speaker combobox
            self.speaker_var = tk.StringVar(value="")
            self.speaker_combo = ttk.Combobox(
                speaker_frame, textvariable=self.speaker_var, 
                values=["Interviewer", "Subject", "Unknown"], state="readonly", width=12
            )
            self.speaker_combo.pack(side="left", padx=5)
            # Label to show disagreement (not used for editable)
            self.speaker_label = None
        else:
            # Read-only panels: speaker label with potential disagreement highlighting
            self.speaker_var = None
            self.speaker_combo = None
            self.speaker_label = ttk.Label(speaker_frame, text="", foreground="black", font=("Arial", 10))
            self.speaker_label.pack(side="left", padx=5)
        
        # Text editor
        ttk.Label(self, text="Text:").pack(anchor="w", padx=5, pady=(10, 0))
        self.text_widget = tk.Text(self, height=4, wrap="word", font=("Arial", 10))
        self.text_widget.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Configure text widget tags for highlighting
        self.text_widget.tag_config("difference", background="red", foreground="white")
        
        # Disable text widget if read-only
        if read_only:
            self.text_widget.config(state="disabled")
    
    def load_transcript(self, transcript_data: TranscriptData):
        """Load transcript data for editing."""
        self.transcript_data = transcript_data
        
        if transcript_data and transcript_data.segments:
            self.current_segment_index = 0
            self._display_segment(0)
        else:
            self.text_widget.config(state="disabled")
    
    def set_comparison_transcript(self, transcript: Optional[TranscriptData]):
        """Set transcript to compare against for highlighting differences."""
        self.comparison_transcript = transcript
        # Refresh current display to show highlighting
        if self.transcript_data:
            self._display_segment(self.current_segment_index)
    
    def _display_segment(self, index: int):
        """Display segment at given index with word-level highlighting."""
        if not self.transcript_data or index >= len(self.transcript_data.segments):
            return
        
        seg = self.transcript_data.segments[index]
        self.current_segment_index = index
        
        # Update speaker display
        if self.speaker_var is not None:
            # Editable panel
            self.speaker_var.set(seg.get("speaker", ""))
        elif self.speaker_label is not None:
            # Read-only panel: show speaker and highlight if it disagrees with correction
            speaker_text = seg.get("speaker", "Unknown")
            self.speaker_label.config(text=speaker_text, foreground="black")
            
            # Check if it disagrees with the comparison (correction) transcript
            if self.comparison_transcript and index < len(self.comparison_transcript.segments):
                correction_seg = self.comparison_transcript.segments[index]
                correction_speaker = correction_seg.get("speaker", "Unknown")
                if speaker_text != correction_speaker:
                    # Highlight in red if different from correction
                    self.speaker_label.config(foreground="red", font=("Arial", 10, "bold"))
        
        self.text_widget.config(state="normal")
        self.text_widget.delete("1.0", tk.END)
        
        # Get text and words
        text = seg.get("text", "").strip()
        words = seg.get("words", [])
        
        # Get comparison segment for highlighting
        comparison_text = ""
        if self.comparison_transcript and index < len(self.comparison_transcript.segments):
            comparison_seg = self.comparison_transcript.segments[index]
            comparison_text = comparison_seg.get("text", "").strip()
        
        # Insert text with highlighting for different words
        if words:
            # Insert with word-level highlighting
            for i, word_obj in enumerate(words):
                word_text = word_obj.get("word", "")
                
                # Check if word differs from comparison (ignoring punctuation)
                highlight = False
                if comparison_text:
                    comparison_words = comparison_text.split()
                    if i < len(comparison_words):
                        # Compare words with punctuation stripped
                        word_clean = word_text.translate(str.maketrans('', '', string.punctuation))
                        comparison_clean = comparison_words[i].translate(str.maketrans('', '', string.punctuation))
                        if word_clean != comparison_clean:
                            highlight = True
                
                start_pos = self.text_widget.index(tk.END + "-1c")
                self.text_widget.insert(tk.END, word_text)
                
                if highlight:
                    end_pos = self.text_widget.index(tk.END + "-1c")
                    self.text_widget.tag_add("difference", start_pos, end_pos)
                
                if i < len(words) - 1:
                    self.text_widget.insert(tk.END, " ")
        else:
            # No word-level data, insert plain text
            self.text_widget.insert("1.0", text)
        
        # Disable text widget if read-only
        if self.read_only:
            self.text_widget.config(state="disabled")
    
    def save_segment(self):
        """Save changes to current segment and update individual words."""
        if not self.transcript_data or self.read_only:
            return
        
        seg = self.transcript_data.segments[self.current_segment_index]
        new_text = self.text_widget.get("1.0", tk.END).strip()
        seg["text"] = new_text
        seg["speaker"] = self.speaker_var.get()
        
        # Update words array to match the new text
        if "words" in seg and seg["words"]:
            new_words_list = new_text.split()
            old_words = seg["words"]
            
            # Match new words to old word objects, preserving timing and score where possible
            for i, new_word in enumerate(new_words_list):
                if i < len(old_words):
                    # Update existing word object
                    old_words[i]["word"] = new_word
                else:
                    # Add new word object with placeholder values
                    old_words.append({
                        "word": new_word,
                        "start": 0,
                        "end": 0,
                        "score": 0,
                        "speaker": seg.get("speaker", "")
                    })
            
            # Remove extra words if text is shorter
            while len(old_words) > len(new_words_list):
                old_words.pop()
    
    def get_transcript(self) -> Optional[TranscriptData]:
        """Get edited transcript data."""
        return self.transcript_data


class MetricsPanel(ttk.Frame):
    """Panel for displaying metrics comparison."""
    
    def __init__(self, parent, **kwargs):
        super().__init__(parent, **kwargs)
        
        # Text widget for metrics display
        ttk.Label(self, text="Metrics & Comparison", font=("Arial", 12, "bold")).pack(fill="x", padx=5, pady=5)
        
        self.metrics_text = tk.Text(self, height=15, wrap="word", font=("Courier", 9))
        self.metrics_text.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Buttons
        button_frame = ttk.Frame(self)
        button_frame.pack(fill="x", padx=5, pady=5)
        
        ttk.Button(button_frame, text="Calculate WER (M1 vs Corrected)", command=self._on_calculate_wer_m1).pack(side="left", padx=2)
        ttk.Button(button_frame, text="Calculate WER (M2 vs Corrected)", command=self._on_calculate_wer_m2).pack(side="left", padx=2)
        ttk.Button(button_frame, text="Calculate DER", command=self._on_calculate_der).pack(side="left", padx=2)
        ttk.Button(button_frame, text="Compare Speakers", command=self._on_compare_speakers).pack(side="left", padx=2)
        ttk.Button(button_frame, text="Clear", command=self._on_clear).pack(side="left", padx=2)
        
        self.corrected_transcript: Optional[TranscriptData] = None
        self.method1_transcript: Optional[TranscriptData] = None
        self.method2_transcript: Optional[TranscriptData] = None
    
    def set_transcripts(self, corrected: Optional[TranscriptData], method1: Optional[TranscriptData], method2: Optional[TranscriptData]):
        """Set transcripts for comparison against corrected Method 2."""
        self.corrected_transcript = corrected
        self.method1_transcript = method1
        self.method2_transcript = method2
    
    def _on_calculate_wer_m1(self):
        """Calculate WER for Method 1 vs Corrected Method 2."""
        if not self.corrected_transcript or not self.method1_transcript:
            messagebox.showwarning("Warning", "Both transcripts must be loaded")
            return
        
        ref_text = self.corrected_transcript.full_text
        hyp_text = self.method1_transcript.full_text
        
        wer_score = MetricsCalculator.calculate_wer(ref_text, hyp_text)
        
        output = f"Word Error Rate (Method 1 vs Corrected)\n"
        output += f"{'=' * 50}\n"
        output += f"WER Score: {wer_score:.4f}\n\n"
        output += f"Reference (Corrected) text length: {len(ref_text.split())} words\n"
        output += f"Method 1 text length: {len(hyp_text.split())} words\n"
        
        self.metrics_text.config(state="normal")
        self.metrics_text.delete("1.0", tk.END)
        self.metrics_text.insert("1.0", output)
        self.metrics_text.config(state="disabled")
    
    def _on_calculate_wer_m2(self):
        """Calculate WER for Method 2 vs Corrected Method 2."""
        if not self.corrected_transcript or not self.method2_transcript:
            messagebox.showwarning("Warning", "Both transcripts must be loaded")
            return
        
        ref_text = self.corrected_transcript.full_text
        hyp_text = self.method2_transcript.full_text
        
        wer_score = MetricsCalculator.calculate_wer(ref_text, hyp_text)
        
        output = f"Word Error Rate (Method 2 vs Corrected)\n"
        output += f"{'=' * 50}\n"
        output += f"WER Score: {wer_score:.4f}\n\n"
        output += f"Reference (Corrected) text length: {len(ref_text.split())} words\n"
        output += f"Method 2 text length: {len(hyp_text.split())} words\n"
        
        self.metrics_text.config(state="normal")
        self.metrics_text.delete("1.0", tk.END)
        self.metrics_text.insert("1.0", output)
        self.metrics_text.config(state="disabled")
    
    def _on_compare_speakers(self):
        """Compare speaker assignments for both methods against corrected."""
        if not self.corrected_transcript:
            messagebox.showwarning("Warning", "Corrected transcript must be loaded")
            return
        
        output = f"Speaker Assignment Comparison\n"
        output += f"{'=' * 50}\n\n"
        
        # Compare Method 1 to Corrected
        if self.method1_transcript:
            comparison_m1 = MetricsCalculator.compare_speakers(
                self.corrected_transcript, 
                self.method1_transcript
            )
            output += f"Method 1 vs Corrected:\n"
            output += f"{'-' * 50}\n"
            output += f"Total segments: {comparison_m1['total_segments']}\n"
            output += f"Agreed segments: {comparison_m1['agreed_segments']}\n"
            output += f"Agreement rate: {comparison_m1['agreement_rate']:.2%}\n\n"
        
        # Compare Method 2 to Corrected
        if self.method2_transcript:
            comparison_m2 = MetricsCalculator.compare_speakers(
                self.corrected_transcript, 
                self.method2_transcript
            )
            output += f"Method 2 vs Corrected:\n"
            output += f"{'-' * 50}\n"
            output += f"Total segments: {comparison_m2['total_segments']}\n"
            output += f"Agreed segments: {comparison_m2['agreed_segments']}\n"
            output += f"Agreement rate: {comparison_m2['agreement_rate']:.2%}\n"
        
        self.metrics_text.config(state="normal")
        self.metrics_text.delete("1.0", tk.END)
        self.metrics_text.insert("1.0", output)
        self.metrics_text.config(state="disabled")
    
    def _on_clear(self):
        """Clear metrics display."""
        self.metrics_text.config(state="normal")
        self.metrics_text.delete("1.0", tk.END)
        self.metrics_text.config(state="disabled")
    
    def _on_calculate_der(self):
        """Calculate DER for both methods against corrected Method 2."""
        if not PYANNOTE_AVAILABLE:
            messagebox.showerror("Error", "pyannote.metrics not installed. Install with: pip install pyannote.metrics")
            return
        
        if not self.corrected_transcript:
            messagebox.showwarning("Warning", "Corrected transcript must be loaded")
            return
        
        output = f"Diarization Error Rate (DER) Comparison\n"
        output += f"{'=' * 50}\n\n"
        
        # Calculate DER for Method 1 vs Corrected
        if self.method1_transcript:
            der_m1 = MetricsCalculator.calculate_der(
                self.corrected_transcript,
                self.method1_transcript
            )
            if der_m1 is not None:
                output += f"Method 1 vs Corrected:\n"
                output += f"{'-' * 50}\n"
                output += f"DER Score: {der_m1:.4f}\n"
                output += f"(Lower is better, 0 = perfect match)\n\n"
            else:
                output += f"Method 1 vs Corrected:\n"
                output += f"{'-' * 50}\n"
                output += f"Could not calculate DER (check transcript format)\n\n"
        
        # Calculate DER for Method 2 vs Corrected
        if self.method2_transcript:
            der_m2 = MetricsCalculator.calculate_der(
                self.corrected_transcript,
                self.method2_transcript
            )
            if der_m2 is not None:
                output += f"Method 2 vs Corrected:\n"
                output += f"{'-' * 50}\n"
                output += f"DER Score: {der_m2:.4f}\n"
                output += f"(Lower is better, 0 = perfect match)\n"
            else:
                output += f"Method 2 vs Corrected:\n"
                output += f"{'-' * 50}\n"
                output += f"Could not calculate DER (check transcript format)\n"
        
        self.metrics_text.config(state="normal")
        self.metrics_text.delete("1.0", tk.END)
        self.metrics_text.insert("1.0", output)
        self.metrics_text.config(state="disabled")


class MainWindow(tk.Tk):
    """Main application window."""
    
    def __init__(self, method1_dir: str = None, method2_dir: str = None, audio_dir: str = None, output_dir: str = None):
        super().__init__()
        self.title("Transcript Validation & Metrics Benchmarking")
        self.geometry("1400x900")
        
        # Use defaults if not provided
        method1_dir = method1_dir or DEFAULT_METHOD1_DIR
        method2_dir = method2_dir or DEFAULT_METHOD2_DIR
        audio_dir = audio_dir or DEFAULT_AUDIO_DIR
        output_dir = output_dir or DEFAULT_OUTPUT_DIR
        
        # Allow user to override defaults
        dialog_result = messagebox.askyesno(
            "Directory Configuration",
            "Use default directories? (Yes) or Select manually? (No)"
        )
        
        if not dialog_result:
            # User wants to select manually
            method1_dir = filedialog.askdirectory(title="Select Method 1 Directory (Raw Transcripts)")
            method2_dir = filedialog.askdirectory(title="Select Method 2 Directory (Gemini Diarization)")
            audio_dir = filedialog.askdirectory(title="Select Audio Directory")
            output_dir = filedialog.askdirectory(title="Select Output Directory (for corrected transcripts)")
        
        # Verify directories exist
        missing_dirs = []
        for dir_path, name in [(method1_dir, "Method 1"), (method2_dir, "Method 2"), 
                                (audio_dir, "Audio"), (output_dir, "Output")]:
            if not Path(dir_path).exists():
                missing_dirs.append(f"{name}: {dir_path}")
        
        if missing_dirs:
            messagebox.showwarning("Warning", f"Some directories don't exist:\n" + "\n".join(missing_dirs))
        
        if not (method1_dir and method2_dir and audio_dir and output_dir):
            messagebox.showerror("Error", "All directories required")
            self.destroy()
            return
        
        # Create output subdirectories
        self.output_method1_dir = Path(output_dir) / "method1_corrected"
        self.output_method2_dir = Path(output_dir) / "method2_corrected"
        self.output_method1_dir.mkdir(parents=True, exist_ok=True)
        self.output_method2_dir.mkdir(parents=True, exist_ok=True)
        
        self.data_manager = DataManager(method1_dir, method2_dir, audio_dir)
        self.current_participant: Optional[str] = None
        self.current_files: Dict = {}
        self.all_matches: Dict = {}  # Cache for matched files
        
        # Layout
        self._build_ui()
        self._load_participant_list()
    
    def _build_ui(self):
        """Build main UI."""
        # Top frame: Participant selector
        top_frame = ttk.Frame(self)
        top_frame.pack(fill="x", padx=10, pady=10)
        
        ttk.Label(top_frame, text="Participant:", font=("Arial", 11, "bold")).pack(side="left")
        self.participant_var = tk.StringVar()
        self.participant_combo = ttk.Combobox(
            top_frame, textvariable=self.participant_var, state="readonly", width=20
        )
        self.participant_combo.pack(side="left", padx=5)
        self.participant_combo.bind("<<ComboboxSelected>>", self._on_participant_selected)
        
        ttk.Button(top_frame, text="Save All Changes", command=self._on_save_all).pack(side="left", padx=5)
        ttk.Button(top_frame, text="Open Audio File", command=self._on_open_audio).pack(side="left", padx=5)
        ttk.Button(top_frame, text="Batch Compute Metrics to CSV", command=self._on_batch_compute_metrics).pack(side="left", padx=5)
        
        # Navigation frame: Segment navigation (shared)
        nav_frame = ttk.Frame(self)
        nav_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        ttk.Label(nav_frame, text="Segment Navigation:", font=("Arial", 10, "bold")).pack(side="left", padx=(0, 10))
        self.segment_label = ttk.Label(nav_frame, text="0/0", font=("Arial", 10))
        self.segment_label.pack(side="left", padx=5)
        
        ttk.Button(nav_frame, text="◄ Previous", command=self._prev_segment).pack(side="left", padx=2)
        ttk.Button(nav_frame, text="Next ►", command=self._next_segment).pack(side="left", padx=2)
        ttk.Button(nav_frame, text="Save Segment", command=self._save_segment).pack(side="left", padx=5)
        ttk.Button(nav_frame, text="Add Before", command=self._add_segment_before).pack(side="left", padx=2)
        ttk.Button(nav_frame, text="Add After", command=self._add_segment_after).pack(side="left", padx=2)
        
        # Main content: Three panels for transcripts + metrics
        content_frame = ttk.Frame(self)
        content_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Left: Method 1 editor (read-only)
        left_frame = ttk.LabelFrame(content_frame, text="Method 1: Raw Transcript", padding=5)
        left_frame.pack(side="left", fill="both", expand=True, padx=(0, 2))
        
        self.method1_panel = TranscriptEditorPanel(left_frame, "Method 1", read_only=True)
        self.method1_panel.pack(fill="both", expand=True)
        
        # Center: Method 2 editor (read-only)
        center_frame = ttk.LabelFrame(content_frame, text="Method 2: Gemini Diarization", padding=5)
        center_frame.pack(side="left", fill="both", expand=True, padx=2)
        
        self.method2_panel = TranscriptEditorPanel(center_frame, "Method 2", read_only=True)
        self.method2_panel.pack(fill="both", expand=True)
        
        # Right: Corrected Method 2 (editable)
        right_frame = ttk.LabelFrame(content_frame, text="Method 2: Corrected (Reference) - EDITABLE", padding=5)
        right_frame.pack(side="left", fill="both", expand=True, padx=(2, 0))
        
        self.corrected_panel = TranscriptEditorPanel(right_frame, "Corrected M2", read_only=False)
        self.corrected_panel.pack(fill="both", expand=True)
        
        # Bottom: Metrics panel
        metrics_frame = ttk.LabelFrame(self, text="Metrics & Comparison (vs. Corrected Method 2)", padding=5)
        metrics_frame.pack(fill="both", expand=True, padx=10, pady=(10, 10))
        
        self.metrics_panel = MetricsPanel(metrics_frame)
        self.metrics_panel.pack(fill="both", expand=True)
    
    def _load_participant_list(self):
        """Load list of matched participants and mark those with saved corrected transcripts."""
        self.all_matches = self.data_manager.find_matching_files()
        participants = sorted(self.all_matches.keys())
        
        # Check which participants have saved corrected transcripts
        participants_with_status = []
        for participant in participants:
            # Check if corrected Method 2 file exists for this participant
            method2_file = self.all_matches[participant].get("method2")
            if method2_file:
                original_filename = Path(method2_file).name
                corrected_path = self.output_method2_dir / original_filename
                if corrected_path.exists():
                    # Add tick mark before participant ID
                    participants_with_status.append(f"✓ {participant}")
                else:
                    participants_with_status.append(participant)
            else:
                participants_with_status.append(participant)
        
        self.participant_combo.config(values=participants_with_status)
        
        if participants_with_status:
            self.participant_combo.current(0)
            self._on_participant_selected(None)
    
    def _on_participant_selected(self, event):
        """Load transcripts for selected participant."""
        participant_display = self.participant_var.get()
        if not participant_display:
            return
        
        # Remove tick mark if present to get actual participant ID
        participant = participant_display.replace("✓ ", "").strip()
        
        self.current_files = self.all_matches.get(participant, {})
        self.current_participant = participant
        
        # Load transcripts
        method1_transcript = self.data_manager.load_transcript(self.current_files.get("method1"))
        method2_transcript = self.data_manager.load_transcript(self.current_files.get("method2"))
        
        # Try to load corrected Method 2 from output directory
        corrected_method2_transcript = None
        if self.current_files.get("method2"):
            original_filename = Path(self.current_files["method2"]).name
            corrected_path = self.output_method2_dir / original_filename
            if corrected_path.exists():
                corrected_method2_transcript = self.data_manager.load_transcript(corrected_path)
        
        # If no corrected version exists, create an independent copy of Method 2 as base
        if not corrected_method2_transcript and method2_transcript:
            corrected_method2_transcript = TranscriptData(
                segments=copy.deepcopy(method2_transcript.segments),
                raw_json=copy.deepcopy(method2_transcript.raw_json)
            )
        
        # Update panels
        self.method1_panel.load_transcript(method1_transcript)
        self.method2_panel.load_transcript(method2_transcript)
        self.corrected_panel.load_transcript(corrected_method2_transcript)
        
        # Set comparison transcripts for highlighting differences (compare to corrected Method 2)
        self.method1_panel.set_comparison_transcript(corrected_method2_transcript)
        self.method2_panel.set_comparison_transcript(corrected_method2_transcript)
        self.corrected_panel.set_comparison_transcript(None)  # No comparison for reference
        
        # Set metrics to compare both against corrected Method 2
        self.metrics_panel.set_transcripts(corrected_method2_transcript, method1_transcript, method2_transcript)
        self._update_segment_label()
    
    def _update_segment_label(self):
        """Update the segment navigation label."""
        if self.corrected_panel.transcript_data:
            total = len(self.corrected_panel.transcript_data.segments)
            current = self.corrected_panel.current_segment_index + 1
            self.segment_label.config(text=f"{current}/{total}")
    
    def _prev_segment(self):
        """Navigate to previous segment across all panels and save current segment."""
        if self.corrected_panel.current_segment_index > 0:
            # Save current segment without showing popup
            self.corrected_panel.save_segment()
            # Refresh comparison highlighting after save
            self.method1_panel.set_comparison_transcript(self.corrected_panel.transcript_data)
            self.method2_panel.set_comparison_transcript(self.corrected_panel.transcript_data)
            
            new_index = self.corrected_panel.current_segment_index - 1
            self.method1_panel._display_segment(new_index)
            self.method2_panel._display_segment(new_index)
            self.corrected_panel._display_segment(new_index)
            self._update_segment_label()
    
    def _next_segment(self):
        """Navigate to next segment across all panels and save current segment."""
        if self.corrected_panel.transcript_data and self.corrected_panel.current_segment_index < len(self.corrected_panel.transcript_data.segments) - 1:
            # Save current segment without showing popup
            self.corrected_panel.save_segment()
            # Refresh comparison highlighting after save
            self.method1_panel.set_comparison_transcript(self.corrected_panel.transcript_data)
            self.method2_panel.set_comparison_transcript(self.corrected_panel.transcript_data)
            
            new_index = self.corrected_panel.current_segment_index + 1
            self.method1_panel._display_segment(new_index)
            self.method2_panel._display_segment(new_index)
            self.corrected_panel._display_segment(new_index)
            self._update_segment_label()
    
    def _save_segment(self):
        """Save the current segment (only corrected panel is editable)."""
        self.corrected_panel.save_segment()
        messagebox.showinfo("Saved", "Segment and words updated (not yet written to file)")
        # Refresh comparison highlighting after save
        self.method1_panel.set_comparison_transcript(self.corrected_panel.transcript_data)
        self.method2_panel.set_comparison_transcript(self.corrected_panel.transcript_data)
    
    def _add_segment_before(self):
        """Add a new segment before the current segment in the corrected panel."""
        if not self.corrected_panel.transcript_data:
            messagebox.showwarning("Warning", "No transcript loaded")
            return
        
        # First save the current segment
        self.corrected_panel.save_segment()
        
        # Create new segment
        new_segment = {
            "start": 0.0,
            "end": 0.0,
            "text": "",
            "speaker": "Interviewer",
            "words": []
        }
        
        current_idx = self.corrected_panel.current_segment_index
        
        # Insert before current segment in all three panels
        self.corrected_panel.transcript_data.segments.insert(current_idx, new_segment.copy())
        
        # Add empty segment to method 1 and 2 panels to keep indices synchronized
        if self.method1_panel.transcript_data:
            self.method1_panel.transcript_data.segments.insert(current_idx, new_segment.copy())
        if self.method2_panel.transcript_data:
            self.method2_panel.transcript_data.segments.insert(current_idx, new_segment.copy())
        
        # Display the new segment (same index now points to the new segment)
        self.corrected_panel._display_segment(current_idx)
        self.method1_panel._display_segment(current_idx)
        self.method2_panel._display_segment(current_idx)
        self._update_segment_label()
        messagebox.showinfo("Added", "New segment added before current segment in all panels. Add your text and click 'Save Segment'")
    
    def _add_segment_after(self):
        """Add a new segment after the current segment in the corrected panel."""
        if not self.corrected_panel.transcript_data:
            messagebox.showwarning("Warning", "No transcript loaded")
            return
        
        # First save the current segment
        self.corrected_panel.save_segment()
        
        # Create new segment
        new_segment = {
            "start": 0.0,
            "end": 0.0,
            "text": "",
            "speaker": "",
            "words": []
        }
        
        current_idx = self.corrected_panel.current_segment_index
        
        # Insert after current segment in all three panels
        self.corrected_panel.transcript_data.segments.insert(current_idx + 1, new_segment.copy())
        
        # Add empty segment to method 1 and 2 panels to keep indices synchronized
        if self.method1_panel.transcript_data:
            self.method1_panel.transcript_data.segments.insert(current_idx + 1, new_segment.copy())
        if self.method2_panel.transcript_data:
            self.method2_panel.transcript_data.segments.insert(current_idx + 1, new_segment.copy())
        
        # Display the new segment
        self.corrected_panel._display_segment(current_idx + 1)
        self.method1_panel._display_segment(current_idx + 1)
        self.method2_panel._display_segment(current_idx + 1)
        self._update_segment_label()
        messagebox.showinfo("Added", "New segment added after current segment in all panels. Add your text and click 'Save Segment'")
    
    def _on_open_audio(self):
        """Open the audio file with the default system audio player."""
        audio_path = self.current_files.get("audio")
        if not audio_path or not Path(audio_path).exists():
            messagebox.showwarning("Warning", "No audio file found for this participant")
            return
        
        try:
            audio_path = Path(audio_path)
            if os.name == 'nt':  # Windows
                os.startfile(str(audio_path))
            elif os.name == 'posix':  # macOS and Linux
                subprocess.Popen(['open', str(audio_path)])
        except Exception as e:
            messagebox.showerror("Error", f"Could not open audio file: {e}")
    
    def _on_save_all(self):
        """Save all changes to output directory (never modifying original files)."""
        method1_data = self.method1_panel.get_transcript()
        method2_data = self.method2_panel.get_transcript()
        corrected_data = self.corrected_panel.get_transcript()
        
        if not self.current_participant:
            messagebox.showwarning("Warning", "No participant selected")
            return
        
        success = True
        saved_files = []
        
        # Save Method 1 transcript to output directory
        if method1_data and self.current_files.get("method1"):
            original_filename = Path(self.current_files["method1"]).name
            output_path = self.output_method1_dir / original_filename
            if self.data_manager.save_transcript(output_path, method1_data):
                saved_files.append(f"Method 1: {output_path}")
            else:
                success = False
        
        # Save Method 2 transcript to output directory
        if method2_data and self.current_files.get("method2"):
            original_filename = Path(self.current_files["method2"]).name
            output_path = self.output_method2_dir / original_filename
            if self.data_manager.save_transcript(output_path, method2_data):
                saved_files.append(f"Method 2: {output_path}")
            else:
                success = False
        
        # Save Corrected Method 2 transcript to output directory
        if corrected_data and self.current_files.get("method2"):
            original_filename = Path(self.current_files["method2"]).name
            output_path = self.output_method2_dir / original_filename
            if self.data_manager.save_transcript(output_path, corrected_data):
                saved_files.append(f"Corrected Method 2: {output_path}")
            else:
                success = False
        
        if success and saved_files:
            msg = "Files saved to output directory:\n\n" + "\n".join(saved_files)
            messagebox.showinfo("Success", msg)
            # Refresh participant list to update tick marks
            self._load_participant_list()
        elif success:
            messagebox.showwarning("Warning", "No data to save")
        else:
            messagebox.showerror("Error", "Some files could not be saved")
    
    def _on_batch_compute_metrics(self):
        """Compute all metrics for all participants with corrected Method 2 files and save to CSV."""
        # Ask user for output CSV file
        csv_file = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialfile="transcript_metrics.csv"
        )
        
        if not csv_file:
            return
        
        try:
            # Find all corrected Method 2 files
            corrected_files = list(self.output_method2_dir.glob("*.json"))
            if not corrected_files:
                messagebox.showwarning("Warning", f"No corrected Method 2 files found in {self.output_method2_dir}")
                return
            
            # Prepare CSV data
            rows = []
            header = ["Participant ID", "WER M1 vs Corrected", "WER M2 vs Corrected", 
                     "DER M1 vs Corrected", "DER M2 vs Corrected", "Speaker Agreement M1", "Speaker Agreement M2"]
            
            # Process each corrected file
            for corrected_file in corrected_files:
                # Extract participant ID from filename
                stem = corrected_file.stem
                participant_id = stem.split("_")[0]
                
                # Get file paths from all_matches
                file_paths = self.all_matches.get(participant_id)
                if not file_paths:
                    continue
                
                # Load transcripts
                method1_transcript = self.data_manager.load_transcript(file_paths.get("method1"))
                method2_transcript = self.data_manager.load_transcript(file_paths.get("method2"))
                corrected_transcript = self.data_manager.load_transcript(corrected_file)
                
                if not corrected_transcript:
                    continue
                
                # Compute all metrics
                metrics = MetricsCalculator.compute_all_metrics(
                    corrected_transcript,
                    method1_transcript,
                    method2_transcript
                )
                
                # Build row
                row = [participant_id]
                row.append(f"{metrics['wer_m1']:.4f}" if metrics['wer_m1'] is not None else "N/A")
                row.append(f"{metrics['wer_m2']:.4f}" if metrics['wer_m2'] is not None else "N/A")
                row.append(f"{metrics['der_m1']:.4f}" if metrics['der_m1'] is not None else "N/A")
                row.append(f"{metrics['der_m2']:.4f}" if metrics['der_m2'] is not None else "N/A")
                row.append(f"{metrics['speaker_agree_m1']:.4f}" if metrics['speaker_agree_m1'] is not None else "N/A")
                row.append(f"{metrics['speaker_agree_m2']:.4f}" if metrics['speaker_agree_m2'] is not None else "N/A")
                
                rows.append(row)
            
            # Write CSV
            with open(csv_file, 'w', newline='', encoding='utf-8') as f:
                writer = csv.writer(f)
                writer.writerow(header)
                writer.writerows(rows)
            
            messagebox.showinfo("Success", f"Metrics computed for {len(rows)} participants.\nSaved to: {csv_file}")
        
        except Exception as e:
            messagebox.showerror("Error", f"Failed to compute batch metrics: {e}")


if __name__ == "__main__":
    app = MainWindow()
    app.mainloop()
