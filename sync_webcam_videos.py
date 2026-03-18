"""
Sync two webcam videos using a clap (sharp audio transient) in the first few seconds.

Strategy:
  1. Extract the first N seconds of audio from each video via ffmpeg (fast, reliable).
  2. Compute a short-time energy envelope and detect the sharpest transient (the clap).
  3. Align the two videos so the clap frames coincide.
  4. Trim both to the same duration and export synced files + a side-by-side check.

Dependencies: numpy, scipy, ffmpeg (must be on PATH).
"""

import os
import subprocess
import struct
import numpy as np
from scipy.signal import find_peaks


# ---------------------------------------------------------------------------
# Audio extraction (ffmpeg, no Python media library needed)
# ---------------------------------------------------------------------------

def extract_audio_ffmpeg(video_path: str, duration: float = 10.0,
                         sample_rate: int = 96000) -> np.ndarray:
    """
    Extract the first *duration* seconds of audio from *video_path* as a
    mono float32 numpy array using ffmpeg over a pipe.
    """
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error",
        "-i", video_path,
        "-t", str(duration),
        "-ac", "1",                     # mono
        "-ar", str(sample_rate),        # resample
        "-f", "s16le",                  # raw PCM signed 16-bit little-endian
        "-acodec", "pcm_s16le",
        "pipe:1",                       # output to stdout
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg failed for {video_path}:\n{result.stderr.decode()}"
        )
    raw = result.stdout
    n_samples = len(raw) // 2
    samples = struct.unpack(f"<{n_samples}h", raw)
    audio = np.array(samples, dtype=np.float32) / 32768.0
    return audio


# ---------------------------------------------------------------------------
# Clap detection
# ---------------------------------------------------------------------------

def detect_clap(audio: np.ndarray, sample_rate: int = 96000,
                window_ms: float = 2.0, search_seconds: float = 8.0) -> float:
    """
    Detect the most prominent transient (clap) within the first
    *search_seconds* of the audio signal.

    Returns the clap time in seconds.

    Algorithm:
      - Compute a short-time energy envelope (RMS in small sliding windows).
      - Differentiate the envelope to emphasise sudden energy jumps.
      - Pick the highest peak in the derivative as the coarse clap onset.
      - Refine to sample-level precision by finding the exact sample where
        the amplitude first exceeds a fraction of the peak.
    """
    # Limit search range
    max_samples = int(search_seconds * sample_rate)
    audio = audio[:max_samples]

    # Short-time energy envelope (RMS) – small window for precision
    win = max(int(window_ms / 1000.0 * sample_rate), 4)
    hop = win // 2
    n_frames = (len(audio) - win) // hop + 1
    envelope = np.array([
        np.sqrt(np.mean(audio[i * hop : i * hop + win] ** 2))
        for i in range(n_frames)
    ])

    # Onset-strength: first-order difference, half-wave rectified
    onset_strength = np.diff(envelope)
    onset_strength = np.maximum(onset_strength, 0)

    # Find peaks in onset strength
    median_val = np.median(onset_strength)
    threshold = max(median_val * 5, 0.005)

    peaks, properties = find_peaks(
        onset_strength,
        height=threshold,
        distance=int(0.05 * sample_rate / hop),   # at least 50 ms apart
    )

    if len(peaks) == 0:
        peak_idx = int(np.argmax(onset_strength))
        print("  ⚠ No clear peak found – using global max as clap position.")
    else:
        best = int(np.argmax(properties["peak_heights"]))
        peak_idx = peaks[best]

    # Coarse clap position (in samples)
    coarse_sample = peak_idx * hop

    # --- Sample-level refinement ---
    # Look in a window around the coarse position and find the exact onset:
    # the first sample whose absolute value exceeds 20% of the local peak.
    margin = int(0.02 * sample_rate)  # ±20 ms search window
    region_start = max(0, coarse_sample - margin)
    region_end = min(len(audio), coarse_sample + margin)
    region = np.abs(audio[region_start:region_end])

    local_peak = np.max(region)
    onset_threshold = local_peak * 0.2
    onset_indices = np.where(region >= onset_threshold)[0]

    if len(onset_indices) > 0:
        refined_sample = region_start + onset_indices[0]
    else:
        refined_sample = coarse_sample

    clap_time = refined_sample / sample_rate
    return clap_time


# ---------------------------------------------------------------------------
# Video trimming with ffmpeg (stream-copy = no re-encode, very fast)
# ---------------------------------------------------------------------------

def trim_video_ffmpeg(input_path: str, output_path: str,
                      start: float, n_frames: int | None = None,
                      duration: float | None = None) -> None:
    """
    Trim a video with ffmpeg using re-encoding for frame-accurate cuts.

    If *n_frames* is given the output is limited to exactly that many
    video frames (via ``-frames:v``), guaranteeing both synced videos
    have an identical frame count.
    """
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-ss", f"{start:.6f}",
        "-i", input_path,
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k",
        "-avoid_negative_ts", "make_zero",
    ]
    if n_frames is not None:
        cmd += ["-frames:v", str(n_frames)]
    elif duration is not None:
        cmd += ["-t", f"{duration:.6f}"]
    cmd.append(output_path)
    result = subprocess.run(cmd, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg trim failed:\n{result.stderr.decode()}"
        )


def get_duration_ffprobe(path: str) -> float:
    """Get the duration of a media file in seconds via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-show_entries", "format=duration",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return float(result.stdout.strip())


def get_fps_ffprobe(path: str) -> float:
    """Get the video frame rate via ffprobe (as a float)."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-show_entries", "stream=r_frame_rate",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    # r_frame_rate comes as e.g. "30/1" or "30000/1001"
    text = result.stdout.decode().strip()
    num, den = text.split("/")
    return float(num) / float(den)


def get_frame_count(path: str) -> int:
    """Count the exact number of video frames via ffprobe."""
    cmd = [
        "ffprobe", "-v", "error",
        "-select_streams", "v:0",
        "-count_frames",
        "-show_entries", "stream=nb_read_frames",
        "-of", "default=noprint_wrappers=1:nokey=1",
        path,
    ]
    result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    return int(result.stdout.decode().strip())


def make_side_by_side(path1: str, path2: str, output_path: str,
                      height: int = 480) -> None:
    """Create a side-by-side verification video (re-encodes at low res)."""
    filter_str = (
        f"[0:v]scale=-2:{height}[l];"
        f"[1:v]scale=-2:{height}[r];"
        f"[l][r]hstack=inputs=2[v]"
    )
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", path1,
        "-i", path2,
        "-filter_complex", filter_str,
        "-map", "[v]",
        "-map", "0:a?",
        "-c:v", "libx264", "-preset", "fast", "-crf", "23",
        "-c:a", "aac", "-b:a", "128k",
        "-shortest",
        output_path,
    ]
    result = subprocess.run(cmd, stderr=subprocess.PIPE)
    if result.returncode != 0:
        raise RuntimeError(
            f"ffmpeg side-by-side failed:\n{result.stderr.decode()}"
        )


def synced_output_path(original_path: str) -> str:
    """Derive the synced output path: same dir/name with '_synced.avi'."""
    base, _ = os.path.splitext(original_path)
    return base + "_synced.avi"


# ---------------------------------------------------------------------------
# Main pipeline
# ---------------------------------------------------------------------------

def refine_offset_xcorr(audio1: np.ndarray, audio2: np.ndarray,
                       clap1_sample: int, clap2_sample: int,
                       sample_rate: int, window_ms: float = 80.0) -> float:
    """
    Given two audio signals and approximate clap positions (in samples),
    extract short windows around each clap and cross-correlate them to
    find the sub-sample precise offset.

    Returns the refined offset in seconds (positive = audio1 clap is later).
    """
    from scipy.signal import correlate, correlation_lags

    half_win = int(window_ms / 1000.0 * sample_rate / 2)

    # Extract windows, zero-pad if near edges
    s1 = max(0, clap1_sample - half_win)
    e1 = min(len(audio1), clap1_sample + half_win)
    s2 = max(0, clap2_sample - half_win)
    e2 = min(len(audio2), clap2_sample + half_win)

    seg1 = audio1[s1:e1].copy()
    seg2 = audio2[s2:e2].copy()

    # Normalize segments
    seg1 -= np.mean(seg1)
    seg2 -= np.mean(seg2)
    seg1 /= (np.std(seg1) + 1e-10)
    seg2 /= (np.std(seg2) + 1e-10)

    corr = correlate(seg1, seg2, mode='full', method='fft')
    lags = correlation_lags(len(seg1), len(seg2), mode='full')

    best_lag = lags[np.argmax(corr)]

    # The total offset in samples:
    #   coarse offset  = clap1_sample - clap2_sample
    #   xcorr tweak    = best_lag  (seg1 leads seg2 by best_lag samples)
    #   but seg1 was centred on clap1, seg2 on clap2, so:
    refined_offset_samples = (clap1_sample - clap2_sample) + best_lag
    refined_offset_seconds = refined_offset_samples / sample_rate

    return refined_offset_seconds


def sync_and_crop_videos(path1: str, path2: str,
                         analyze_seconds: float = 10.0) -> None:
    sample_rate = 96000  # high rate for sub-ms precision

    # --- Derive output paths (same dir/name + '_synced.avi') ---
    out1 = synced_output_path(path1)
    out2 = synced_output_path(path2)

    # --- 1. Extract audio from the first N seconds ---
    print(f"[1/7] Extracting audio (first {analyze_seconds}s) at {sample_rate} Hz …")
    audio1 = extract_audio_ffmpeg(path1, duration=analyze_seconds, sample_rate=sample_rate)
    audio2 = extract_audio_ffmpeg(path2, duration=analyze_seconds, sample_rate=sample_rate)
    print(f"  Audio 1: {len(audio1)} samples  |  Audio 2: {len(audio2)} samples")

    # --- 2. Detect the clap in each track (coarse + sample-level) ---
    print("[2/7] Detecting clap transient …")
    clap1 = detect_clap(audio1, sample_rate=sample_rate)
    clap2 = detect_clap(audio2, sample_rate=sample_rate)
    print(f"  Clap in Video 1 at {clap1:.6f}s (sample {int(clap1 * sample_rate)})")
    print(f"  Clap in Video 2 at {clap2:.6f}s (sample {int(clap2 * sample_rate)})")

    coarse_offset = clap1 - clap2
    print(f"  Coarse offset: {coarse_offset * 1000:.2f} ms")

    # --- 3. Refine with cross-correlation around the clap ---
    print("[3/7] Refining with cross-correlation …")
    clap1_sample = int(clap1 * sample_rate)
    clap2_sample = int(clap2 * sample_rate)
    refined_offset = refine_offset_xcorr(
        audio1, audio2, clap1_sample, clap2_sample, sample_rate
    )
    print(f"  Refined offset: {refined_offset * 1000:.2f} ms")

    # --- 4. Compute trim offsets so the claps line up ---
    if refined_offset > 0:
        trim1, trim2 = refined_offset, 0.0
        print(f"  → Trimming {refined_offset * 1000:.2f} ms from the start of Video 1")
    else:
        trim1, trim2 = 0.0, -refined_offset
        print(f"  → Trimming {-refined_offset * 1000:.2f} ms from the start of Video 2")

    # --- 5. Compute common frame count ---
    fps1 = get_fps_ffprobe(path1)
    fps2 = get_fps_ffprobe(path2)
    dur1 = get_duration_ffprobe(path1) - trim1
    dur2 = get_duration_ffprobe(path2) - trim2
    common_dur = min(dur1, dur2)

    # Use the lower FPS to compute frame count so neither video runs out
    frames1 = int(dur1 * fps1)
    frames2 = int(dur2 * fps2)
    common_frames = min(frames1, frames2)
    print(f"[4/7] FPS: Video 1 = {fps1:.2f}, Video 2 = {fps2:.2f}")
    print(f"  Available frames after trim: {frames1} / {frames2}")
    print(f"  Common frame count: {common_frames}  ({common_dur:.2f}s)")

    # --- 6. Trim (re-encode for frame-accurate cuts, exact frame count) ---
    print("[5/7] Writing synced videos (re-encoding, frame-accurate) …")
    trim_video_ffmpeg(path1, out1, start=trim1, n_frames=common_frames)
    trim_video_ffmpeg(path2, out2, start=trim2, n_frames=common_frames)
    print(f"  ✓ {out1}")
    print(f"  ✓ {out2}")

    # --- 7. Verify frame counts match ---
    print("[6/7] Verifying frame counts …")
    actual1 = get_frame_count(out1)
    actual2 = get_frame_count(out2)
    print(f"  Video 1: {actual1} frames  |  Video 2: {actual2} frames")
    if actual1 != actual2:
        print(f"  ⚠ Frame mismatch! Truncating the longer video to {min(actual1, actual2)} frames …")
        target = min(actual1, actual2)
        if actual1 > target:
            _truncate_to_n_frames(out1, target)
        else:
            _truncate_to_n_frames(out2, target)
        print(f"  ✓ Both videos now have {target} frames.")
    else:
        print(f"  ✓ Both videos have exactly {actual1} frames.")

    # --- 8. Side-by-side check ---
    side_path = os.path.join(
        os.path.dirname(path1), "sync_check_side_by_side.avi"
    )
    print("[7/7] Generating side-by-side verification …")
    make_side_by_side(out1, out2, side_path)
    print(f"  ✓ {side_path}")

    print("\nDone! Review the side-by-side video to confirm the sync.")


def _truncate_to_n_frames(path: str, n_frames: int) -> None:
    """Re-encode a video file in-place to exactly *n_frames* frames."""
    tmp = path + ".tmp.avi"
    cmd = [
        "ffmpeg", "-hide_banner", "-loglevel", "error", "-y",
        "-i", path,
        "-frames:v", str(n_frames),
        "-c:v", "libx264", "-preset", "fast", "-crf", "18",
        "-c:a", "aac", "-b:a", "192k",
        tmp,
    ]
    subprocess.run(cmd, check=True, stderr=subprocess.PIPE)
    os.replace(tmp, path)


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    video_1_path = r"C:\Users\Aimar\Downloads\sam-mediapipe-tracking\Aimar's calibration\Videos\cam0_2026-02-25 16-28-25.mkv"
    video_2_path = r"C:\Users\Aimar\Downloads\sam-mediapipe-tracking\Aimar's calibration\Videos\cam1_2026-02-25 16-28-25.mkv"

    sync_and_crop_videos(video_1_path, video_2_path)