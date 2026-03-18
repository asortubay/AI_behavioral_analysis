# Instructions for Recording Synchronized Webcam Videos with OBS

This guide outlines the procedure for setting up OBS Studio to record two webcams simultaneously with audio on both tracks, ensuring they can be synchronized later using the `sync_webcam_videos.py` script.

## Pre-requisites

1.  **OBS Studio** installed (latest version recommended).
2.  **Source Record Plugin**: This plugin allows you to record specific sources (cameras) to separate files effectively.
    *   Download here: [OBS Source Record Plugin](https://obsproject.com/forum/resources/source-record.1285/)

## Step 1: OBS Configuration

### 1. General Settings
*   Open OBS Settings.
*   **Video:** Set your Base (Canvas) and Output (Scaled) Resolution to match your highest camera quality (usually 1920x1080).
*   **FPS:** Set "Common FPS Values" to **30** (or 60, just ensure it is consistent).
*   **Output > Recording:**
    *   **Recording Format:** `mkv` (Safest if OBS crashes; can be remuxed to mp4 later, though the sync script handles mkv fine).
    *   **Encoder:** Hardware encoder (NVENC or QSV) is recommended to reduce CPU load.

### 2. Audio Setup (CRITICAL)
For the synchronization script to work, **BOTH** video files must contain clear audio of the clap.

*   **Microphone:** Use a single high-quality microphone (or one of the webcam mics) as the input.
*   **Audio Monitoring:**
    *   Go to the **Audio Mixer** panel.
    *   Click the gear icon (Settings) -> **Advanced Audio Properties**.
    *   Ensure your Microphone is set to **Monitor Off** (to avoid echo in your ears) but verify it is active on the tracks you are recording.

### 3. Setting Up "Source Record" Filter
This allows you to record Camera 1 and Camera 2 as separate files simultaneously.

1.  **Add Camera 1:**
    *   Add your first webcam as a "Video Capture Device". Name it `cam0`.
    *   Right-click `cam0` in the Sources list -> **Filters**.
    *   Click `+` -> **Source Record**.
    *   **Record Mode:** `Always` (or `Recording`, to start when you hit the main Record button).
    *   **Path:** Choose a folder and a filename pattern (e.g., `cam0_`).
    *   **Audio:** Select your primary microphone source so this video gets audio.

2.  **Add Camera 2:**
    *   Add your second webcam as a "Video Capture Device". Name it `cam1`.
    *   Right-click `cam1` -> **Filters**.
    *   Click `+` -> **Source Record**.
    *   **Record Mode:** Same as above.
    *   **Path:** Choose the same folder but distinct name (e.g., `cam1_`).
    *   **Audio:** **IMPORTANT:** Select the **SAME** primary microphone source. Both videos *must* hear the same room audio (the clap).

## Step 2: Crucial OBS Settings for Frame-to-Frame Sync (Safety Checks)

To ensure that the videos stay synced from the beginning of the recording to the end (preventing audio/video "drift"), you must configure OBS correctly before hitting record:

### 1. Lock Frame Rates
Both webcams must be set to the exact same frame rate in their source properties. If one is running at 30 FPS and the other at 29.97 FPS, they will slowly drift out of sync over time, making frame-to-frame synchronization impossible.

### 2. Disable Device Timestamps
In OBS, double-click each Video Capture Device (webcam) to open its properties. Scroll down and uncheck **Use Device Timestamps**. When this is checked, OBS tries to rely on the webcam's internal clock, which can often be slightly inaccurate or drop frames. Unchecking it forces OBS to manage the timing, which is much more stable for multi-camera setups.

### 3. CBR Encoding
When setting up your Source Record filters, ensure you are using a **Constant Bitrate (CBR)** rather than Variable Bitrate (VBR). This keeps the data processing consistent and helps prevent micro-stutters that can throw off sync.

## Step 3: Advanced "Drift Prevention" Protocol

To get the tightest possible synchronization and prevent "drift" (where audio and video slowly separate over time) or "jitter" (variable frame rates), you need to force your hardware to march to the same beat as OBS.

### Phase 1: Audio Sample Rates
The #1 cause of gradual desync over long recordings is a mismatch in sample rates (e.g., your mic is 44.1 kHz and OBS is 48 kHz).

1.  **Check OBS Settings:**
    *   Go to **Settings > Audio**.
    *   Note the **Sample Rate** (usually 48 kHz is recommended for video).

2.  **Match Windows Settings:**
    *   Right-click the **Speaker icon** in your Windows taskbar -> **Sounds** (or Sound Settings > More Sound Settings).
    *   Go to the **Recording** tab.
    *   Double-click your Microphone/Audio Interface -> **Advanced** tab.
    *   **Crucial:** Ensure the "Default Format" matches your OBS setting exactly (e.g., 2 channel, 16 bit, 48000 Hz).
    *   Repeat this for every audio device you are using.

### Phase 2: Webcam Frame Rate
Webcams are "smart" devices that will sacrifice frame rate to get a brighter image. This creates Variable Frame Rate (VFR) footage, which is a nightmare for sync. You must make them "dumb" and consistent.

1.  **Disable "Low Light Compensation":**
    *   In OBS, right-click your Webcam Source > **Properties**.
    *   Click **Configure Video** (this opens the manufacturer's driver window).
    *   Find the **Camera Control** tab.
    *   **Uncheck "Low Light Compensation"** (sometimes called "RightLight" or "Auto Exposure").
    *   *Why:* If this is on, your cam will drop to 15fps or 20fps in dark rooms, instantly breaking sync with your 30fps project.


2.  **Disable Auto-Focus:**
    *   In the same menu, turn off **Auto-Focus**. Focus hunting changes the focal length slightly ("breathing"), which can mess with frame alignment algorithms. Set it manually once and lock it.

### Phase 3: OBS Internal Clock Sync
You want OBS to be the master clock, not the webcams.

1.  **Disable Device Timestamps:**
    *   In the Webcam Source **Properties** window (in OBS).
    *   **Uncheck "Use Device Timestamps"**.
    *   *Why:* This forces OBS to ignore the webcam's internal clock (which might be cheap and drift) and use the system clock instead.

2.  **Force Custom Resolution/FPS:**
    *   In the same Properties window:
    *   **Resolution/FPS Type:** Set to **Custom**.
    *   **Resolution:** Set to your desired output (e.g., 1920x1080).
    *   **FPS:** Set to **Highest FPS** or match your project FPS (e.g., 30 or 60). Do not leave it on "Match Output".

### Phase 4: The Source Record Filter Settings
Since you are using the Source Record plugin, you have a specific setting to check.

1.  Right-click your Source > **Filters** > **Source Record**.
2.  **Record Mode:** Ensure this is set to **Recording** (so it starts/stops exactly when you hit the main Record button).
3.  **Encoders:** If you have a powerful GPU (NVIDIA), use **NVENC** for both your main recording and your Source Record filters. Using CPU (x264) for one and GPU for another can sometimes introduce tiny processing start-up delays.

### Phase 5: The "Sync Offset" Check (Final Manual Tweak)
Even with perfect settings, some cameras (like HDMI capture cards vs. USB webcams) have different internal processing speeds (latency).

1.  **Perform the "Clap Test" inside OBS:**
    *   Look at your **Audio Mixer** bars.
    *   Clap your hands on camera.
    *   Watch the audio bar jump. Does the visual of your hands touching happen exactly when the green bar spikes?

2.  **Adjust if needed:**
    *   If the Audio is early (bar spikes before hands touch), you need to delay the audio.
    *   Click the **Gear Icon** next to the Audio Mixer > **Advanced Audio Properties**.
    *   Add a **Sync Offset** (start with 50ms, 100ms, etc.) until the clap feels instant.

## Step 4: The Recording Protocol

1.  **Prepare the Subject:**
    *   Ensure the subject is visible in both cameras.
    *   Ensure the microphone is on and levels are bouncing in OBS.

2.  **Start Recording:**
    *   Press **Start Recording** in OBS. (If using Source Record set to "Recording", this triggers both independent files).

3.  **The Synchronization Clap:**
    *   Wait about **2-3 seconds** after starting.
    *   Perform a **single, loud, sharp CLAP** with your hands visible in the frame (if possible, though audio is what matters most).
    *   *Tip:* Do not speak immediately before or after the clap. Give it a second of silence.

4.  **Conduct the Session:**
    *   Proceed with your interview or experiment.

5.  **Stop Recording:**
    *   Stop OBS. Wait for the files to finalize.

## Step 5: Post-Processing

1.  Navigate to your output folder.
2.  You should see two video files (e.g., `cam0_timestamp.mkv` and `cam1_timestamp.mkv`).
3.  These are the files you will input into `sync_webcam_videos.py`.

---
