"""
ZuCo 2.0 (OSF) Preprocessing Pipeline - Done by lynntheophene
----------------------------------------------------------------------------
Task: Normal Reading (NR)
Sampling Rate: 500 Hz
Reference: Average Reference (Applied in original ZuCo 2.0 dataset release)
Unit: Microvolts (µV)

Scientific Features:
1. Two-Stage Artifact Rejection (Coarse >150µV, Fine >100µV).
   - EOG artifacts are partially mitigated via fixation-based segmentation 
     and this two-stage amplitude rejection.
2. Gaussian Noise Padding:
   - Fills missing channels with uncorrelated noise scaled to subject variance.
   - Prevents zero-cliffs without leaking spatial information.
3. Explicit Notch Filtering (50Hz) & Bandpass (0.5-40Hz).
   - 0.5 Hz high-pass chosen to preserve slow semantic ERPs (N400, P600).
4. Fixation Concatenation for temporal continuity.
5. Forensic Logging (95th/99th percentile amplitudes).
"""

import os
import re
import numpy as np
import h5py
import glob
import json
from tqdm import tqdm
from scipy.signal import butter, filtfilt, iirnotch

# --- Configuration ---
ZUCO_ROOT = "/media/mando/0E5F-F655/datasets/zuco-2/task1 - NR/Matlab files"
OUT_DIR = "./processed_zuco2_diamond"
TASK = "NR"

# Physics & filtering
TARGET_CHANNELS = 105
SAMPLING_RATE = 500
NOTCH_FREQ = 50.0   # 50Hz (EU/India Grid)
NOTCH_QUALITY = 30 

# Time Thresholds
MIN_FIXATION_DURATION = 30   # 60ms
MAX_LEN_SAMPLES = 1000       # 2s max
BASELINE_WINDOW = 25         # 50ms

# Artifact Handling
COARSE_THRESHOLD_UV = 150.0  
FINE_THRESHOLD_UV = 100.0    
BAD_CHANNEL_PERCENT = 0.20   

# Reproducibility
np.random.seed(42)

os.makedirs(OUT_DIR, exist_ok=True)

# Global Stats
global_stats = {
    "total_samples": 0,
    "stage1_rejections": 0,
    "stage2_rejections": 0,
    "padded_subjects": [],
    "unit_conversion_triggered": False,
    "amplitude_percentiles": {}
}

# --- Signal Processing Functions ---

def notch_filter(data, fs, freq=50.0, q=30):
    b, a = iirnotch(freq / (fs / 2), q)
    return filtfilt(b, a, data, axis=-1)

def butter_bandpass_filter(data, lowcut, highcut, fs, order=4):
    """
    Standard zero-phase bandpass.
    0.5 Hz chosen to preserve slow semantic ERPs (N400, P600).
    """
    nyq = 0.5 * fs
    b, a = butter(order, [lowcut/nyq, highcut/nyq], btype='bandpass')
    return filtfilt(b, a, data, axis=-1)

def pad_or_crop_channels_noise(signal, target_ch):
    """
    Improved Padding: Uses scaled Gaussian noise.
    Returns: (processed_signal, valid_channel_count, channel_mask, noise_scale_used)
    """
    current_ch, time = signal.shape
    mask = np.ones(target_ch, dtype=np.float32)
    
    if current_ch == target_ch:
        return signal, target_ch, mask, 0.0
    
    if current_ch < target_ch:
        pad_needed = target_ch - current_ch
        
        # Calculate noise floor of existing channels (std dev)
        # We use a fraction (10%) of the real signal's variance to simulate "quiet" electrodes
        noise_scale = np.mean(np.std(signal, axis=1)) * 0.1
        
        # Generate Gaussian noise (Uncorrelated)
        padding = np.random.normal(loc=0.0, scale=noise_scale, size=(pad_needed, time))
        padded_signal = np.concatenate([signal, padding], axis=0)
        
        # Mark padded area in mask
        mask[current_ch:] = 0.0
        return padded_signal, current_ch, mask, noise_scale
        
    if current_ch > target_ch:
        return signal[:target_ch, :], target_ch, mask, 0.0

def load_matlab_string(ref, f):
    try:
        if not isinstance(ref, h5py.Reference): return "UNK"
        obj = f[ref]
        data = obj[()]
        if isinstance(data, np.ndarray):
            return "".join([chr(int(c)) for c in data.flatten() if c > 0]).strip()
        return str(data)
    except: return "UNK"


# --- Main Pipeline ---

files = glob.glob(os.path.join(ZUCO_ROOT, "**", f"*{TASK}*.mat"), recursive=True)

for file_path in files:
    subject = re.search(r"(Y[A-Z]{2})", file_path).group(1) if re.search(r"(Y[A-Z]{2})", file_path) else "UNK"
    print(f"\n>>> Subject: {subject}")

    eeg_samples, text_samples, meta, masks = [], [], [], []
    rej_s1, rej_s2 = 0, 0
    channels_detected = False
    
    # Forensic amplitude tracking
    all_amplitudes = []

    # Define Global Meta for this file
    meta_global = {
        "reference": "average (dataset-provided)",
        "sampling_rate": SAMPLING_RATE,
        "filters": {"notch": NOTCH_FREQ, "bandpass": [0.5, 40.0]},
        "artifact_thresholds": {"pre_filter": COARSE_THRESHOLD_UV, "post_filter": FINE_THRESHOLD_UV}
    }

    try:
        with h5py.File(file_path, "r") as f:
            if 'sentenceData' not in f or 'word' not in f['sentenceData']:
                print(f"Skipping {subject}: Structure mismatch.")
                continue
            
            word_dataset = f['sentenceData']['word']
            
            for s_idx in tqdm(range(len(word_dataset)), desc=f"Processing {subject}"):
                try:
                    sent_ref = word_dataset[s_idx][0]
                    if not isinstance(sent_ref, h5py.Reference): continue
                    sent_obj = f[sent_ref]

                    if 'content' not in sent_obj or 'rawEEG' not in sent_obj: continue
                    content_ds, eeg_ds = sent_obj['content'], sent_obj['rawEEG']
                    
                    if not hasattr(content_ds, 'shape') or not hasattr(eeg_ds, 'shape'): continue
                    n_words = min(content_ds.shape[0], eeg_ds.shape[0])
                    
                    for w_idx in range(n_words):
                        try:
                            # A. Text
                            word_text = load_matlab_string(content_ds[w_idx][0], f)

                            # B. EEG Extraction
                            eeg_ref = eeg_ds[w_idx][0]
                            if not isinstance(eeg_ref, h5py.Reference): continue
                            
                            signal_raw = f[eeg_ref][()]
                            signal = None
                            fix_lengths = []

                            # --- Fixation Stitching ---
                            if signal_raw.dtype == 'object':
                                fix_signals = []
                                for i in range(signal_raw.shape[0]):
                                    try:
                                        ref = signal_raw[i][0]
                                        fix = np.array(f[ref][()], dtype=np.float32)
                                        if 100 <= fix.shape[0] <= 130 and fix.shape[1] > fix.shape[0]: pass
                                        elif 100 <= fix.shape[1] <= 130 and fix.shape[0] > fix.shape[1]: fix = fix.T
                                        else: continue
                                        fix_signals.append(fix)
                                    except: continue
                                
                                if fix_signals:
                                    fix_lengths = [x.shape[1] for x in fix_signals]
                                    signal = np.concatenate(fix_signals, axis=1)
                                else: continue
                            else:
                                signal = np.array(signal_raw, dtype=np.float32)
                                if 100 <= signal.shape[0] <= 130 and signal.shape[1] > signal.shape[0]: pass
                                elif 100 <= signal.shape[1] <= 130 and signal.shape[0] > signal.shape[1]: signal = signal.T
                                else: continue
                                fix_lengths = [signal.shape[1]]

                            if signal is None: continue

                            # --- C. Physics Correction ---
                            if np.max(np.abs(signal)) < 0.01:
                                signal *= 1e6
                                global_stats["unit_conversion_triggered"] = True

                            # --- D. Baseline Correction ---
                            if signal.shape[1] >= BASELINE_WINDOW:
                                baseline = np.mean(signal[:, :BASELINE_WINDOW], axis=1, keepdims=True)
                                signal -= baseline
                            else:
                                signal -= np.mean(signal, axis=1, keepdims=True)

                            # --- E. STAGE 1: Coarse Artifact Rejection ---
                            if np.max(np.abs(signal)) > COARSE_THRESHOLD_UV:
                                rej_s1 += 1
                                continue

                            # --- F. Filtering Chain ---
                            # 50Hz Notch for Line Noise
                            signal = notch_filter(signal, SAMPLING_RATE, freq=NOTCH_FREQ, q=NOTCH_QUALITY)
                            # 0.5-40Hz Bandpass (Preserves N400/P600)
                            signal = butter_bandpass_filter(signal, 0.5, 40.0, SAMPLING_RATE)

                            # --- G. STAGE 2: Fine Artifact Rejection ---
                            bad_channels = np.sum(np.max(np.abs(signal), axis=1) > FINE_THRESHOLD_UV)
                            if (bad_channels / signal.shape[0]) > BAD_CHANNEL_PERCENT:
                                rej_s2 += 1
                                continue

                            # Log valid amplitudes
                            if len(all_amplitudes) < 10000: 
                                all_amplitudes.extend(np.abs(signal).flatten()[:100])

                            # --- H. Standardization (Gaussian Noise Padding) ---
                            signal, valid_ch, ch_mask, noise_used = pad_or_crop_channels_noise(signal, TARGET_CHANNELS)
                            if valid_ch < TARGET_CHANNELS and subject not in global_stats["padded_subjects"]:
                                global_stats["padded_subjects"].append(subject)

                            # I. Safety Clipping
                            if signal.shape[1] < MIN_FIXATION_DURATION: continue
                            if signal.shape[1] > MAX_LEN_SAMPLES:
                                signal = signal[:, :MAX_LEN_SAMPLES]

                            # J. Z-Score Normalization
                            mean = signal.mean(axis=1, keepdims=True)
                            std = signal.std(axis=1, keepdims=True)
                            signal = (signal - mean) / (std + 1e-6)

                            if not channels_detected:
                                print(f"[INFO] {subject} valid. Shape: {signal.shape} (Mask Sum: {int(ch_mask.sum())})")
                                channels_detected = True

                            eeg_samples.append(signal)
                            text_samples.append(word_text)
                            masks.append(ch_mask)
                            
                            meta.append({
                                "subject": subject,
                                "s_idx": s_idx, 
                                "w_idx": w_idx,
                                "valid_channels": valid_ch,
                                "fix_boundaries": fix_lengths,
                                "noise_scale": float(noise_used) # Forensic Log
                            })

                        except Exception: continue
                except Exception: continue
    except Exception as e:
        print(f"Error {subject}: {e}")
        continue

    # Update Global Stats
    global_stats["total_samples"] += len(eeg_samples)
    global_stats["stage1_rejections"] += rej_s1
    global_stats["stage2_rejections"] += rej_s2
    
    # Forensic Stats
    if all_amplitudes:
        p95 = float(np.percentile(all_amplitudes, 95))
        p99 = float(np.percentile(all_amplitudes, 99))
        global_stats["amplitude_percentiles"][subject] = {"p95": round(p95, 2), "p99": round(p99, 2)}

    if eeg_samples:
        out_path = os.path.join(OUT_DIR, f"{TASK}_{subject}.npz")
        
        eeg_array = np.empty(len(eeg_samples), dtype=object)
        eeg_array[:] = eeg_samples
        meta_array = np.empty(len(meta), dtype=object)
        meta_array[:] = meta
        
        np.savez(out_path, 
                 eeg=eeg_array, 
                 text=np.array(text_samples), 
                 masks=np.array(masks),
                 meta=meta_array,
                 info=meta_global)
        
        print(f" SAVED {len(eeg_samples)} samples. Rejected (S1:{rej_s1}, S2:{rej_s2}). 95th Percentile: {global_stats['amplitude_percentiles'][subject]['p95']} µV")
    else:
        print(f" FAILED {subject}.")

# Save run report
with open(os.path.join(OUT_DIR, "preprocessing_report.json"), "w") as f:
    json.dump(global_stats, f, indent=4)
print("\n--- Diamond-Standard (10/10) Processing Complete ---")