import os
import numpy as np
import torch
import pickle
from tqdm import tqdm
import torch.nn.functional as F

# --- CONFIG ---
SOURCE_PATH = "./processed_zuco2_diamond"  # Your current data
OUTPUT_DIR = "./optimized_zuco"            # Where to save the fast version
FIXED_EEG_LEN = 840
FIXED_CHANNELS = 105

os.makedirs(OUTPUT_DIR, exist_ok=True)

def convert():
    files = [f for f in os.listdir(SOURCE_PATH) if f.endswith('.npz')]
    print(f"Found {len(files)} files. Scanning for total count...")

    # Pass 1: Count total samples to pre-allocate disk space
    total_samples = 0
    for f in tqdm(files, desc="Scanning"):
        try:
            with np.load(os.path.join(SOURCE_PATH, f), allow_pickle=True) as data:
                # Handle inconsistent keys
                key = 'text' if 'text' in data else 'sentences'
                total_samples += len(data[key])
        except:
            pass
            
    print(f" Total Samples: {total_samples}")
    print(f" Pre-allocating {total_samples}x{FIXED_EEG_LEN}x{FIXED_CHANNELS} float32 array (~{total_samples * FIXED_EEG_LEN * FIXED_CHANNELS * 4 / 1e9:.2f} GB)...")

    # Create Memory-Mapped File on Disk (Zero RAM usage)
    eeg_memmap_path = os.path.join(OUTPUT_DIR, "eeg_data.npy")
    fp_eeg = np.memmap(eeg_memmap_path, dtype='float32', mode='w+', shape=(total_samples, FIXED_EEG_LEN, FIXED_CHANNELS))
    
    # Store Metadata (Text, etc.) in a simple list
    metadata = []
    
    # Pass 2: Write Data
    current_idx = 0
    for f in tqdm(files, desc="Converting"):
        path = os.path.join(SOURCE_PATH, f)
        try:
            with np.load(path, allow_pickle=True) as data:
                # Extract Data
                txt_key = 'text' if 'text' in data else 'sentences'
                eeg_key = 'eeg' if 'eeg' in data else 'input_embeddings'
                
                texts = data[txt_key]
                eegs = data[eeg_key]
                
                for i in range(len(texts)):
                    # 1. Process EEG
                    raw_eeg = torch.tensor(eegs[i], dtype=torch.float32)
                    
                    # Pad/Crop to Fixed Size immediately
                    t, c = raw_eeg.shape
                    
                    # Fix Time
                    if t > FIXED_EEG_LEN: raw_eeg = raw_eeg[:FIXED_EEG_LEN, :]
                    else: raw_eeg = F.pad(raw_eeg, (0,0, 0, FIXED_EEG_LEN-t))
                    
                    # Fix Channels
                    curr_c = raw_eeg.shape[1]
                    if curr_c > FIXED_CHANNELS: raw_eeg = raw_eeg[:, :FIXED_CHANNELS]
                    else: raw_eeg = F.pad(raw_eeg, (0, FIXED_CHANNELS-curr_c, 0,0))
                    
                    # Write to Disk (Memmap)
                    fp_eeg[current_idx] = raw_eeg.numpy()
                    
                    # 2. Process Text
                    metadata.append({
                        'text': str(texts[i]),
                        'mask_len': min(t, FIXED_EEG_LEN) # Store real length for masking
                    })
                    
                    current_idx += 1
                    
        except Exception as e:
            print(f"Error skipping {f}: {e}")

    # Flush changes to disk
    fp_eeg.flush()
    
    # Save Metadata
    with open(os.path.join(OUTPUT_DIR, "metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)
        
    print(" :) Conversion Complete! Use './optimized_zuco' for training.")

if __name__ == "__main__":
    convert()