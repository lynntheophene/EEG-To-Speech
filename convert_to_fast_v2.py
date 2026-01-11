import os
import numpy as np
import torch
import pickle
from tqdm import tqdm
import torch.nn.functional as F

# --- CONFIG ---
SOURCE_PATH = "./processed_zuco2_diamond"
OUTPUT_DIR = "./optimized_zuco"
FIXED_EEG_LEN = 840
FIXED_CHANNELS = 105

os.makedirs(OUTPUT_DIR, exist_ok=True)

def convert():
    files = [f for f in os.listdir(SOURCE_PATH) if f.endswith('.npz')]
    print(f"🕵️ Found {len(files)} files. Scanning...")

    # Pass 1: Count total samples
    total_samples = 0
    for f in tqdm(files, desc="Scanning"):
        try:
            with np.load(os.path.join(SOURCE_PATH, f), allow_pickle=True) as data:
                key = 'text' if 'text' in data else 'sentences'
                total_samples += len(data[key])
        except: pass
            
    print(f"📦 Total Samples: {total_samples}")
    
    # Open Memmap
    eeg_memmap_path = os.path.join(OUTPUT_DIR, "eeg_data.npy")
    fp_eeg = np.memmap(eeg_memmap_path, dtype='float32', mode='w+', shape=(total_samples, FIXED_EEG_LEN, FIXED_CHANNELS))
    
    metadata = []
    current_idx = 0
    
    for f in tqdm(files, desc="Converting"):
        path = os.path.join(SOURCE_PATH, f)
        try:
            with np.load(path, allow_pickle=True) as data:
                txt_key = 'text' if 'text' in data else 'sentences'
                eeg_key = 'eeg' if 'eeg' in data else 'input_embeddings'
                
                texts = data[txt_key]
                eegs = data[eeg_key]
                
                for i in range(len(texts)):
                    raw_eeg = torch.tensor(eegs[i], dtype=torch.float32)
                    
                    # --- CRITICAL FIX: DETECT AND TRANSPOSE ---
                    # If shape is [105, Time], flip it to [Time, 105]
                    if raw_eeg.shape[0] == 105 and raw_eeg.shape[1] != 105:
                        raw_eeg = raw_eeg.transpose(0, 1)
                    
                    # Also handle the [840 features] case if it exists
                    # If shape is [840, Time] -> This implies concatenated bands? 
                    # For now, we assume standard ZuCo [105 channels]
                    
                    t, c = raw_eeg.shape
                    
                    # 1. Fix Time (Pad/Crop dim 0)
                    if t > FIXED_EEG_LEN: 
                        raw_eeg = raw_eeg[:FIXED_EEG_LEN, :]
                    elif t < FIXED_EEG_LEN:
                        # Pad Bottom
                        raw_eeg = F.pad(raw_eeg, (0,0, 0, FIXED_EEG_LEN - t))
                    
                    # 2. Fix Channels (Pad/Crop dim 1)
                    curr_c = raw_eeg.shape[1]
                    if curr_c > FIXED_CHANNELS: 
                        raw_eeg = raw_eeg[:, :FIXED_CHANNELS]
                    elif curr_c < FIXED_CHANNELS:
                        # Pad Right
                        raw_eeg = F.pad(raw_eeg, (0, FIXED_CHANNELS - curr_c, 0,0))
                    
                    # Write
                    fp_eeg[current_idx] = raw_eeg.numpy()
                    
                    metadata.append({
                        'text': str(texts[i]),
                        'mask_len': min(t, FIXED_EEG_LEN) # This should now be ~200-500, not 105
                    })
                    current_idx += 1
                    
        except Exception as e:
            print(f"Skipping {f}: {e}")

    fp_eeg.flush()
    with open(os.path.join(OUTPUT_DIR, "metadata.pkl"), "wb") as f:
        pickle.dump(metadata, f)
        
    print("✅ Conversion Complete! Data orientation fixed.")

if __name__ == "__main__":
    convert()