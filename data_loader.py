import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer
from tqdm import tqdm

class ZuCoDataset(Dataset):
    def __init__(self, data_dir, tokenizer_name="bert-base-uncased", max_seq_len=256):
        self.data_dir = data_dir
        self.max_seq_len = max_seq_len
        
        # 1. Load Tokenizer Once
        print(f"⚙️ Pre-loading Tokenizer ({tokenizer_name})...")
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        
        self.files = [f for f in os.listdir(data_dir) if f.endswith('.npz')]
        self.index = []
        self.precomputed_tokens = [] # Store tokens here to save CPU time
        
        print(f"📦 Indexing & Pre-tokenizing {len(self.files)} files...")
        
        # Buffer to batch tokenize (much faster)
        temp_texts = []
        temp_indices = []
        
        for fname in tqdm(self.files):
            path = os.path.join(data_dir, fname)
            try:
                # Fast header read
                with np.load(path, allow_pickle=True) as data:
                    n_samples = len(data['eeg'])
                    texts = data['text'] # Load all texts in this file
                    
                    for i in range(n_samples):
                        # Store pointer
                        self.index.append((path, i))
                        
                        # Handle text format
                        t_raw = texts[i]
                        t_str = str(t_raw) if not isinstance(t_raw, np.ndarray) else str(t_raw)
                        temp_texts.append(t_str)
                        
            except Exception as e:
                print(f"⚠️ Error {fname}: {e}")

        # 2. Bulk Tokenize (Massive Speedup)
        print("⚡ Running Bulk Tokenization (this might take 10s)...")
        encoded = self.tokenizer(
            temp_texts,
            padding='max_length',
            truncation=True,
            max_length=self.max_seq_len,
            return_tensors='pt'
        )
        
        # Store tensors in CPU RAM (Total size is tiny: ~50MB)
        self.input_ids = encoded['input_ids']
        self.attention_masks = encoded['attention_mask']
        self.raw_texts = temp_texts
        
        print(f"✅ Ready! Indexed {len(self.index)} samples.")

    def __len__(self):
        return len(self.index)

    def __getitem__(self, idx):
        # 1. Lazy Load EEG (Heavy) from Disk
        filepath, local_idx = self.index[idx]
        
        with np.load(filepath, allow_pickle=True) as data:
            eeg = data['eeg'][local_idx]
            mask = data['masks'][local_idx]
            
        eeg_tensor = torch.tensor(eeg, dtype=torch.float32)
        mask_tensor = torch.tensor(mask, dtype=torch.float32)
        
        # 2. Fetch Pre-computed Tokens (Instant)
        # No tokenizer overhead here!
        return {
            "eeg": eeg_tensor,
            "channel_mask": mask_tensor,
            "input_ids": self.input_ids[idx],       # Already a tensor
            "attention_mask": self.attention_masks[idx], # Already a tensor
            "text": self.raw_texts[idx]
        }

def custom_collate_fn(batch):
    # (Same as before)
    max_time = max([item['eeg'].shape[1] for item in batch])
    
    eeg_batch = []
    masks_batch = []
    input_ids_batch = []
    attn_masks_batch = []
    
    for item in batch:
        eeg = item['eeg']
        channels, time = eeg.shape
        pad_amount = max_time - time
        if pad_amount > 0:
            eeg_padded = torch.nn.functional.pad(eeg, (0, pad_amount), "constant", 0.0)
        else:
            eeg_padded = eeg
            
        eeg_batch.append(eeg_padded)
        masks_batch.append(item['channel_mask'])
        input_ids_batch.append(item['input_ids'])
        attn_masks_batch.append(item['attention_mask'])
        
    return {
        "eeg": torch.stack(eeg_batch),
        "channel_mask": torch.stack(masks_batch),
        "input_ids": torch.stack(input_ids_batch),
        "attention_mask": torch.stack(attn_masks_batch)
    }