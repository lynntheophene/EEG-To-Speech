import os
import torch
import numpy as np
import pickle
import re 
from torch.utils.data import Dataset

class ZuCoDataset(Dataset):
    def __init__(self, data_dir, tokenizer_name="bert-base-uncased", max_len=32):
        self.data_dir = data_dir
        
        with open(os.path.join(data_dir, "metadata.pkl"), "rb") as f:
            self.metadata = pickle.load(f)
            
        # LOW RAM MODE: Use memmap 'r' (Read from disk)
        self.eeg_data = np.memmap(
            os.path.join(data_dir, "eeg_data.npy"), 
            dtype='float32', mode='r', 
            shape=(len(self.metadata), 840, 105)
        )

        # Subject ID Extraction
        self.subject_ids = []
        for meta in self.metadata:
            # Try to find ID in metadata, else default to 0
            # Ideally update your convert script to store 'subject_id' in metadata dict
            self.subject_ids.append(meta.get('subject_id', 0))

        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.max_len = max_len

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        # Read from disk
        eeg_tensor = torch.tensor(self.eeg_data[idx], dtype=torch.float32)
        
        valid_len = self.metadata[idx]['mask_len']
        mask = torch.zeros(840, dtype=torch.bool)
        mask[:valid_len] = True
        
        text = self.metadata[idx]['text']
        encoding = self.tokenizer(
            text, padding='max_length', truncation=True, 
            max_length=self.max_len, return_tensors='pt'
        )
        
        return {
            'eeg': eeg_tensor,
            'channel_mask': mask,
            'input_ids': encoding['input_ids'].squeeze(0),
            'attention_mask': encoding['attention_mask'].squeeze(0),
            'subject_id': self.subject_ids[idx]
        }

def fixed_collate_fn(batch):
    return {
        'eeg': torch.stack([x['eeg'] for x in batch]),
        'channel_mask': torch.stack([x['channel_mask'] for x in batch]),
        'input_ids': torch.stack([x['input_ids'] for x in batch]),
        'attention_mask': torch.stack([x['attention_mask'] for x in batch]),
        'subject_ids': torch.tensor([x['subject_id'] for x in batch], dtype=torch.long)
    }