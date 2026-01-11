import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class BrainBridge(nn.Module):
    def __init__(self, eeg_encoder, text_dim=768):
        super().__init__()
        self.eeg_encoder = eeg_encoder
        
        # Learnable Temperature
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, eeg, mask, subject_ids=None):
        # 1. Get embedding from Encoder (Already pooled to [Batch, 768])
        z_eeg = self.eeg_encoder(eeg, mask, subject_ids) 
        
        # 2. Normalize 
        z_eeg = z_eeg / z_eeg.norm(dim=1, keepdim=True)
        
        # 3. Clamp Temperature 
        with torch.no_grad():
            self.logit_scale.data.clamp_(0, 4.6052) 

        return {
            'z_text': z_eeg, 
            'logit_scale': self.logit_scale.exp()
        }

class BrainCLIPLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_img = nn.CrossEntropyLoss()
        self.loss_txt = nn.CrossEntropyLoss()

    def forward(self, z_eeg, z_text, logit_scale):
        # Normalize text too
        z_text = z_text / z_text.norm(dim=1, keepdim=True)
        
        logits = (z_eeg @ z_text.t()) * logit_scale
        labels = torch.arange(logits.size(0), device=logits.device)
        
        loss_e = self.loss_img(logits, labels)
        loss_t = self.loss_txt(logits.t(), labels)
        
        return (loss_e + loss_t) / 2