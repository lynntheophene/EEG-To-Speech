import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiScaleTemporalBlock(nn.Module):
    """
    Captures EEG features at different timescales (Gamma, Alpha, Delta/Theta).
    """
    def __init__(self, in_channels, out_channels):
        super().__init__()
        # Branch 1: Fast features (Gamma/Beta) - Small kernel
        self.branch1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        
        # Branch 2: Medium features (Alpha/Theta) - Medium kernel
        self.branch2 = nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2)
        
        # Branch 3: Slow features (N400/P600 ERPs) - Large kernel
        self.branch3 = nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3)

        self.bn = nn.BatchNorm1d(out_channels * 3)
        self.gelu = nn.GELU()
        
        # Projection to mix scales
        self.proj = nn.Conv1d(out_channels * 3, out_channels, kernel_size=1)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        
        out = torch.cat([b1, b2, b3], dim=1) # Concatenate filters
        out = self.bn(out)
        out = self.gelu(out)
        return self.proj(out)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x):
        # x shape: [Batch, SeqLen, D_Model]
        return x + self.pe[:, :x.size(1)]

class SuperiorEEGEncoder(nn.Module):
    def __init__(self, 
                 num_channels=105, 
                 d_model=768,      # Target dimension (e.g. for GPT-2 or LLaMA)
                 enc_dim=256,      # Internal CNN dimension
                 n_heads=4, 
                 n_layers=4,
                 num_subjects=20): # For subject adaptation
        super().__init__()

        # 1. Spatial Projection (Channel Mixing)
        # Mixes 105 physical channels into 'enc_dim' latent spatial features
        self.spatial_proj = nn.Conv1d(num_channels, enc_dim, kernel_size=1)
        
        # 2. Multi-Scale Temporal Encoder
        self.temporal_block = MultiScaleTemporalBlock(enc_dim, enc_dim)
        
        # 3. Downsampling (Stability & Efficiency)
        # Reduces 500Hz signal to manageable token sequence length
        self.downsample = nn.AvgPool1d(kernel_size=4, stride=4) # 500Hz -> 125Hz effective
        
        # 4. Subject Adaptation
        self.subj_embed = nn.Embedding(num_subjects, enc_dim)
        
        # 5. Transformer Encoder (The "Brain")
        encoder_layer = nn.TransformerEncoderLayer(d_model=enc_dim, nhead=n_heads, batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.pos_encoder = PositionalEncoding(enc_dim)
        
        # 6. Final Projection to LLM Space
        self.output_proj = nn.Linear(enc_dim, d_model)
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, mask, subject_ids=None):
        """
        x: [Batch, 105, Time] (EEG Signal)
        mask: [Batch, 105] (Channel Mask from preprocessing)
        subject_ids: [Batch] (Int IDs for subject adaptation)
        """
        
        # --- A. Channel Masking (Critical Feature) ---
        # Explicitly zero out padded channels using your precomputed mask
        # x: [B, 105, T], mask: [B, 105] -> mask needs unsqueeze
        x = x * mask.unsqueeze(-1)
        
        # --- B. Spatial Mixing ---
        # [B, 105, T] -> [B, 256, T]
        x = self.spatial_proj(x)
        x = F.gelu(x)
        
        # --- C. Multi-Scale Temporal Encoding ---
        # Captures N400 (slow) vs Gamma (fast) simultaneously
        x = self.temporal_block(x)
        
        # --- D. Downsampling ---
        # [B, 256, T] -> [B, 256, T/4]
        x = self.downsample(x)
        
        # --- E. Prepare for Transformer ---
        # Permute to [Batch, SeqLen, Dim]
        x = x.permute(0, 2, 1) 
        
        # Add Subject Embeddings (if provided)
        if subject_ids is not None:
            # [B, 1, Dim] - broadcasts across time
            s_emb = self.subj_embed(subject_ids).unsqueeze(1)
            x = x + s_emb
            
        # Add Positional Encoding
        x = self.pos_encoder(x)
        
        # --- F. Transformer Layers ---
        x = self.transformer(x)
        
        # --- G. Final Projection ---
        x = self.output_proj(x) # [B, Seq, 768]
        x = self.layer_norm(x)
        
        return x

# --- Quick Test ---
if __name__ == "__main__":
    # Simulate a batch
    batch_size = 2
    time_steps = 200 # 400ms at 500Hz
    
    model = SuperiorEEGEncoder(num_channels=105, d_model=768)
    
    # Fake Data (mimicking your output)
    dummy_eeg = torch.randn(batch_size, 105, time_steps)
    
    # Fake Mask (Subject 1 has all channels, Subject 2 has 4 padded)
    dummy_mask = torch.ones(batch_size, 105)
    dummy_mask[1, 101:] = 0 
    
    # Fake Subjects
    dummy_subjs = torch.tensor([0, 1])
    
    output = model(dummy_eeg, dummy_mask, dummy_subjs)
    
    print(f"Input EEG: {dummy_eeg.shape}")       # [2, 105, 200]
    print(f"Output Emb: {output.shape}")         # [2, 50, 768] (Time downsampled by 4)
    print(" Forward pass successful. Encoder is ready.")