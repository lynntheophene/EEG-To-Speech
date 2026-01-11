import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class SEBlock(nn.Module):
    """Squeeze-and-Excitation Block to reweight noisy channels"""
    def __init__(self, channels, reduction=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, t = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1)
        return x * y.expand_as(x)

class MultiScaleTemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.branch1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1)
        self.branch2 = nn.Conv1d(in_channels, out_channels, kernel_size=5, padding=2)
        self.branch3 = nn.Conv1d(in_channels, out_channels, kernel_size=7, padding=3)
        self.bn = nn.BatchNorm1d(out_channels * 3)
        self.gelu = nn.GELU()
        self.proj = nn.Conv1d(out_channels * 3, out_channels, kernel_size=1)

    def forward(self, x):
        b1 = self.branch1(x)
        b2 = self.branch2(x)
        b3 = self.branch3(x)
        out = torch.cat([b1, b2, b3], dim=1)
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
        return x + self.pe[:, :x.size(1)]

class SuperiorEEGEncoder(nn.Module):
    def __init__(self, num_channels=105, d_model=768, enc_dim=256, n_heads=4, n_layers=4):
        super().__init__()

        # 1. Instance Norm & Dropout
        self.inst_norm = nn.InstanceNorm1d(num_channels)
        self.input_dropout = nn.Dropout(p=0.1)

        # 2. Spatial Projection & Attention
        self.spatial_proj = nn.Conv1d(num_channels, enc_dim, kernel_size=1)
        self.se_block = SEBlock(channels=enc_dim) 
        
        # 3. Temporal Encoder
        self.temporal_block = MultiScaleTemporalBlock(enc_dim, enc_dim)
        
        # 4. Downsampling (840 -> 210)
        self.downsample = nn.AvgPool1d(kernel_size=4, stride=4) 
        
        # 5. Transformer
        encoder_layer = nn.TransformerEncoderLayer(d_model=enc_dim, nhead=n_heads, 
                                                   dim_feedforward=enc_dim*4, 
                                                   dropout=0.25, 
                                                   batch_first=True)
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=n_layers)
        self.pos_encoder = PositionalEncoding(enc_dim)
        self.pos_dropout = nn.Dropout(p=0.25)
        
        # 6. Subject Embedding (Learns user bias)
        self.subject_embed = nn.Embedding(20, enc_dim) 

        # 7. Non-Linear Projection Head (MLP)
        self.output_proj = nn.Sequential(
            nn.Linear(enc_dim, enc_dim),
            nn.GELU(),
            nn.Linear(enc_dim, d_model)
        )
        self.layer_norm = nn.LayerNorm(d_model)

    def forward(self, x, mask=None, subject_ids=None):
        # x: [Batch, 840, 105]
        
        # --- Channel Dropout (Training Only) ---
        if self.training:
            c_mask = torch.bernoulli(torch.full((x.size(0), 1, x.size(2)), 0.85, device=x.device))
            x = x * c_mask / 0.85
        
        x = x.permute(0, 2, 1) # [B, 105, 840]
        x = self.inst_norm(x)
        
        # Spatial Processing
        x = self.spatial_proj(x)
        x = self.se_block(x) 
        x = F.gelu(x)
        x = self.input_dropout(x)
        x = self.temporal_block(x)
        
        # Downsample
        x = self.downsample(x) # [B, 256, 210]
        
        if mask is not None:
            mask_float = mask.float().unsqueeze(1)
            mask_down = F.avg_pool1d(mask_float, kernel_size=4, stride=4).squeeze(1)
            mask_down = mask_down > 0 
        else:
            mask_down = torch.ones(x.size(0), x.size(2), device=x.device, dtype=torch.bool)

        x = x.permute(0, 2, 1) # [B, 210, 256]

        # Add Subject Bias 
        if subject_ids is not None:
            # Clamp IDs just in case
            subject_ids = subject_ids.clamp(0, 19)
            sub_emb = self.subject_embed(subject_ids).unsqueeze(1) # [B, 1, 256]
            x = x + sub_emb

        x = self.pos_encoder(x)
        x = self.pos_dropout(x)
        
        # Transformer
        padding_mask = ~mask_down 
        x = self.transformer(x, src_key_padding_mask=padding_mask)
        
        # Projection
        x = self.output_proj(x) 
        x = self.layer_norm(x)
        
        # --- SMART POOLING (Critical Fix) ---
        x = x * mask_down.unsqueeze(-1)
        sum_embeddings = x.sum(dim=1) 
        valid_counts = mask_down.sum(dim=1, keepdim=True).clamp(min=1.0)
        mean_embedding = sum_embeddings / valid_counts
        
        return mean_embedding