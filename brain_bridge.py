import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# Import your encoder from the previous step
from eeg_encoder import SuperiorEEGEncoder

class AttentionPooling(nn.Module):
    """
    Step C1: Pools the sequence of EEG tokens into a single 'Sentence Vector'.
    Instead of simple mean pooling, it learns which time steps matter.
    """
    def __init__(self, d_model):
        super().__init__()
        self.attention = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.Tanh(),
            nn.Linear(d_model // 2, 1)
        )

    def forward(self, x):
        # x: [Batch, Seq_Len, Dim]
        
        # Calculate weights for each time step
        # weights: [Batch, Seq_Len, 1]
        weights = self.attention(x)
        weights = F.softmax(weights, dim=1)
        
        # Weighted Sum -> [Batch, Dim]
        # This is z_sem (The "Semantic Latent")
        pooled = torch.sum(x * weights, dim=1)
        return pooled

class ProjectionHead(nn.Module):
    """
    Step C2/C3: Maps the Semantic Latent to Target Space (Text or Audio).
    Uses an MLP as recommended by CLIP/LLaVA papers.
    """
    def __init__(self, embedding_dim, target_dim, dropout=0.1):
        super().__init__()
        self.projection = nn.Sequential(
            nn.Linear(embedding_dim, embedding_dim),
            nn.GELU(),
            nn.Linear(embedding_dim, target_dim),
            nn.Dropout(dropout)
        )
        self.layer_norm = nn.LayerNorm(target_dim)

    def forward(self, x):
        projected = self.projection(x)
        projected = self.layer_norm(projected)
        return projected

class BrainBridge(nn.Module):
    """
    The Full Model: EEG Encoder + Semantic Alignment + Projections
    """
    def __init__(self, 
                 eeg_encoder,           # Your SuperiorEEGEncoder instance
                 text_dim=768,          # Target dimension (e.g., GPT-2/BERT)
                 audio_dim=768,         # Target dimension (e.g., HuBERT)
                 freeze_encoder=False):
        super().__init__()
        
        self.encoder = eeg_encoder
        
        if freeze_encoder:
            for param in self.encoder.parameters():
                param.requires_grad = False
        
        # 1. Semantic Pooling (Sequence -> Vector)
        # We assume encoder output dim is 768 (d_model from previous step)
        self.pooler = AttentionPooling(d_model=768)
        
        # 2. Text Alignment Head (EEG -> Text Space)
        self.text_projector = ProjectionHead(embedding_dim=768, target_dim=text_dim)
        
        # 3. Audio Alignment Head (EEG -> Audio Space) - Future Proofing
        self.audio_projector = ProjectionHead(embedding_dim=768, target_dim=audio_dim)
        
        # Learnable Temperature for CLIP Loss
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, eeg, mask, subject_ids=None):
        """
        Returns dictionary containing all representations needed for Hybrid Loss.
        """
        # 1. Get Sequence Output [B, T, D]
        # Used for: Generative Loss (Next Token Prediction)
        z_seq = self.encoder(eeg, mask, subject_ids)
        
        # 2. Get Pooled Output [B, D]
        # Used for: The "Concept" of the sentence
        z_sem = self.pooler(z_seq)
        
        # 3. Project to Target Spaces
        # Used for: Contrastive Loss (BrainCLIP)
        z_text_aligned = self.text_projector(z_sem)
        z_audio_aligned = self.audio_projector(z_sem)
        
        return {
            "z_seq": z_seq,             # For Decoder (LLM)
            "z_sem": z_sem,             # Raw Brain Concept
            "z_text": z_text_aligned,   # Aligned to Text
            "z_audio": z_audio_aligned, # Aligned to Audio
            "logit_scale": self.logit_scale
        }

class BrainCLIPLoss(nn.Module):
    """
    The Contrastive Alignment Loss (InfoNCE).
    Forces the EEG embedding to be close to its corresponding Text embedding.
    """
    def __init__(self):
        super().__init__()
        self.cross_entropy = nn.CrossEntropyLoss()

    def forward(self, eeg_features, text_features, logit_scale):
        # Normalize features (Cosine Similarity requires normalization)
        eeg_features = F.normalize(eeg_features, dim=1)
        text_features = F.normalize(text_features, dim=1)
        
        # Calculate similarity matrix
        # [Batch, Batch]
        logits = logit_scale.exp() * (eeg_features @ text_features.t())
        
        # Labels are just the diagonal (0, 1, 2...)
        # e.g., EEG_0 should match Text_0
        batch_size = logits.shape[0]
        labels = torch.arange(batch_size, device=logits.device)
        
        # Symmetric Loss (EEG->Text and Text->EEG)
        loss_e = self.cross_entropy(logits, labels)
        loss_t = self.cross_entropy(logits.t(), labels)
        
        return (loss_e + loss_t) / 2

# --- Quick Test ---
if __name__ == "__main__":
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 1. Init Encoder
    encoder = SuperiorEEGEncoder(num_channels=105, d_model=768)
    
    # 2. Init Bridge
    bridge = BrainBridge(encoder).to(device)
    loss_fn = BrainCLIPLoss().to(device)
    
    # 3. Dummy Data
    B, C, T = 2, 105, 200
    dummy_eeg = torch.randn(B, C, T).to(device)
    dummy_mask = torch.ones(B, C).to(device)
    dummy_text_emb = torch.randn(B, 768).to(device) # From frozen BERT/GPT
    
    # 4. Forward
    outputs = bridge(dummy_eeg, dummy_mask)
    
    # 5. Calculate Alignment Loss
    loss = loss_fn(outputs['z_text'], dummy_text_emb, outputs['logit_scale'])
    
    print(f"Sequence Output: {outputs['z_seq'].shape} (For Decoder)")
    print(f"Aligned Text:    {outputs['z_text'].shape} (For CLIP)")
    print(f"CLIP Loss:       {loss.item():.4f}")
    print("✅ BrainBridge Architecture Ready")