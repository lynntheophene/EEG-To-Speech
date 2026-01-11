"""
Training Phase 1 - EEG-Text Contrastive Learning

This script implements Phase 1 of the EEG-to-Speech pipeline: learning a joint embedding
space between EEG signals and natural language using contrastive learning (CLIP-style loss).

Key responsibilities:
  - Load ZuCo dataset with optimized memory-mapped EEG data
  - Train EEG encoder + BrainBridge model to align EEG and BERT text embeddings
  - Validate using retrieval accuracy (EEG->Text matching)
  - Save best checkpoints based on accuracy metric

Requires: EEG encoder, Brain Bridge module, BERT tokenizer
Output: Trained encoder checkpoint for Phase 2
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from transformers import AutoModel, get_cosine_schedule_with_warmup

# --- RDNA 4 / ROCm FLAGS ---
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "12.0.0" 
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from eeg_encoder import SuperiorEEGEncoder
from brain_bridge import BrainBridge, BrainCLIPLoss
from data_loader import ZuCoDataset, fixed_collate_fn

# --- CONFIG ---
BATCH_SIZE = 64         
EPOCHS = 20
ACCUM_STEPS = 4  # Effective Batch Size = 256
LEARNING_RATE = 2e-4    
DATA_PATH = "./optimized_zuco"
SAVE_DIR = "./checkpoints_phase1"
TEXT_MODEL_NAME = "bert-base-uncased" 

os.makedirs(SAVE_DIR, exist_ok=True)

def calculate_accuracy(eeg_embeds, text_embeds, k=1):
    logits = eeg_embeds @ text_embeds.t() # [Batch, Batch]
    targets = torch.arange(logits.shape[0], device=logits.device)
    _, indices = logits.topk(k, dim=1)
    correct = indices.eq(targets.view(-1, 1).expand_as(indices))
    return correct.sum().float() / targets.size(0)

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f" Device: {device}")

    # Load Data
    full_dataset = ZuCoDataset(DATA_PATH, tokenizer_name=TEXT_MODEL_NAME)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    loader_args = dict(
        batch_size=BATCH_SIZE, shuffle=True, collate_fn=fixed_collate_fn,
        num_workers=8, persistent_workers=True, pin_memory=True,
        prefetch_factor=4, drop_last=True 
    )
    train_loader = DataLoader(train_dataset, **loader_args)
    loader_args['shuffle'] = False
    val_loader = DataLoader(val_dataset, **loader_args)
    
    # Models
    eeg_encoder = SuperiorEEGEncoder(num_channels=105, d_model=768)
    model = BrainBridge(eeg_encoder, text_dim=768).to(device)
    ## TO LOAD A ALREADY TRAINED MODEL
    if os.path.exists(checkpoint_path):
        print(f"Found checkpoint at {checkpoint_path}. Loading...")
        try:
            # Load the weights
            state_dict = torch.load(checkpoint_path, map_location=device)
            model.load_state_dict(state_dict)
            print("Checkpoint loaded successfully! Resuming training.")
        except Exception as e:
            print(f"Error loading checkpoint: {e}")
            print("Starting from scratch instead.")
    else:
        print("No checkpoint found. Starting training from scratch.")
    ## 
    text_teacher = AutoModel.from_pretrained(TEXT_MODEL_NAME).to(device)
    for param in text_teacher.parameters():
        param.requires_grad = False
    
    # Optimization
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    # Cosine Scheduler
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(0.1*len(train_loader)*EPOCHS), len(train_loader)*EPOCHS)
    loss_fn = BrainCLIPLoss().to(device)

    # --- FIX: Initialize to 0.0 because we want to maximize Accuracy ---
    best_accuracy = 0.0 
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", smoothing=0.1)
        
        for i, batch in enumerate(loop):
            eeg = batch['eeg'].to(device, non_blocking=True)
            mask = batch['channel_mask'].to(device, non_blocking=True)
            sub_ids = batch['subject_ids'].to(device, non_blocking=True)
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attn_mask = batch['attention_mask'].to(device, non_blocking=True)
            
            # Augmentation (Noise)
            if model.training:
                eeg = eeg + (torch.randn_like(eeg) * 0.01) 

            with torch.no_grad():
                text_out = text_teacher(input_ids=input_ids, attention_mask=attn_mask)
                target_text = text_out.last_hidden_state[:, 0, :] 
            
            # BFloat16 Mixed Precision
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                outputs = model(eeg, mask, subject_ids=sub_ids)
                loss = loss_fn(outputs['z_text'], target_text, outputs['logit_scale'])
                loss = loss / ACCUM_STEPS 
            
            loss.backward()
            total_loss += loss.item() * ACCUM_STEPS

            if (i + 1) % ACCUM_STEPS == 0:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                scheduler.step()
                optimizer.zero_grad()
            
            loop.set_postfix(loss=loss.item() * ACCUM_STEPS)

        # Validation
        avg_train = total_loss / len(train_loader)
        model.eval()
        val_loss = 0
        top1_acc = 0
        top5_acc = 0
        
        with torch.no_grad():
            for batch in val_loader:
                eeg = batch['eeg'].to(device)
                mask = batch['channel_mask'].to(device)
                sub_ids = batch['subject_ids'].to(device)
                input_ids = batch['input_ids'].to(device)
                attn_mask = batch['attention_mask'].to(device)
                
                text_out = text_teacher(input_ids, attention_mask=attn_mask)
                target_text = text_out.last_hidden_state[:, 0, :]
                
                with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    outputs = model(eeg, mask, subject_ids=sub_ids)
                    
                    # Normalized Embeddings for Accuracy
                    z_eeg = outputs['z_text'] / outputs['z_text'].norm(dim=1, keepdim=True)
                    z_text = target_text / target_text.norm(dim=1, keepdim=True)
                    
                    loss = loss_fn(z_eeg, z_text, outputs['logit_scale'])
                    
                    # Accumulate Accuracy
                    top1_acc += calculate_accuracy(z_eeg, z_text, k=1).item()
                    top5_acc += calculate_accuracy(z_eeg, z_text, k=5).item()
                
                val_loss += loss.item()
        
        avg_val = val_loss / len(val_loader)
        avg_top1 = top1_acc / len(val_loader)
        avg_top5 = top5_acc / len(val_loader)
        
        print(f"Epoch {epoch+1}: Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")
        print(f"Accuracy: Top-1: {avg_top1:.2%} | Top-5: {avg_top5:.2%}")
        
        # Save if Accuracy Improves
        if avg_top1 > best_accuracy:
            best_accuracy = avg_top1
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model_acc.pth"))
            print(f"New Best Accuracy Model Saved! ({best_accuracy:.2%})")

if __name__ == "__main__":
    train()