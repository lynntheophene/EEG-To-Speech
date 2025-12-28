import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from tqdm import tqdm
from transformers import AutoModel
import time

# --- RDNA 4 / ROCm COMPATIBILITY FLAGS ---
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "12.0.0" 
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

# Import your modules
from eeg_encoder import SuperiorEEGEncoder
from brain_bridge import BrainBridge, BrainCLIPLoss
from data_loader import ZuCoDataset, custom_collate_fn

# --- RX 9060 XT (16GB) CONFIGURATION ---
BATCH_SIZE = 128         
EPOCHS = 20
LEARNING_RATE = 1e-4    
DATA_PATH = "./processed_zuco2_diamond"
SAVE_DIR = "./checkpoints_phase1"
TEXT_MODEL_NAME = "bert-base-uncased" 

os.makedirs(SAVE_DIR, exist_ok=True)

def train():
    # 1. Setup Device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        print(f"🚀 Detected GPU: {gpu_name}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"🎮 VRAM: {vram:.2f} GB (Target: 16GB - Perfect)")
    else:
        print("⚠️ GPU NOT DETECTED. Check your ROCm installation.")

    # 2. Load Dataset
    full_dataset = ZuCoDataset(DATA_PATH, tokenizer_name=TEXT_MODEL_NAME)
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    loader_args = dict(
        batch_size=BATCH_SIZE, 
        shuffle=True, 
        collate_fn=custom_collate_fn,
        num_workers=4,  
        persistent_workers=True, 
        pin_memory=True,
        prefetch_factor=2 
    )
    train_loader = DataLoader(train_dataset, **loader_args)
    
    # Validation loader shouldn't shuffle
    loader_args['shuffle'] = False
    val_loader = DataLoader(val_dataset, **loader_args)
    
    print(f"📦 Train Samples: {len(train_dataset)} | Val Samples: {len(val_dataset)}")

    # 3. Initialize Models
    eeg_encoder = SuperiorEEGEncoder(num_channels=105, d_model=768)
    model = BrainBridge(eeg_encoder, text_dim=768).to(device)
    
    # --- COMPILATION FIX ---
    # We default to Eager Mode (Standard) to fix the 491s/it loop.
    # If you want to try compiling again later, use backend="inductor" 
    # BUT DO NOT use mode="reduce-overhead" unless your batch sizes are static.
    print("⚡ Running in Eager Mode (Stability Fix)...")
    
    # Uncomment the line below ONLY if you are sure shapes are static, 
    # otherwise leave it commented for RDNA 4 stability:
    # model = torch.compile(model, backend="inductor") 

    print("❄️ Loading Frozen Text Teacher (BERT)...")
    text_teacher = AutoModel.from_pretrained(TEXT_MODEL_NAME).to(device)
    for param in text_teacher.parameters():
        param.requires_grad = False
    
    # 4. Setup Optimizer
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    loss_fn = BrainCLIPLoss().to(device)
    scaler = torch.amp.GradScaler("cuda")

    # --- Training Loop ---
    best_val_loss = float('inf')
    
    for epoch in range(EPOCHS):
        model.train()
        total_loss = 0
        
        # Added smoothing to tqdm to see stable speed faster
        loop = tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}", smoothing=0.1)
        
        for batch in loop:
            eeg = batch['eeg'].to(device, non_blocking=True)
            mask = batch['channel_mask'].to(device, non_blocking=True)
            input_ids = batch['input_ids'].to(device, non_blocking=True)
            attn_mask = batch['attention_mask'].to(device, non_blocking=True)
            
            with torch.no_grad():
                text_out = text_teacher(input_ids=input_ids, attention_mask=attn_mask)
                target_text_emb = text_out.last_hidden_state[:, 0, :] 
            
            with torch.amp.autocast("cuda"):
                outputs = model(eeg, mask)
                loss = loss_fn(outputs['z_text'], target_text_emb, outputs['logit_scale'])
            
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            loop.set_postfix(loss=loss.item())

        avg_train_loss = total_loss / len(train_loader)
        
        # --- Validation ---
        model.eval()
        val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                eeg = batch['eeg'].to(device)
                mask = batch['channel_mask'].to(device)
                input_ids = batch['input_ids'].to(device)
                attn_mask = batch['attention_mask'].to(device)
                
                text_out = text_teacher(input_ids, attention_mask=attn_mask)
                target_text_emb = text_out.last_hidden_state[:, 0, :]
                
                with torch.amp.autocast("cuda"):
                    outputs = model(eeg, mask)
                    loss = loss_fn(outputs['z_text'], target_text_emb, outputs['logit_scale'])
                val_loss += loss.item()
        
        avg_val_loss = val_loss / len(val_loader)
        print(f"📉 Epoch {epoch+1} Summary: Train: {avg_train_loss:.4f} | Val: {avg_val_loss:.4f}")
        
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model_phase1.pth"))
            print("💾 New Best Model Saved!")

if __name__ == "__main__":
    train()