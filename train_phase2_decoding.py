"""
Training Phase 2 - EEG to Text Decoding (Seq2Seq)

This script implements Phase 2 of the EEG-to-Speech pipeline: fine-tuning a BART decoder
to generate natural language text directly from EEG signals using the encoder from Phase 1.

Key responsibilities:
  - Load Phase 1 pretrained EEG encoder
  - Combine encoder with BART decoder for sequence-to-sequence generation
  - Train BrainTranslator to decode EEG -> text generation
  - Evaluate using BLEU, METEOR, and other NLG metrics
  - Handle subject-specific fine-tuning if needed

Requires: Phase 1 checkpoint, BART model, ZuCo dataset
Output: Final EEG-to-text generation model
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, random_split, Subset
from tqdm import tqdm
from transformers import BartTokenizer, BartForConditionalGeneration, get_cosine_schedule_with_warmup
from transformers.modeling_outputs import BaseModelOutput
import gc 

# --- 🛡️ CRITICAL STABILITY FIXES FOR RDNA 4 ---
# 1. Force Math Backend
torch.backends.cuda.enable_flash_sdp(False)
torch.backends.cuda.enable_mem_efficient_sdp(False)
torch.backends.cuda.enable_math_sdp(True)

# 2. Disable MIOpen Fusion (Prevents Convolution Crashes)
os.environ["MIOPEN_DISABLE_CACHE"] = "1" 
os.environ["MIOPEN_ENABLE_LOGGING"] = "0"

# --- RDNA 4 / ROCm FLAGS ---
os.environ["HSA_OVERRIDE_GFX_VERSION"] = "12.0.0" 
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "0"
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from eeg_encoder import SuperiorEEGEncoder
from data_loader import ZuCoDataset, fixed_collate_fn

# --- CONFIG (ADJUSTED FOR FP32) ---
BATCH_SIZE = 32        # REDUCED from 32 (FP32 uses 2x RAM)
ACCUM_STEPS = 4         # INCREASED from 2 (To keep effective batch = 64)
EPOCHS = 30             
LEARNING_RATE = 5e-5    
DATA_PATH = "./optimized_zuco"
PHASE1_CHECKPOINT = "./checkpoints_phase1/best_model_acc.pth"
SAVE_DIR = "./checkpoints_phase2"
DECODER_MODEL = "facebook/bart-base" 
SUBJECT_FILTER = None 

os.makedirs(SAVE_DIR, exist_ok=True)

class BrainTranslator(nn.Module):
    def __init__(self, encoder, decoder_name):
        super().__init__()
        self.encoder = encoder
        self.decoder = BartForConditionalGeneration.from_pretrained(decoder_name)
        
    def forward(self, eeg, mask, input_ids, attention_mask, subject_ids=None):
        brain_embedding = self.encoder(eeg, mask, subject_ids)
        brain_hidden_states = brain_embedding.unsqueeze(1)
        outputs = self.decoder(
            encoder_outputs=(brain_hidden_states,), 
            labels=input_ids,                      
            decoder_attention_mask=attention_mask   
        )
        return outputs

def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f" Phase 2 Device: {device} (Running in FP32 Mode)")

    # 1. Load Data
    full_dataset = ZuCoDataset(DATA_PATH, tokenizer_name=DECODER_MODEL)
    if SUBJECT_FILTER is not None:
        indices = [i for i, x in enumerate(full_dataset.subject_ids) if x == SUBJECT_FILTER]
        full_dataset = Subset(full_dataset, indices)
    
    train_size = int(0.9 * len(full_dataset))
    val_size = len(full_dataset) - train_size
    train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

    # Low Worker Count for Stability
    loader_args = dict(
        batch_size=BATCH_SIZE, shuffle=True, collate_fn=fixed_collate_fn,
        num_workers=2, persistent_workers=True, pin_memory=True,
        prefetch_factor=2, drop_last=True 
    )
    train_loader = DataLoader(train_dataset, **loader_args)
    loader_args['shuffle'] = False
    val_loader = DataLoader(val_dataset, **loader_args)

    # 2. Initialize Models
    eeg_encoder = SuperiorEEGEncoder(num_channels=105, d_model=768)
    model = BrainTranslator(eeg_encoder, DECODER_MODEL).to(device)

    # Resume Logic
    phase2_ckpt = os.path.join(SAVE_DIR, "best_model_phase2.pth")
    emergency_ckpt = os.path.join(SAVE_DIR, "emergency_save.pth")
    
    if os.path.exists(emergency_ckpt):
        print(f"Found EMERGENCY Checkpoint. Recovering...")
        try:
            model.load_state_dict(torch.load(emergency_ckpt, map_location=device))
        except:
            print(" Emergency file corrupted. Skipping.")
    elif os.path.exists(phase2_ckpt):
        print(f"Found PHASE 2 Checkpoint. Resuming...")
        model.load_state_dict(torch.load(phase2_ckpt, map_location=device))
    elif os.path.exists(PHASE1_CHECKPOINT):
        print(f"Loading PHASE 1 weights...")
        state_dict = torch.load(PHASE1_CHECKPOINT, map_location=device)
        encoder_dict = {k.replace('eeg_encoder.', ''): v for k, v in state_dict.items() if k.startswith('eeg_encoder.')}
        eeg_encoder.load_state_dict(encoder_dict, strict=False)

    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.01)
    num_training_steps = len(train_loader) * EPOCHS
    scheduler = get_cosine_schedule_with_warmup(optimizer, int(0.1 * num_training_steps), num_training_steps)

    best_val_loss = float('inf')
    tokenizer = BartTokenizer.from_pretrained(DECODER_MODEL)

    print("Starting Robust Training Loop...")
    
    try:
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
                
                if model.training:
                    eeg = eeg + (torch.randn_like(eeg) * 0.01)

                # --- CHANGED: Use float32 instead of bfloat16 ---
                with torch.amp.autocast("cuda", dtype=torch.float32):
                    outputs = model(eeg, mask, input_ids, attn_mask, sub_ids)
                    loss = outputs.loss / ACCUM_STEPS

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
            
            with torch.no_grad():
                for batch in val_loader:
                    eeg = batch['eeg'].to(device)
                    mask = batch['channel_mask'].to(device)
                    sub_ids = batch['subject_ids'].to(device)
                    input_ids = batch['input_ids'].to(device)
                    attn_mask = batch['attention_mask'].to(device)
                    
                    # --- CHANGED: Use float32 ---
                    with torch.amp.autocast("cuda", dtype=torch.float32):
                        outputs = model(eeg, mask, input_ids, attn_mask, sub_ids)
                        loss = outputs.loss
                    val_loss += loss.item()

            avg_val = val_loss / len(val_loader)
            print(f"Summary: Train Loss: {avg_train:.4f} | Val Loss: {avg_val:.4f}")
            
            torch.cuda.empty_cache()
            gc.collect()

            if avg_val < best_val_loss:
                best_val_loss = avg_val
                torch.save(model.state_dict(), os.path.join(SAVE_DIR, "best_model_phase2.pth"))
                print("Saved Best Generator!")
                
                sample_eeg = eeg[0].unsqueeze(0)
                sample_mask = mask[0].unsqueeze(0)
                sample_sub = sub_ids[0].unsqueeze(0)
                truth = tokenizer.decode(input_ids[0], skip_special_tokens=True)
                print(f"    Target: '{truth}'")

                with torch.no_grad():
                    emb = model.encoder(sample_eeg, sample_mask, sample_sub).unsqueeze(1)
                    encoder_outputs = BaseModelOutput(last_hidden_state=emb)
                    generated_ids = model.decoder.generate(
                        encoder_outputs=encoder_outputs, 
                        max_length=30, num_beams=5, no_repeat_ngram_size=2, 
                        repetition_penalty=1.2, early_stopping=True
                    )
                    pred_text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
                    print(f"    Brain:  '{pred_text}'")

    except Exception as e:
        print(f"\n CRASH DETECTED: {e}")
        print(" Saving EMERGENCY checkpoint...")
        try:
            torch.save(model.state_dict(), os.path.join(SAVE_DIR, "emergency_save.pth"))
            print(" Safe.")
        except:
            print(" Could not save emergency checkpoint (Drive/GPU likely dead).")
        raise e

if __name__ == "__main__":
    train()