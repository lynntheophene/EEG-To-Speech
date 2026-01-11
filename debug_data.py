import torch
from torch.utils.data import DataLoader
from data_loader import ZuCoDataset, fixed_collate_fn

# Config
DATA_PATH = "./optimized_zuco"
BATCH_SIZE = 64

def check_data():
    print(f"🕵️ Inspecting data in {DATA_PATH}...")
    
    # Load dataset
    dataset = ZuCoDataset(DATA_PATH)
    loader = DataLoader(dataset, batch_size=BATCH_SIZE, collate_fn=fixed_collate_fn)
    
    # Grab one batch
    batch = next(iter(loader))
    eeg = batch['eeg']
    mask = batch['channel_mask']
    
    # --- CHECK 1: Is the EEG data just zeros? ---
    print("\n📊 EEG DATA STATISTICS:")
    print(f"Shape: {eeg.shape}")
    print(f"Min Value: {eeg.min().item():.4f}")
    print(f"Max Value: {eeg.max().item():.4f}")
    print(f"Mean Value: {eeg.mean().item():.4f}")
    
    if eeg.abs().sum() == 0:
        print("❌ CRITICAL ERROR: EEG Tensors are all ZEROS!")
    else:
        print("✅ EEG data looks valid (non-zero).")

    # --- CHECK 2: Is the mask working? ---
    print("\n🎭 MASK STATISTICS:")
    print(f"Mask True Count (avg per sample): {mask.sum(dim=1).float().mean().item():.1f} / 840")
    
    if mask.sum() == 0:
        print("❌ CRITICAL ERROR: Mask is empty!")

if __name__ == "__main__":
    check_data()