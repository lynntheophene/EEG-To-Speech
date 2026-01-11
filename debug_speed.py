"""
Speed Benchmarking Tool

This utility script benchmarks the forward pass speed of the EEG encoder and BrainBridge
models on GPU. Useful for:
  - Profiling performance on different hardware (AMD RDNA, NVIDIA, etc.)
  - Identifying bottlenecks in the model architecture
  - Validating GPU acceleration is working correctly
  - Testing compile() optimization impact

Runs on dummy batch data (no disk I/O) to isolate model computation time.
Reports throughput in samples/second and latency.
"""

import torch
import time
import os

# 1. Force AMD Optimization
os.environ["TORCH_ROCM_AOTRITON_ENABLE_EXPERIMENTAL"] = "1"

# Import your model
from eeg_encoder import SuperiorEEGEncoder
from brain_bridge import BrainBridge

def speed_test():
    # Setup Device
    if torch.cuda.is_available():
        device = torch.device('cuda')
        name = torch.cuda.get_device_name(0)
        print(f" Device: {name}")
    else:
        print(" CRITICAL: Running on CPU! Check your PyTorch install.")
        return

    # Initialize Model
    print("⏳ Initializing Model...")
    encoder = SuperiorEEGEncoder(num_channels=105, d_model=768)
    model = BrainBridge(encoder).to(device)
    
    # OPTIONAL: Uncomment to test if compile is the issue
    # model = torch.compile(model) 

    # Fake Data (In VRAM - No Disk I/O)
    batch_size = 32
    dummy_eeg = torch.randn(batch_size, 105, 200).to(device)
    dummy_mask = torch.ones(batch_size, 105).to(device)
    
    print(" Starting Warmup (10 steps)...")
    # Warmup to wake up the GPU
    for _ in range(10):
        _ = model(dummy_eeg, dummy_mask)
    torch.cuda.synchronize()

    print(" Measuring Speed (50 steps)...")
    start_time = time.time()
    
    for _ in range(50):
        output = model(dummy_eeg, dummy_mask)
        # Force wait for GPU to finish
        torch.cuda.synchronize()
        
    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / 50
    
    print(f"\ RESULTS:")
    print(f"Total Time: {total_time:.4f}s")
    print(f"Time Per Batch: {avg_time:.4f}s")
    
    if avg_time < 0.2:
        print(" GPU IS FAST. The problem is your Disk/DataLoader.")
    else:
        print(" GPU IS SLOW. The problem is your ROCm/PyTorch setup.")

if __name__ == "__main__":
    speed_test()