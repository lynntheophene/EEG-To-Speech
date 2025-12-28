import h5py
import numpy as np

# Path to the file that is failing
file_path = "/media/mando/0E5F-F655/datasets/zuco-2/task1 - NR/Matlab files/resultsYRK_NR.mat"

def scan_node(name, obj):
    """Callback function to inspect every node in the HDF5 file."""
    indent = "  " * name.count('/')
    node_name = name.split('/')[-1]
    
    print(f"{indent}Path: {node_name}")
    print(f"{indent}  Type: {type(obj)}")
    
    if isinstance(obj, h5py.Dataset):
        print(f"{indent}  Shape: {obj.shape}")
        print(f"{indent}  Dtype: {obj.dtype}")
        
        # If it's small and numeric, show some data
        if obj.size < 10 and np.issubdtype(obj.dtype, np.number):
            print(f"{indent}  Values: {obj[()]}")
        # If it's a reference, let the user know
        if h5py.check_dtype(ref=obj.dtype):
            print(f"{indent}  [!] Contains Object References")

    elif isinstance(obj, h5py.Group):
        print(f"{indent}  Keys: {list(obj.keys())[:10]} (Total: {len(obj.keys())})")

print(f"--- Deep Scanning: {file_path} ---")
with h5py.File(file_path, "r") as f:
    # We only scan the first few levels to avoid massive logs
    # But we focus specifically on sentenceData
    if "sentenceData" in f:
        sd = f["sentenceData"]
        print("\n[STEP 1] Inspecting sentenceData:")
        for key in sd.keys():
            item = sd[key]
            print(f"  Field: {key} | Type: {type(item)}")
            if hasattr(item, 'shape'): print(f"    Shape: {item.shape}")

        # Deep dive into the first word entry
        print("\n[STEP 2] Attempting to reach a single word object:")
        try:
            # sentenceData -> word -> first sentence ref
            word_col = sd['word']
            first_sent_ref = word_col[0][0]
            print(f"  Found Sentence Ref: {first_sent_ref}")
            
            sent_obj = f[first_sent_ref]
            print(f"  Sent Object Type: {type(sent_obj)}")
            print(f"  Sent Object Keys: {list(sent_obj.keys())}")
            
            # Pick a word key (usually '0' or word)
            sample_key = 'word' if 'word' in sent_obj else '0'
            first_word_ref = sent_obj[sample_key]
            
            # If it's an array of refs, get the first one
            if hasattr(first_word_ref, 'shape'):
                first_word_ref = first_word_ref[0][0]
                
            word_data = f[first_word_ref]
            print(f"\n[STEP 3] Success! Keys found inside a Word Object:")
            print(f"  {list(word_data.keys())}")
            
            for k in word_data.keys():
                val = word_data[k]
                shape = val.shape if hasattr(val, 'shape') else "N/A"
                print(f"    - {k}: {type(val)} | Shape: {shape}")

        except Exception as e:
            print(f"\n[!] Failed to reach word level: {e}")