Structure of the zuco-2 dataset 

zuco-2/
│
├── answers/
│   ├── YAC/
│   │   ├── results_NR_YAC_1.mat
│   │   ├── results_NR_YAC_2.mat
│   │   └── ...
│   ├── YAG/
│   ├── YDR/
│   └── YTL/
│
├── task1-NR/
│   ├── Matlab_files/
│   ├── preprocessed/
│   └── raw_data/
│
├── task2-TSR/
│   ├── Matlab_files/
│   ├── preprocessed/
│   └── raw_data/
│
└── task_materials/


# ZuCo-2 EEG Preprocessing Documentation

This document describes the preprocessing pipeline used to extract **word-level EEG signals and corresponding text** from the **ZuCo-2 dataset**, preparing the data for downstream EEG → language / speech models.

---

## 1. Dataset Overview

**ZuCo-2** contains EEG recordings collected while subjects read sentences.  
The dataset is organized by:

- **Subjects**: `YAC`, `YAG`, `YDR`, `YTL`
- **Tasks**:
  - `NR`  – Normal Reading
  - `TSR` – Task-Specific Reading

Each `.mat` file contains sentence-level data, which further contains **word-level EEG recordings** aligned to individual words.

> ⚠️ ZuCo-2 does **not** contain audio. Only EEG and text are available.

---

## 2. Goal of Preprocessing

The preprocessing script extracts:

- **Word-level EEG segments**
- **Corresponding word text**

and saves them in a compact `.npz` format suitable for:
- EEG → text models
- Multimodal alignment (EEG ↔ text / audio SSL units)
- Diffusion-based speech synthesis pipelines

---

## 3. Output Format

For each **subject × task**, the script produces one file:

