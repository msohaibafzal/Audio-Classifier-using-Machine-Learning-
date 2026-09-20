# Audio Classifier using Machine Learning

A PyTorch CNN that classifies 3-second audio clips into **Music**, **Noise**, or **Speech**. Trained on the MUSAN corpus as a Signals and Systems Lab semester project (Fall 2024).

---

## Table of Contents

- [Overview](#overview)
- [Results](#results)
- [Dataset](#dataset)
- [Pipeline](#pipeline)
- [Handling Class Imbalance](#handling-class-imbalance)
- [Data Augmentation](#data-augmentation)
- [Model Architecture](#model-architecture)
- [Training](#training)
- [Project Structure](#project-structure)
- [Setup](#setup)
- [Usage](#usage)
- [Reproducibility](#reproducibility)
- [Notebook Walkthrough](#notebook-walkthrough)
- [Course Objectives](#course-objectives)
- [Limitations](#limitations)
- [References](#references)

---

## Overview

This project implements a complete audio classification system that distinguishes between three broad categories of sound: **music**, **noise**, and **speech**. It was built to satisfy the requirements of the Signals and Systems Lab semester project, which asks for:

- Signal preprocessing using digital filters
- Pitch / mel-frequency feature extraction
- Classification of audio into speech, music, and noise
- Testing and evaluation

The pipeline covers preprocessing (pre-emphasis, band-pass filtering), feature extraction (log-mel spectrograms, MFCCs, zero-crossing rate, RMS, spectral centroid), a CNN classifier in PyTorch, and full evaluation with per-class metrics and a confusion matrix.

---

## Results

Test set: 1,298 clips, evaluated on a held-out split with no source-file leakage.

| Class  | Precision | Recall | F1-score | Support |
|--------|-----------|--------|----------|---------|
| music  | 0.9442    | 0.9264 | 0.9352   | 584     |
| noise  | 0.8415    | 0.9162 | 0.8773   | 394     |
| speech | 0.9865    | 0.9125 | 0.9481   | 320     |
| **macro avg** | **0.9240** | **0.9184** | **0.9202** | **1298** |

**Overall accuracy: 91.99%**
**Best validation macro-F1: 0.9237** (epoch 2)

### Confusion Matrix

![Confusion Matrix](results/confusion.png)

### Observations

- **Speech** has the highest precision (0.9865) — the model almost never predicts speech when it isn't.
- **Noise** has the lowest precision (0.8415) — some music and speech clips get misclassified as noise, which is expected given MUSAN's noise subset is small and acoustically diverse.
- Overall macro-F1 of **0.9202** indicates balanced performance across classes despite the training-set imbalance.

---

## Dataset

**MUSAN** — a corpus of music, speech, and noise recordings used widely in audio research.

- Original release: [OpenSLR #17](https://www.openslr.org/17/)
- Kaggle mirror used here: [nhattruongdev/musan-noise](https://www.kaggle.com/datasets/nhattruongdev/musan-noise)

### Structure

```
musan/
├── music/
├── noise/
└── speech/
```

### Statistics After Cropping

| Split      | Music | Noise | Speech | Total |
|------------|-------|-------|--------|-------|
| Train      | 4,191 | 3,058 | 2,750  | 9,999 |
| Validation | 500   | 384   | 336    | 1,220 |
| Test       | 584   | 394   | 320    | 1,298 |
| **Total**  | 5,275 | 3,836 | 3,406  | 10,509 |

- 2,016 unique source files
- Up to 8 non-overlapping crops per source file (capped to prevent long music tracks from dominating)

---

## Pipeline

```
Audio file (16 kHz, mono)
        │
        ▼
Load 3-second clip
        │
        ▼
Pre-emphasis  (y[n] = y[n] - 0.97 · y[n-1])
        │
        ▼
Band-pass filter  (4th-order Butterworth, 50–7600 Hz)
        │
        ▼
Log-mel spectrogram  (128 mel bands, n_fft=1024, hop=512)
        │
        ▼
Z-normalize + resize to 128×128
        │
        ▼
CNN  →  Softmax  →  {music, noise, speech}
```

### Feature Details

| Feature | Purpose |
|---|---|
| **Log-mel spectrogram (128×128)** | Main CNN input |
| **MFCC (40 coefficients)** | Complementary representation, inspected in notebook |
| **Zero-crossing rate** | Distinguishes tonal (music) from noisy (percussive) signals |
| **RMS energy** | Loudness / signal power |
| **Spectral centroid** | "Brightness" of the signal — higher for noise, lower for music |

---

## Handling Class Imbalance

MUSAN is heavily imbalanced in its raw form:

- Music: ~42 hours
- Speech: ~60 hours
- Noise: only ~6 hours

Without correction, the model would learn to predict music/speech and ignore noise. Two complementary techniques are used:

1. **WeightedRandomSampler** — during training, batches are drawn so each class appears roughly equally often. Sample weights are inversely proportional to class frequency.
2. **Label smoothing (0.05)** — softens targets, improves calibration and generalization on the minority classes.

**Note:** The loss function is *not* additionally class-weighted. Combining a weighted sampler with a class-weighted loss over-corrects and degrades majority-class performance, so only one imbalance mechanism is applied at the loss level.

---

## Data Augmentation

Applied only during training, in the `Dataset.__getitem__` method:

| Augmentation | Probability | Details |
|---|---|---|
| Crop offset jitter | always | ±1.5 s on the start offset |
| Time shift | 50% | ±20 frames (roll along time axis) |
| Additive Gaussian noise | 30% | σ = 0.05 on the normalized log-mel |
| Frequency masking | 30% | mask width 4–16 mel bands |
| Time masking | 30% | mask width 4–16 frames |

Frequency and time masking are the spectrogram equivalents of SpecAugment, which is standard for audio CNNs.

---

## Model Architecture

A small 2D CNN with ~240 K trainable parameters.

| Layer | Output Shape | Params |
|---|---|---|
| Conv2d(1→32, 3×3) + BN + ReLU + MaxPool(2) | 64×64×32 | 320 |
| Conv2d(32→64, 3×3) + BN + ReLU + MaxPool(2) | 32×32×64 | 18,496 |
| Conv2d(64→128, 3×3) + BN + ReLU + MaxPool(2) | 16×16×128 | 73,856 |
| Conv2d(128→128, 3×3) + BN + ReLU + MaxPool(2) | 8×8×128 | 147,584 |
| AdaptiveAvgPool2d(1) | 128 | 0 |
| Dropout(0.4) | 128 | 0 |
| Linear(128→3) | 3 | 387 |

**Total trainable parameters:** ~240,643

### Design choices

- **BatchNorm after every conv** — stabilizes training and allows a higher learning rate.
- **AdaptiveAvgPool2d(1)** instead of Flatten — drastically reduces parameters compared to a fully-connected head, avoiding overfitting on a modest dataset.
- **Dropout(0.4)** before the classifier head — further regularization.

---

## Training

| Setting | Value |
|---|---|
| Optimizer | AdamW |
| Learning rate | 3e-4 |
| Weight decay | 1e-4 |
| LR schedule | CosineAnnealingLR (T_max=20) |
| Loss | CrossEntropyLoss(label_smoothing=0.05) |
| Batch size | 64 |
| Max epochs | 20 |
| Early stopping | patience 4 on validation macro-F1 |
| Device | CUDA (Kaggle T4) |

### Training Log

```
Ep01 | train acc 0.848 | val acc 0.894 | val macroF1 0.893
Ep02 | train acc 0.903 | val acc 0.923 | val macroF1 0.924   ← best
Ep03 | train acc 0.917 | val acc 0.920 | val macroF1 0.921
Ep04 | train acc 0.922 | val acc 0.919 | val macroF1 0.919
Ep05 | train acc 0.927 | val acc 0.865 | val macroF1 0.870
Ep06 | train acc 0.925 | val acc 0.921 | val macroF1 0.920
Early stop.
Best val macro-F1: 0.9237
```

Training stopped at epoch 6 (4 epochs without improvement). The best checkpoint was from epoch 2 — after that, the model began to overfit (train accuracy continued to climb while validation dipped).

---

## Project Structure

```
Audio-Classifier-using-Machine-Learning-/
│
├── Notebook/
│   └── audio-classification.ipynb    # Full end-to-end pipeline
├── dataset/                          # MUSAN (not tracked in git)
├── docs/                             # Report, references
├── results/
│   └── confusion.png                 # Test-set confusion matrix
├── src/                              # Refactored source (optional)
├── requirements.txt
├── .gitignore
└── README.md
```

---

## Setup

### Requirements

```bash
pip install -r requirements.txt
```

Contents:

```
torch
torchaudio
librosa
numpy
pandas
scikit-learn
matplotlib
seaborn
scipy
tqdm
soundfile
```

### Dataset

Place MUSAN under `dataset/musan/` with the layout:

```
dataset/musan/
├── music/
├── noise/
└── speech/
```

On Kaggle, attach the dataset as an input; the notebook auto-detects the path.

---

## Usage

### Run the notebook

Open `Notebook/audio-classification.ipynb` and run all cells. It performs:

1. Setup and seed fixing
2. Dataset root discovery
3. Clip index building
4. Grouped train/val/test split
5. Feature extraction
6. Dataset class + balanced sampler
7. Model definition
8. Training with early stopping
9. Test-set evaluation + confusion matrix

The trained model is saved as `best.pt` inside the notebook's working directory.

### Inference (single file)

```python
import torch
from model import AudioCNN

model = AudioCNN().cuda()
ckpt = torch.load("best.pt", map_location="cuda", weights_only=False)
model.load_state_dict(ckpt["model"])
model.eval()

from features import extract_features, to_image
x = to_image(extract_features("path/to/audio.wav")).unsqueeze(0).cuda()
pred = model(x).argmax(1).item()
print(ckpt["classes"][pred])
```

---

## Notebook Walkthrough

The notebook is divided into 10 sections, each mapped to a project requirement:

| Section | Cell | Purpose |
|---|---|---|
| 0. Setup | 1 | Imports, seeds, hyperparameters |
| 1. Dataset discovery | 2 | Locate `music/`, `noise/`, `speech/` |
| 2. Clip index | 3 | Build training table with per-file crops |
| 3. Grouped split | 4 | Train/val/test split by source file |
| 4. Feature extraction | 5 | Pre-emphasis, band-pass, log-mel |
| 5. Dataset class | 6 | PyTorch Dataset with augmentation |
| 6. Sampler + loaders | 7 | WeightedRandomSampler, DataLoaders |
| 7. Model | 8 | AudioCNN definition |
| 8. Training loop | 9 | Epoch loop, early stopping |
| 9. Evaluation | 10 | Classification report, confusion matrix |

---
