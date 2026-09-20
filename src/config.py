# ---------- 0. Setup ----------
import os, glob, random, warnings
import numpy as np, pandas as pd
import soundfile as sf
import librosa
import torch, torch.nn as nn, torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from scipy.signal import butter, sosfilt
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import matplotlib.pyplot as plt, seaborn as sns
 
warnings.filterwarnings("ignore")
 
SEED = 42
random.seed(SEED); np.random.seed(SEED)
torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
print("Device:", DEVICE)
 
SR       = 16000
DUR      = 3.0                      # seconds per training clip
N_MELS   = 128
N_MFCC   = 40
HOP      = 512
IMG      = (128, 128)               # mel resized to this before the CNN
MAX_CROPS_PER_FILE = 8              # cap so long music tracks don't dominate
CLASSES  = ["music", "noise", "speech"]      # alphabetical -> 0, 1, 2
C2I      = {c: i for i, c in enumerate(CLASSES)}
AUDIO_EXT = (".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac")
Device: cuda
# ---------- 1. Find the dataset root ----------
# The exact folder depth varies between Kaggle dataset uploads, so instead of
# hard-coding it we search for the directories actually named music/noise/speech.
SEARCH_ROOTS = [
    "/kaggle/input/datasets/nhattruongdev/musan-noise",   # path you gave
    "/kaggle/input/musan-noise",                          # usual Kaggle form
    "/kaggle/input",                                      # last-resort sweep
]
 
 
def find_class_dirs(roots, classes, max_depth=6):
    """Return {class_name: [dirs]} for every directory named after a class."""
    found = {c: [] for c in classes}
    for root in roots:
        if not os.path.isdir(root):
            continue
        base_depth = root.rstrip("/").count("/")
        for dirpath, dirnames, _ in os.walk(root):
            if dirpath.count("/") - base_depth > max_depth:
                dirnames[:] = []
                continue
            name = os.path.basename(dirpath).lower()
            if name in found:
                found[name].append(dirpath)
        if any(found[c] for c in classes):
            break                                   # stop at the first root that works
    return found
 
 
class_dirs = find_class_dirs(SEARCH_ROOTS, CLASSES)
for c in CLASSES:
    print(f"{c:6s}: {len(class_dirs[c])} dir(s)", class_dirs[c][:2])
 
missing = [c for c in CLASSES if not class_dirs[c]]
if missing:
    print("\n!! No directory found for:", missing)
    print("Here is what /kaggle/input actually contains — copy the right path in:")
    for dirpath, dirnames, filenames in os.walk("/kaggle/input"):
        depth = dirpath.count("/") - 2
        if depth > 3:
            dirnames[:] = []
            continue
        print("  " * depth + os.path.basename(dirpath) + f"/  ({len(filenames)} files)")
    raise SystemExit(
        "Fix SEARCH_ROOTS above. Note: this dataset slug is 'musan-noise' — if it only "
        "ships the noise subset you also need to attach the music and speech subsets."
    )
 