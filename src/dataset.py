# ---------- 5. Dataset ----------
class MusanDS(Dataset):
    def __init__(self, frame, train=False):
        self.df = frame.reset_index(drop=True)
        self.train = train
 
    def __len__(self):
        return len(self.df)
 
    def __getitem__(self, i):
        row = self.df.iloc[i]
        offset = float(row["offset"])
        if self.train:                                   # jitter the crop start
            offset = max(0.0, offset + random.uniform(-DUR / 2, DUR / 2))
        try:
            logmel = extract_features(row["path"], offset)
        except Exception:
            logmel = np.zeros((N_MELS, TARGET_LEN // HOP + 1), dtype=np.float32)
        x = to_image(logmel)
 
        if self.train:
            if random.random() < 0.5:                                    # time shift
                x = torch.roll(x, shifts=random.randint(-20, 20), dims=2)
            if random.random() < 0.3:                                    # noise
                x = x + 0.05 * torch.randn_like(x)
            if random.random() < 0.3:                                    # freq mask
                f0 = random.randint(0, IMG[0] - 17)
                x[:, f0:f0 + random.randint(4, 16), :] = 0
            if random.random() < 0.3:                                    # time mask
                t0 = random.randint(0, IMG[1] - 17)
                x[:, :, t0:t0 + random.randint(4, 16)] = 0
        return x, int(row["label"])
 
 
train_ds = MusanDS(train_df, train=True)
val_ds   = MusanDS(val_df)
test_ds  = MusanDS(test_df)
 
# ---------- 6. Balanced sampler + DataLoaders ----------
# Imbalance is corrected ONCE, in the sampler. Do not also weight the loss —
# stacking both over-corrects and hurts the majority classes.
counts = (train_df["label"].value_counts()
          .reindex(range(len(CLASSES)), fill_value=0).values.astype(float))
class_w = 1.0 / np.maximum(counts, 1)
sample_w = class_w[train_df["label"].values]
sampler = WeightedRandomSampler(torch.as_tensor(sample_w, dtype=torch.double),
                                num_samples=len(sample_w), replacement=True)
 
BS, NW = 64, 2
train_dl = DataLoader(train_ds, batch_size=BS, sampler=sampler,
                      num_workers=NW, pin_memory=(DEVICE == "cuda"), drop_last=True)
val_dl   = DataLoader(val_ds,  batch_size=BS, shuffle=False, num_workers=NW)
test_dl  = DataLoader(test_ds, batch_size=BS, shuffle=False, num_workers=NW)
 