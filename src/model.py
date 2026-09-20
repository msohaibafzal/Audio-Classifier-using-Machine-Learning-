# ---------- 7. Model ----------
class AudioCNN(nn.Module):
    def __init__(self, n_cls=3):
        super().__init__()
 
        def blk(i, o):
            return nn.Sequential(nn.Conv2d(i, o, 3, padding=1), nn.BatchNorm2d(o),
                                 nn.ReLU(inplace=True), nn.MaxPool2d(2))
 
        self.net = nn.Sequential(blk(1, 32), blk(32, 64), blk(64, 128), blk(128, 128),
                                 nn.AdaptiveAvgPool2d(1))
        self.head = nn.Sequential(nn.Flatten(), nn.Dropout(0.4), nn.Linear(128, n_cls))
 
    def forward(self, x):
        return self.head(self.net(x))
 
 
model = AudioCNN(len(CLASSES)).to(DEVICE)
criterion = nn.CrossEntropyLoss(label_smoothing=0.05)
optim = torch.optim.AdamW(model.parameters(), lr=3e-4, weight_decay=1e-4)
EPOCHS = 20
sched = torch.optim.lr_scheduler.CosineAnnealingLR(optim, T_max=EPOCHS)