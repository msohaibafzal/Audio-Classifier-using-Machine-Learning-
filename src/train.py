# ---------- 8. Train ----------
EPOCHS = 20
def run_epoch(dl, train):
    model.train(train)
    loss_sum, correct, tot = 0.0, 0, 0
    ys, ps = [], []
    for x, y in dl:
        x, y = x.to(DEVICE, non_blocking=True), y.to(DEVICE, non_blocking=True)
        with torch.set_grad_enabled(train):
            out = model(x)
            loss = criterion(out, y)
        if train:
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
        pred = out.argmax(1)
        loss_sum += loss.item() * y.size(0)
        correct  += (pred == y).sum().item()
        tot      += y.size(0)
        ys += y.cpu().tolist(); ps += pred.cpu().tolist()
    return loss_sum / tot, correct / tot, f1_score(ys, ps, average="macro")
 
 
best_f1, patience, bad = 0.0, 4, 0
for ep in range(1, EPOCHS + 1):
    trl, tra, trf = run_epoch(train_dl, True)
    with torch.no_grad():
        vl, va, vf1 = run_epoch(val_dl, False)      # one pass, not two
    sched.step()
    print(f"Ep{ep:02d} | train acc {tra:.3f} | val acc {va:.3f} | val macroF1 {vf1:.3f}")
    if vf1 > best_f1:
        best_f1, bad = vf1, 0
        torch.save({"model": model.state_dict(), "classes": CLASSES}, "best.pt")
    else:
        bad += 1
        if bad >= patience:
            print("Early stop.")
            break
print(f"Best val macro-F1: {best_f1:.4f}")
 