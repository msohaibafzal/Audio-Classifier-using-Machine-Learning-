# ---------- 9. Test ----------
ckpt = torch.load("best.pt", map_location=DEVICE, weights_only=False)
model.load_state_dict(ckpt["model"])
model.eval()
 
ys, ps = [], []
with torch.no_grad():
    for x, y in test_dl:
        ps += model(x.to(DEVICE)).argmax(1).cpu().tolist()
        ys += y.tolist()
 
present = sorted(set(ys) | set(ps))
print(classification_report(ys, ps, labels=present,
                            target_names=[CLASSES[i] for i in present], digits=4))
cm = confusion_matrix(ys, ps, labels=present)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=[CLASSES[i] for i in present],
            yticklabels=[CLASSES[i] for i in present])
plt.xlabel("Predicted"); plt.ylabel("True"); plt.title("Confusion Matrix")
plt.tight_layout(); plt.show()