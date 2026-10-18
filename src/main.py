"""Reproduces every figure/table:  python src/main.py   (EPOCHS=12 TL_EPOCHS=6 by default)"""
import os, json, platform, numpy as np, pandas as pd, torch, torch.nn.functional as F
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import f1_score, confusion_matrix
from torchvision import datasets

from data import *
from models import CNN, n_params, layer_table
from train import predict, evaluate
import ablations, transfer
from gradcam import GradCAM

ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
FIG, TAB = f"{ROOT_DIR}/results/figures", f"{ROOT_DIR}/results/tables"
EPOCHS, TL_EPOCHS = int(os.environ.get("EPOCHS", 12)), int(os.environ.get("TL_EPOCHS", 6))
dev = get_device(); set_seed()
hw = torch.cuda.get_device_name(0) if torch.cuda.is_available() else f"CPU ({os.cpu_count()} threads)"
print("Hardware:", hw, "| seed:", SEED)


def save(name): plt.tight_layout(); plt.savefig(f"{FIG}/{name}.png", dpi=150); plt.close()


# ---------------- Task 1(a): filtering view ----------------
raw = datasets.FashionMNIST(ROOT, True, download=True)
img = torch.tensor(np.array(raw[0][0]), dtype=torch.float32)[None, None] / 255
k = {"Sobel-x": [[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], "Sobel-y": [[-1, -2, -1], [0, 0, 0], [1, 2, 1]],
     "Gaussian": [[1, 2, 1], [2, 4, 2], [1, 2, 1]]}
fig, ax = plt.subplots(1, 4, figsize=(10, 3)); ax[0].imshow(img[0, 0], cmap="gray"); ax[0].set_title("original")
for a, (n, w) in zip(ax[1:], k.items()):
    w = torch.tensor(w, dtype=torch.float32)[None, None]; w = w / 16 if n == "Gaussian" else w
    a.imshow(F.conv2d(img, w, padding=1)[0, 0], cmap="gray"); a.set_title(n)
for a in ax: a.axis("off")
save("task1a_filters")

# ---------------- Task 2: ablation suite includes the default CNN ----------------
S, table = ablations.run_all(dev, EPOCHS)
table.to_csv(f"{TAB}/ablation_summary.csv", index=False); print(table)

cnn = S["CNN-2blk-max"]["model"]                      # default architecture (hand table in report)
lt = pd.DataFrame(layer_table(cnn)); lt.to_csv(f"{TAB}/cnn_layer_table.csv", index=False)
print(lt); print("trainable params:", n_params(cnn), "| sum of table:", lt.params.sum())

h = S["CNN-2blk-max"]["hist"]; ep = np.arange(1, len(h["val_acc"]) + 1)
fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
ax[0].plot(ep, h["train_loss"], label="train"); ax[0].plot(ep, h["val_loss"], label="val")
ax[0].axvline(h["min_val_loss_epoch"], ls="--", c="gray"); ax[0].set_title("loss"); ax[0].legend()
ax[1].plot(ep, h["val_acc"]); ax[1].set_title("val accuracy"); save("task2d_curves")

W = cnn.features[0].weight.detach().cpu()            # Task 2(e)
fig, ax = plt.subplots(4, 8, figsize=(8, 4))
for i, a in enumerate(ax.flat): a.imshow(W[i, 0], cmap="gray"); a.axis("off")
save("task2e_kernels")
x = eval_tf()(raw[0][0])[None].to(dev)
with torch.no_grad(): fm = F.relu(cnn.features[0](x))[0].cpu()
fig, ax = plt.subplots(4, 8, figsize=(10, 5))
for i, a in enumerate(ax.flat): a.imshow(fm[i], cmap="gray"); a.axis("off")
save("task2e_featuremaps")

# ---------------- Task 3 plots ----------------
depth = {d: S[f"CNN-{d}blk-max"]["row"]["val_acc"] for d in [1, 2, 3, 4]}
plt.plot(list(depth), list(depth.values()), "o-"); plt.xlabel("conv blocks"); plt.ylabel("val acc")
save("task3c_depth")
nb, na = S[S["_best_noaug"]]["hist"], S[S["_best_noaug"] + "+aug"]["hist"]
fig, ax = plt.subplots(1, 2, figsize=(10, 3.5))
ax[0].plot(nb["val_acc"], label="no aug"); ax[0].plot(na["val_acc"], label="aug"); ax[0].legend(); ax[0].set_title("val acc")
ax[1].plot(np.array(nb["train_acc"]) - np.array(nb["val_acc"]), label="no aug")
ax[1].plot(np.array(na["train_acc"]) - np.array(na["val_acc"]), label="aug"); ax[1].legend(); ax[1].set_title("train-val gap")
save("task3d_augmentation")

# best custom CNN chosen on VALIDATION
cand = [S[S["_best_noaug"]], S[S["_best_noaug"] + "+aug"]]
best_cnn_entry = max(cand, key=lambda e: e["row"]["val_acc"])
best_cnn = best_cnn_entry["model"].to(dev)

# ---------------- Task 4 ----------------
R, final_key = transfer.run_strategies(dev, TL_EPOCHS)
rows = [dict(strategy=k, trainable_params=v["trainable"], best_val_acc=v["hist"]["best_val_acc"],
             val_acc_per_epoch=[round(a, 4) for a in v["hist"]["val_acc"]],
             sec_per_epoch=float(np.mean(v["hist"]["epoch_time"]))) for k, v in R.items()]
pd.DataFrame(rows).to_csv(f"{TAB}/transfer_summary.csv", index=False); print(pd.DataFrame(rows))
with torch.no_grad():
    m0 = R[final_key]["model"].eval(); print("ResNet-18 pre-fc feature map shape @64x64:",
        torch.nn.Sequential(*list(m0.children())[:-2])(torch.zeros(1, 3, 64, 64, device=dev)).shape,
        "| logits:", m0(torch.zeros(1, 3, 64, 64, device=dev)).shape)
for k, v in R.items(): plt.plot(v["hist"]["val_acc"], label=k)
plt.legend(); plt.xlabel("epoch"); plt.ylabel("val acc"); save("task4c_strategies")
best_tl = R[final_key]["model"]                       # selected on validation

# ---------------- Task 5: TEST SET, used once ----------------
_, _, te = get_loaders(aug=False)
_, _, te_tl = transfer.tl_loaders()
res = {}
for name, model, loader in [("MLP", S["MLP"]["model"].to(dev), te), ("Custom CNN", best_cnn, te),
                            ("ResNet-18 (" + final_key + ")", best_tl, te_tl)]:
    p, y = predict(model, loader, dev)
    res[name] = dict(pred=p, y=y, acc=float((p == y).mean()), f1=float(f1_score(y, p, average="macro")))
sec = {"MLP": S["MLP"]["row"]["sec_per_epoch"], "Custom CNN": best_cnn_entry["row"]["sec_per_epoch"],
       "ResNet-18 (" + final_key + ")": float(np.mean(R[final_key]["hist"]["epoch_time"]))}
par = {"MLP": S["MLP"]["row"]["params"], "Custom CNN": n_params(best_cnn),
       "ResNet-18 (" + final_key + ")": R[final_key]["trainable"]}
eff = pd.DataFrame([dict(model=n, trainable_params=par[n], sec_per_epoch=sec[n],
                         test_acc=r["acc"], macro_f1=r["f1"]) for n, r in res.items()])
eff.to_csv(f"{TAB}/efficiency_table.csv", index=False); print(eff)

better = max(res, key=lambda n: res[n]["acc"]); r = res[better]
cm = confusion_matrix(r["y"], r["pred"]); np.savetxt(f"{TAB}/confusion_matrix.csv", cm, fmt="%d", delimiter=",")
plt.figure(figsize=(7, 6)); plt.imshow(cm, cmap="Blues"); plt.xticks(range(10), CLASSES, rotation=60)
plt.yticks(range(10), CLASSES); plt.title(f"Confusion matrix: {better}")
for i in range(10):
    for j in range(10): plt.text(j, i, cm[i, j], ha="center", va="center", fontsize=6)
save("task5_confusion")
off = cm + cm.T; np.fill_diagonal(off, 0); i, j = np.unravel_index(off.argmax(), off.shape)
print(f"Most confused pair: {CLASSES[i]} <-> {CLASSES[j]} ({cm[i, j]} + {cm[j, i]})")

# ---------------- Task 6: Grad-CAM on the ResNet ----------------
model = best_tl.to(dev)
for p in model.parameters(): p.requires_grad_(True)
cam = GradCAM(model, model.layer4[-1])                # block output after final ReLU
ds = te_tl.dataset
p_tl, y_tl = res["ResNet-18 (" + final_key + ")"]["pred"], res["ResNet-18 (" + final_key + ")"]["y"]


def show(ax, idx, title):
    x = ds[idx][0][None].to(dev); heat, c, prob = cam(x)
    im = (x[0, 0].detach().cpu() * IMN_STD[0] + IMN_MEAN[0]).clamp(0, 1)
    ax.imshow(im, cmap="gray"); ax.imshow(heat, cmap="jet", alpha=0.45); ax.axis("off")
    ax.set_title(f"{title}\npred {CLASSES[c]} ({prob[c]:.2f})", fontsize=8)


coat = int(np.where((y_tl == 4) & (p_tl == 4))[0][0]); shirt = int(np.where((y_tl == 6) & (p_tl == 6))[0][0])
wrong = np.where((y_tl == 6) & (p_tl != 6))[0]; wrong = int(wrong[0]) if len(wrong) else int(np.where(y_tl != p_tl)[0][0])
fig, ax = plt.subplots(1, 3, figsize=(9, 3.4))
show(ax[0], coat, "Coat (correct)"); show(ax[1], shirt, "Shirt (correct)"); show(ax[2], wrong, f"MISCLASSIFIED, true {CLASSES[y_tl[wrong]]}")
save("task6_gradcam")

json.dump(dict(hardware=hw, seed=SEED, epochs=EPOCHS, tl_epochs=TL_EPOCHS, best_cnn=best_cnn_entry["row"]["variant"],
               best_tl=final_key, test={n: {k: v for k, v in r.items() if k in ("acc", "f1")} for n, r in res.items()}),
          open(f"{TAB}/run_info.json", "w"), indent=2)
