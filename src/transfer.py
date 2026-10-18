"""Task 4: ResNet-18 at 64x64 on a stratified 10k/2k subset."""
import torch, numpy as np
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms as T
from data import ROOT, SEED, IMN_MEAN, IMN_STD, split_indices, stratified, set_seed
from models import resnet18_tl, n_params
from train import fit


def tl_tf():
    return T.Compose([T.Resize(64), T.Grayscale(num_output_channels=3), T.ToTensor(),
                      T.Normalize(IMN_MEAN, IMN_STD)])


def tl_loaders(bs=64):
    full = datasets.FashionMNIST(ROOT, True, download=True, transform=tl_tf())
    test = datasets.FashionMNIST(ROOT, False, download=True, transform=tl_tf())
    tr_idx, va_idx = split_indices()
    tgt = full.targets.numpy()
    tr = stratified(tgt, tr_idx, 1000, SEED); va = stratified(tgt, va_idx, 200, SEED)
    g = torch.Generator().manual_seed(SEED)
    return (DataLoader(Subset(full, tr), bs, shuffle=True, generator=g),
            DataLoader(Subset(full, va), 256), DataLoader(test, 256))


def run_strategies(device, epochs=6, lr=1e-3):
    tl, vl, _ = tl_loaders(); R = {}
    for s in ["feature_extraction", "layer4", "differential"]:
        set_seed(SEED); m, opt = resnet18_tl(s, lr); print(f"\n== {s}: {n_params(m):,} trainable")
        h = fit(m, tl, vl, device, epochs, optimizer=opt)
        R[s] = dict(model=m, hist=h, trainable=n_params(m))
    best = max(R, key=lambda k: R[k]["hist"]["best_val_acc"])                   # VAL only
    set_seed(SEED); m, opt = resnet18_tl(best, lr)
    sch = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=epochs)
    h = fit(m, tl, vl, device, epochs, optimizer=opt, scheduler=sch)
    R[best + "+cosine"] = dict(model=m, hist=h, trainable=n_params(m))
    final = max(R, key=lambda k: R[k]["hist"]["best_val_acc"])
    return R, final
