"""Task 3: one-factor-at-a-time ablations. Seed, optimizer, lr, epochs fixed."""
import numpy as np, pandas as pd
from data import get_loaders, set_seed, SEED
from models import CNN, mlp, n_params
from train import fit


def run(name, build, aug, device, epochs, store):
    set_seed(SEED)
    tl, vl, _ = get_loaders(aug=aug)
    model = build(); print(f"\n== {name} ({n_params(model):,} params)")
    h = fit(model, tl, vl, device, epochs)
    store[name] = dict(model=model, hist=h, row=dict(
        variant=name, params=n_params(model), val_acc=h["best_val_acc"],
        best_epoch=h["best_epoch"], sec_per_epoch=float(np.mean(h["epoch_time"]))))
    return h


def run_all(device, epochs=12):
    S = {}
    run("MLP", lambda: mlp(), False, device, epochs, S)                       # 3(a)
    for pool in ["max", "avg", "stride"]:                                      # 3(b)
        run(f"CNN-2blk-{pool}", lambda p=pool: CNN(2, p), False, device, epochs, S)
    for d in [1, 3, 4]:                                                        # 3(c) (d=2 above)
        run(f"CNN-{d}blk-max", lambda d=d: CNN(d, "max"), False, device, epochs, S)
    cnn_names = [k for k in S if k.startswith("CNN")]
    best = max(cnn_names, key=lambda k: S[k]["row"]["val_acc"])                # selected on VAL
    blocks = int(best.split("-")[1][0]); pool = best.split("-")[2]
    S["_best_noaug"] = best
    run(best + "+aug", lambda: CNN(blocks, pool), True, device, epochs, S)     # 3(d)
    S["_best_cfg"] = (blocks, pool)
    table = pd.DataFrame([v["row"] for k, v in S.items() if not k.startswith("_")])
    return S, table
