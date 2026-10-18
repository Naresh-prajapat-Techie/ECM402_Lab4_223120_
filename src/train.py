"""Training / validation loops. Model selection uses VALIDATION only."""
import copy, time, torch, torch.nn as nn, numpy as np


def freeze_bn_if_frozen(model):
    for m in model.modules():
        if isinstance(m, nn.BatchNorm2d) and not any(p.requires_grad for p in m.parameters()):
            m.eval()


@torch.no_grad()
def evaluate(model, loader, device):
    model.eval(); crit = nn.CrossEntropyLoss(reduction="sum")
    loss = correct = n = 0
    for x, y in loader:
        x, y = x.to(device), y.to(device)
        o = model(x); loss += crit(o, y).item()
        correct += (o.argmax(1) == y).sum().item(); n += len(y)
    return loss / n, correct / n


@torch.no_grad()
def predict(model, loader, device):
    model.eval(); P, Y = [], []
    for x, y in loader:
        P.append(model(x.to(device)).argmax(1).cpu()); Y.append(y)
    return torch.cat(P).numpy(), torch.cat(Y).numpy()


def fit(model, tl, vl, device, epochs=12, lr=1e-3, optimizer=None, scheduler=None, verbose=True):
    model.to(device)
    opt = optimizer or torch.optim.Adam(model.parameters(), lr=lr)
    crit = nn.CrossEntropyLoss()
    h = dict(train_loss=[], train_acc=[], val_loss=[], val_acc=[], epoch_time=[])
    best, best_state, best_ep = -1, None, 0
    for ep in range(epochs):
        model.train(); freeze_bn_if_frozen(model)
        t0 = time.time(); tot = cor = n = 0
        for x, y in tl:
            x, y = x.to(device), y.to(device)
            opt.zero_grad(); o = model(x); loss = crit(o, y); loss.backward(); opt.step()
            tot += loss.item() * len(y); cor += (o.argmax(1) == y).sum().item(); n += len(y)
        if scheduler: scheduler.step()
        vloss, vacc = evaluate(model, vl, device)
        h["train_loss"].append(tot / n); h["train_acc"].append(cor / n)
        h["val_loss"].append(vloss); h["val_acc"].append(vacc); h["epoch_time"].append(time.time() - t0)
        if vacc > best: best, best_ep, best_state = vacc, ep, copy.deepcopy(model.state_dict())
        if verbose:
            print(f"ep {ep+1:2d} train_loss {tot/n:.4f} val_loss {vloss:.4f} val_acc {vacc:.4f} "
                  f"({h['epoch_time'][-1]:.1f}s)")
    model.load_state_dict(best_state)                 # best-on-validation weights
    h.update(best_val_acc=best, best_epoch=best_ep + 1,
             min_val_loss_epoch=int(np.argmin(h["val_loss"])) + 1)
    return h
