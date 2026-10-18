"""Custom CNN, MLP baseline, ResNet-18 setup, layer table."""
import torch, torch.nn as nn
from torchvision import models


class CNN(nn.Module):
    def __init__(self, n_blocks=2, pool="max", chans=(32, 64, 128, 128), hidden=128, p=0.4):
        super().__init__()
        layers, c_in = [], 1
        for i in range(n_blocks):
            c = chans[i]
            if pool == "stride":                    # no pooling: stride-2 conv downsamples
                layers += [nn.Conv2d(c_in, c, 3, stride=2, padding=1), nn.ReLU()]
            else:
                layers += [nn.Conv2d(c_in, c, 3, padding=1), nn.ReLU(),
                           nn.MaxPool2d(2) if pool == "max" else nn.AvgPool2d(2)]
            c_in = c
        self.features = nn.Sequential(*layers)
        feat = self.features(torch.zeros(1, 1, 28, 28)).numel()
        self.head = nn.Sequential(nn.Flatten(), nn.Linear(feat, hidden), nn.ReLU(),
                                  nn.Dropout(p), nn.Linear(hidden, 10))

    def forward(self, x):
        return self.head(self.features(x))


def mlp(hidden=(512, 64), p=0.4):
    layers, d = [nn.Flatten()], 784
    for h in hidden:
        layers += [nn.Linear(d, h), nn.ReLU(), nn.Dropout(p)]
        d = h
    return nn.Sequential(*layers, nn.Linear(d, 10))


def n_params(m, trainable=True):
    return sum(p.numel() for p in m.parameters() if (p.requires_grad or not trainable))


def layer_table(model, shape=(1, 1, 28, 28)):
    """Hook-based table used to VERIFY the hand-computed one (Task 2b)."""
    rows, hooks = [], []
    for name, m in model.named_modules():
        if len(list(m.children())) == 0:
            def hook(m, i, o, name=name):
                rows.append(dict(layer=name, type=type(m).__name__, input=tuple(i[0].shape[1:]),
                                 output=tuple(o.shape[1:]),
                                 kernel=getattr(m, "kernel_size", ""), stride=getattr(m, "stride", ""),
                                 padding=getattr(m, "padding", ""),
                                 params=sum(p.numel() for p in m.parameters())))
            hooks.append(m.register_forward_hook(hook))
    model.eval()
    with torch.no_grad():
        model(torch.zeros(shape, device=next(model.parameters()).device))
    for h in hooks: h.remove()
    return rows


def resnet18_tl(strategy="feature_extraction", lr=1e-3):
    """Returns (model, optimizer). strategies: feature_extraction | layer4 | differential"""
    m = models.resnet18(weights="IMAGENET1K_V1")
    m.fc = nn.Linear(512, 10)
    if strategy == "feature_extraction":
        for n, p in m.named_parameters(): p.requires_grad = n.startswith("fc")
        opt = torch.optim.Adam(m.fc.parameters(), lr=lr)
    elif strategy == "layer4":
        for n, p in m.named_parameters(): p.requires_grad = n.startswith(("layer4", "fc"))
        opt = torch.optim.Adam([p for p in m.parameters() if p.requires_grad], lr=lr)
    elif strategy == "differential":
        bb = [p for n, p in m.named_parameters() if not n.startswith("fc")]
        opt = torch.optim.Adam([{"params": bb, "lr": 0.1 * lr},
                                {"params": m.fc.parameters(), "lr": lr}])
    else:
        raise ValueError(strategy)
    return m, opt
