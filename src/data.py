"""Task 1: transforms, splits, loaders, device utility."""
import numpy as np, torch
from torch.utils.data import DataLoader, Subset, random_split
from torchvision import datasets, transforms as T

SEED = 42
ROOT = "data"
FM_MEAN, FM_STD = (0.2860,), (0.3530,)
IMN_MEAN, IMN_STD = [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
CLASSES = ["T-shirt", "Trouser", "Pullover", "Dress", "Coat",
           "Sandal", "Shirt", "Sneaker", "Bag", "Ankle boot"]


def get_device():                                   # Task 1(d)
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def set_seed(seed=SEED):
    import random
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def train_tf(aug=True):                             # Task 1(b)
    ops = []
    if aug:
        ops += [T.RandomHorizontalFlip(),           # garments are left-right symmetric
                T.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1))]
        # NOT used: RandomVerticalFlip / large rotations (destroy class semantics)
    return T.Compose(ops + [T.ToTensor(), T.Normalize(FM_MEAN, FM_STD)])


def eval_tf():
    return T.Compose([T.ToTensor(), T.Normalize(FM_MEAN, FM_STD)])


def split_indices(seed=SEED):                       # Task 1(c): 80/20 with fixed seed
    g = torch.Generator().manual_seed(seed)
    tr, va = random_split(range(60000), [48000, 12000], generator=g)
    return np.array(tr.indices), np.array(va.indices)


def get_loaders(aug=True, bs=128, seed=SEED):
    """Augmentation on the TRAIN split only: two dataset objects, same indices."""
    tr_idx, va_idx = split_indices(seed)
    ds_tr = datasets.FashionMNIST(ROOT, True, download=True, transform=train_tf(aug))
    ds_va = datasets.FashionMNIST(ROOT, True, download=True, transform=eval_tf())
    ds_te = datasets.FashionMNIST(ROOT, False, download=True, transform=eval_tf())
    g = torch.Generator().manual_seed(seed)
    return (DataLoader(Subset(ds_tr, tr_idx.tolist()), bs, shuffle=True, generator=g),
            DataLoader(Subset(ds_va, va_idx.tolist()), 512),
            DataLoader(ds_te, 512))


def stratified(targets, idx, per_class, seed=SEED):
    rng = np.random.RandomState(seed)
    targets = np.asarray(targets)
    out = []
    for c in range(10):
        pool = idx[targets[idx] == c]
        out += rng.choice(pool, per_class, replace=False).tolist()
    return sorted(out)
