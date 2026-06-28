"""Step 4 — dataset + the class-balanced sampler (handles 29:271 imbalance).

We keep every negative and rebalance at the *batch* level: a WeightedRandomSampler
oversamples the rare DETAIL positives so each mini-batch is ~POS_RATIO positive.
This guarantees every batch sees positives without discarding any negatives.
"""
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import config as C


class FeatLabelDataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.as_tensor(X, dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, i):
        return self.X[i], self.y[i]


def balanced_sampler(y, pos_ratio=C.POS_RATIO):
    """Per-sample weights so draws are ~pos_ratio positive / (1-pos_ratio) negative."""
    y = np.asarray(y)
    n_pos = max(int((y == 1).sum()), 1)
    n_neg = max(int((y == 0).sum()), 1)
    w_pos = pos_ratio / n_pos
    w_neg = (1.0 - pos_ratio) / n_neg
    weights = np.where(y == 1, w_pos, w_neg).astype(np.float64)
    return WeightedRandomSampler(weights=torch.as_tensor(weights),
                                 num_samples=len(y), replacement=True)


def make_loader(X, y, batch_size=C.BATCH_SIZE, balanced=True):
    ds = FeatLabelDataset(X, y)
    if balanced:
        return DataLoader(ds, batch_size=batch_size, sampler=balanced_sampler(y))
    return DataLoader(ds, batch_size=batch_size, shuffle=False)
