"""Step 5 — the router head.

Tiny by design (29 positives): a 1-hidden-layer MLP with dropout. A linear
logistic-regression head is available via hidden=0 as the simplest-router anchor.
Output is a single logit -> sigmoid -> P(needs fovea / DETAIL).
"""
import torch
import torch.nn as nn
import config as C


class RouterMLP(nn.Module):
    def __init__(self, in_dim, hidden=C.HIDDEN, dropout=C.DROPOUT):
        super().__init__()
        if hidden and hidden > 0:
            self.net = nn.Sequential(
                nn.Linear(in_dim, hidden),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(hidden, 1),
            )
        else:  # logistic regression
            self.net = nn.Linear(in_dim, 1)

    def forward(self, x):
        return self.net(x).squeeze(-1)   # [B] logits
