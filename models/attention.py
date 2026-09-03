"""
Attention-Weighted Fusion (AI Handoff spec, Section 10).

Given the active branches' feature vectors, each in R^n:
    f_concat = concat(f_1, ..., f_k)                     -> R^(k*n)
    g = softmax(W_g * f_concat + b_g)                    -> R^k, sums to 1
    f_fused = sum_i g_i * f_i                             -> R^n

Implemented generically over k (number of active branches) so the same
module serves:
    - the full model (k = 3: ELA, PRNU, Content)
    - every 2-branch ablation config (k = 2)
Single-branch configs (k = 1) skip fusion entirely — see multistream.py.

The doc specifies the gate as a single linear layer directly from the
concatenated features to k logits (no hidden layer is mentioned), which
is what's implemented here. See Handoff Section 19, "DO NOT INVENT" —
no attention hidden dimension is specified.
"""

import torch
import torch.nn as nn


class AttentionFusion(nn.Module):
    def __init__(self, feature_dim: int, num_branches: int):
        super().__init__()
        assert num_branches >= 2, "Fusion only applies when 2 or more branches are active"
        self.num_branches = num_branches
        self.gate = nn.Linear(feature_dim * num_branches, num_branches)

    def forward(self, features: list[torch.Tensor]):
        """
        features: list of k tensors, each (B, feature_dim), in a fixed
                  consistent order (see MultiStreamModel.BRANCH_ORDER).
        returns:
            f_fused: (B, feature_dim)
            weights: (B, k) — the attention weights [w1, ..., wk], for
                     logging/inspection (e.g. which branch the model is
                     relying on most for a given image).
        """
        assert len(features) == self.num_branches

        f_concat = torch.cat(features, dim=1)              # (B, k*n)
        weights = torch.softmax(self.gate(f_concat), dim=1)  # (B, k)

        stacked = torch.stack(features, dim=1)              # (B, k, n)
        f_fused = (stacked * weights.unsqueeze(-1)).sum(dim=1)  # (B, n)

        return f_fused, weights
