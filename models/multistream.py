"""
MultiStreamModel — one configurable model class covering all 7 rows of
the assigned task table:

    Table row                          -> active_branches
    ----------------------------------------------------
    Single-stream PRNU CNN             -> ["prnu"]
    Single-stream ELA CNN              -> ["ela"]
    Single-Stream Content CNN          -> ["content"]
    Proposed Multistream model         -> ["ela", "prnu", "content"]
    ELA + Content Model with Fusion    -> ["ela", "content"]   (= "no_prnu" ablation)
    PRNU + Content Model with Fusion   -> ["prnu", "content"]  (= "no_ela" ablation)
    PRNU + ELA Model with Fusion       -> ["prnu", "ela"]      (= "no_content" ablation)

Rather than 7 separate model files, each config file in config/*.yaml
selects `active_branches` and this class builds the right thing:
    - 1 active branch  -> branch feeds the classifier directly, no fusion
    - 2 or 3 branches  -> AttentionFusion combines them before the classifier

See ela_branch.py, prnu_branch.py, content_branch.py, attention.py, and
classifier.py for what each piece does and which hyperparameters are
implementation defaults vs. documented requirements.

NOTE on the "prnu" input key: it is the RAW (minimally-preprocessed)
image tensor, not a precomputed residual. PRNUBranch internally runs
the learnable Hybrid Wavelet Layer (wavelet_layer.py) to derive the
residual W = X - D on every forward pass, so gradients from the
classification loss can flow back into the per-subband thresholds
(Handoff Section 4 / Section 13). See prnu_branch.py's docstring.
"""

from typing import Dict, Iterable, Optional

import torch
import torch.nn as nn

from .ela_branch import ELABranch
from .prnu_branch import PRNUBranch
from .content_branch import ContentBranch
from .attention import AttentionFusion
from .classifier import Classifier


class MultiStreamModel(nn.Module):
    # Fixed canonical order so concatenation/stacking order is
    # reproducible regardless of the order branches are listed in config.
    BRANCH_ORDER = ["ela", "prnu", "content"]

    _BRANCH_BUILDERS = {
        "ela": ELABranch,
        "prnu": PRNUBranch,
        "content": ContentBranch,
    }

    def __init__(
        self,
        active_branches: Iterable[str],
        feature_dim: int = 128,
        branch_kwargs: Optional[Dict[str, dict]] = None,
        classifier_hidden_dim: int = 64,
        dropout: float = 0.5,
        num_classes: int = 2,
    ):
        super().__init__()

        active_set = set(active_branches)
        if not active_set:
            raise ValueError("active_branches must contain at least one of "
                              f"{self.BRANCH_ORDER}")
        if not active_set.issubset(self.BRANCH_ORDER):
            raise ValueError(f"Unknown branch(es) in {active_set}; "
                              f"must be a subset of {self.BRANCH_ORDER}")

        self.active_branches = [b for b in self.BRANCH_ORDER if b in active_set]
        branch_kwargs = branch_kwargs or {}

        self.branches = nn.ModuleDict({
            name: self._BRANCH_BUILDERS[name](
                feature_dim=feature_dim, **branch_kwargs.get(name, {})
            )
            for name in self.active_branches
        })

        self.use_fusion = len(self.active_branches) > 1
        if self.use_fusion:
            self.fusion = AttentionFusion(feature_dim, len(self.active_branches))

        self.classifier = Classifier(
            feature_dim=feature_dim,
            hidden_dim=classifier_hidden_dim,
            dropout=dropout,
            num_classes=num_classes,
        )

    def forward(self, inputs: Dict[str, torch.Tensor]):
        """
        inputs: dict mapping branch name -> its preprocessed input tensor.
                Only keys matching self.active_branches are required, e.g.
                for a PRNU-only model: {"prnu": prnu_residual_tensor}
                for the full model: {"ela": ..., "prnu": ..., "content": ...}

        returns:
            logits: (B, num_classes)
            attn_weights: (B, k) if fusion was used, else None
                (k = number of active branches; None for single-branch configs)
        """
        features = [self.branches[name](inputs[name]) for name in self.active_branches]

        if self.use_fusion:
            f_fused, attn_weights = self.fusion(features)
        else:
            f_fused, attn_weights = features[0], None

        logits = self.classifier(f_fused)
        return logits, attn_weights

    @classmethod
    def from_config(cls, config: dict) -> "MultiStreamModel":
        """Build a model from a loaded config/*.yaml dict's `model` section."""
        m = config["model"]
        return cls(
            active_branches=m["active_branches"],
            feature_dim=m.get("feature_dim", 128),
            branch_kwargs=config.get("branch_overrides"),
            classifier_hidden_dim=m.get("classifier_hidden_dim", 64),
            dropout=m.get("dropout", 0.5),
            num_classes=m.get("num_classes", 2),
        )
