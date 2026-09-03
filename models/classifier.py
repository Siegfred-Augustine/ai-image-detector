"""
Classifier head (AI Handoff spec, Section 5/11).

    FC -> ReLU -> Dropout -> FC -> FC -> Softmax
    Output: [P(real), P(AI-generated)]

Returns raw logits from forward(); apply softmax outside the model (or
let nn.CrossEntropyLoss apply it internally during training) rather
than baking softmax into the module — this is standard PyTorch
practice and does not change the documented architecture.

FC layer widths and dropout probability are NOT specified in the
research doc. See Handoff Section 19, "DO NOT INVENT".
"""

import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(
        self,
        feature_dim: int = 128,
        hidden_dim: int = 64,   # NOT specified in research doc — implementation default
        dropout: float = 0.5,   # NOT specified in research doc — implementation default
        num_classes: int = 2,
    ):
        super().__init__()
        self.fc1 = nn.Linear(feature_dim, hidden_dim)
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(dropout)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: f_fused (or the single branch's feature vector), shape (B, feature_dim)
        returns: logits, shape (B, num_classes) = (B, 2) for [real, AI-generated]
        """
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.fc2(x)
        logits = self.fc3(x)
        return logits
