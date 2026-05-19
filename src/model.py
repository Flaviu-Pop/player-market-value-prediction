# === Model Definition =================================================================================================
import torch
import torch.nn as nn

from config import (HIDDEN_DIMS, DROPOUT_RATES)


class PlayerValueNet(nn.Module):
    """
    Fully connected neural network for player market value regression.

    Architecture:
        Input → [Linear → BatchNorm → ReLU → Dropout] × N → Linear (output)

    The output is log(market_value) — exponentiate after prediction.
    """

    def __init__(self, input_dim: int, hidden_dims: list = HIDDEN_DIMS, dropout_rates: list = DROPOUT_RATES):
        super().__init__()

        layers = []
        prev_dim = input_dim

        for hidden_dim, dropout_rate in zip(hidden_dims, dropout_rates):
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
            ])

            if dropout_rate > 0:
                layers.append(nn.Dropout(dropout_rate))

            prev_dim = hidden_dim

        layers.append(nn.Linear(prev_dim, 1))   # Single output: log(value)

        self.network = nn.Sequential(*layers)
        self._initialize_weights()

    def _initialize_weights(self):
        """He initialization for ReLU networks."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
                nn.init.zeros_(m.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.network(x).squeeze(-1)
