"""
src/models.py
Three recurrent architectures for half-hourly load forecasting.
All models share identical interfaces, hidden capacity, and output heads
to ensure a fair apples-to-apples benchmark.
"""

import torch
import torch.nn as nn


# ─────────────────────────────────────────────────────────────────────────────
# Shared Configuration Defaults
# ─────────────────────────────────────────────────────────────────────────────
HIDDEN_SIZE = 64
NUM_LAYERS  = 2
DROPOUT     = 0.15
NUM_FEATURES = 9   # matches FEATURE_COLS in dataset.py


# ─────────────────────────────────────────────────────────────────────────────
# 1. Vanilla RNN (Elman)
# ─────────────────────────────────────────────────────────────────────────────
class VanillaRNN(nn.Module):
    """
    Multi-layer Elman RNN baseline.
    h_t = tanh(W_x * x_t + W_h * h_{t-1} + b)
    Fast to train; typically struggles with long-range dependencies.
    """
    def __init__(
        self,
        input_size:  int = NUM_FEATURES,
        hidden_size: int = HIDDEN_SIZE,
        num_layers:  int = NUM_LAYERS,
        dropout:     float = DROPOUT,
    ):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            nonlinearity="tanh",
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: [B, L, F]
        out, _ = self.rnn(x)        # out: [B, L, H]
        last    = out[:, -1, :]     # [B, H]
        return self.head(last)      # [B, 1]

    @property
    def name(self): return "VanillaRNN"


# ─────────────────────────────────────────────────────────────────────────────
# 2. GRU
# ─────────────────────────────────────────────────────────────────────────────
class GRUModel(nn.Module):
    """
    Multi-layer Gated Recurrent Unit.
    Reset gate + Update gate alleviate the vanishing gradient issue.
    Efficient and well-suited to half-hourly diurnal periodicity.
    """
    def __init__(
        self,
        input_size:  int = NUM_FEATURES,
        hidden_size: int = HIDDEN_SIZE,
        num_layers:  int = NUM_LAYERS,
        dropout:     float = DROPOUT,
    ):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        last    = out[:, -1, :]
        return self.head(last)

    @property
    def name(self): return "GRU"


# ─────────────────────────────────────────────────────────────────────────────
# 3. LSTM
# ─────────────────────────────────────────────────────────────────────────────
class LSTMModel(nn.Module):
    """
    Multi-layer Long Short-Term Memory.
    Forget, Input, Output gates + Cell state allow learning of long-range
    weekly seasonality patterns beyond a single lookback window.
    """
    def __init__(
        self,
        input_size:  int = NUM_FEATURES,
        hidden_size: int = HIDDEN_SIZE,
        num_layers:  int = NUM_LAYERS,
        dropout:     float = DROPOUT,
    ):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(hidden_size, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last    = out[:, -1, :]
        return self.head(last)

    @property
    def name(self): return "LSTM"


# ─────────────────────────────────────────────────────────────────────────────
# Factory
# ─────────────────────────────────────────────────────────────────────────────
def get_all_models():
    """Returns a list of all three benchmark model instances."""
    return [VanillaRNN(), GRUModel(), LSTMModel()]


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)
