"""Jointly read all household histories and predict each household's next demand."""

import torch
from torch import nn


class HouseholdGRU(nn.Module):
    def __init__(self, households, width=64):
        super().__init__()
        # Each step contains household demands followed by observed/missing flags.
        self.history_gru = nn.GRU(2 * households, width, batch_first=True)
        # One upcoming price is known for the shared LCL ToU schedule.
        self.next_demand = nn.Linear(width + 1, households)

    def forward(self, history, next_price):
        _, hidden = self.history_gru(history)
        context = torch.cat((hidden[-1], next_price.reshape(-1, 1)), dim=1)
        return torch.nn.functional.softplus(self.next_demand(context))
