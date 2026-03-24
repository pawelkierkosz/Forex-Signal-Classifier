import torch
import torch.nn as nn
import sys
import os

class LSTMRegressor(nn.Module):
    def __init__(self, input_dim, hidden_size, num_layers, dropout_p=0.3, bidirectional=False):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout_p if num_layers > 1 else 0.0,
            bidirectional=bidirectional
        )
        out_dim = hidden_size * (2 if bidirectional else 1)
        self.fc = nn.Linear(out_dim, 1)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        # Bierzemy ostatni stan z sekwencji
        last = lstm_out[:, -1, :]
        out = self.fc(last)
        return out.squeeze(1)