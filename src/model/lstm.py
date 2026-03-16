import torch
import torch.nn as nn

class LSTMForecaster(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(LSTMForecaster, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)
