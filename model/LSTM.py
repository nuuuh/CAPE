import torch
from torch import nn

class LSTMModel(nn.Module):
    def __init__(self, num_timesteps_input, num_timesteps_output, num_features, nhid=256, dropout=0.5, use_norm=True):
        super(LSTMModel, self).__init__()
        self.num_features = num_features
        self.num_timesteps_input = num_timesteps_input
        self.num_timesteps_output = num_timesteps_output
        self.hidden = nhid
        self.dropout = dropout
        self.use_norm = use_norm

        # LSTM layer
        self.lstm = nn.LSTM(num_features, nhid, batch_first=True)

        # Optional normalization
        if self.use_norm:
            self.norm = nn.LayerNorm(nhid)

        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)

        # Output layer: project from hidden to horizon predictions
        self.out = nn.Linear(nhid, num_timesteps_output)

    def forward(self, x, time=None, dec_time=None, mask=None):
        lstm_out, _ = self.lstm(x)

        if self.use_norm:
            lstm_out = self.norm(lstm_out)

        # Apply dropout
        lstm_out = self.dropout_layer(lstm_out)

        # Use the last timestep's output to predict the future series
        last_lstm_out = lstm_out[:, -1, :]

        # Output layer to predict the next steps
        output = self.out(last_lstm_out)  # [batch, horizon]
        
        return output, time