import torch.nn as nn

import torch.nn.init as init
import torch

class GRUModel(nn.Module):
    def __init__(self, num_timesteps_input, num_timesteps_output, num_features, nhid=256, dropout=0.5, use_norm=True):
        super(GRUModel, self).__init__()
        self.num_features = num_features
        self.num_timesteps_input = num_timesteps_input
        self.num_timesteps_output = num_timesteps_output
        self.hidden = nhid
        self.dropout = dropout
        self.use_norm = use_norm

        # GRU layer
        self.gru = nn.GRU(num_features, nhid, batch_first=True)

        # Optional normalization
        if self.use_norm:
            self.norm = nn.LayerNorm(nhid)

        # Dropout layer
        self.dropout_layer = nn.Dropout(dropout)

        # Output layer: project from hidden to horizon predictions
        self.out = nn.Linear(nhid, num_timesteps_output)

    def forward(self, x, time=None, dec_time=None, mask=None):
        gru_out, _ = self.gru(x)

        if self.use_norm:
            gru_out = self.norm(gru_out)

        # Apply dropout
        gru_out = self.dropout_layer(gru_out)

        # Use the last timestep's output to predict the future series
        last_gru_out = gru_out[:, -1, :]

        # Output layer to predict the next steps
        output = self.out(last_gru_out)  # [batch, horizon]
        
        return output, time

    def initialize(self):
        # Initialize GRU weights
        for name, param in self.gru.named_parameters():
            if 'weight_ih' in name:
                init.xavier_uniform_(param.data)
            elif 'weight_hh' in name:
                init.orthogonal_(param.data)
            elif 'bias' in name:
                param.data.fill_(0)
        
        if self.use_norm:
            self.norm.reset_parameters()

        # Initialize the output layer
        init.xavier_uniform_(self.out.weight)
        self.out.bias.data.fill_(0)
