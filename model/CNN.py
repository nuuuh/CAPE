"""
Temporal Convolutional Network (TCN) / CNN for Time Series Forecasting

References:
- Bai et al., "An Empirical Evaluation of Generic Convolutional and Recurrent Networks for Sequence Modeling" (2018)
- Lea et al., "Temporal Convolutional Networks for Action Segmentation and Detection" (2017)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init


class CausalConv1d(nn.Module):
    """
    Causal 1D convolution - output only depends on current and past inputs.
    Uses left padding to ensure causality.
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation=1):
        super(CausalConv1d, self).__init__()
        self.padding = (kernel_size - 1) * dilation
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size,
                              padding=self.padding, dilation=dilation)
    
    def forward(self, x):
        # x: (batch, channels, seq_len)
        out = self.conv(x)
        # Remove future timesteps (right padding)
        if self.padding > 0:
            out = out[:, :, :-self.padding]
        return out


class ResidualBlock(nn.Module):
    """
    Residual block with dilated causal convolutions.
    
    Structure:
    x -> Conv -> Norm -> ReLU -> Dropout -> Conv -> Norm -> ReLU + x -> out
    """
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.1):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = CausalConv1d(in_channels, out_channels, kernel_size, dilation)
        self.norm1 = nn.BatchNorm1d(out_channels)
        
        self.conv2 = CausalConv1d(out_channels, out_channels, kernel_size, dilation)
        self.norm2 = nn.BatchNorm1d(out_channels)
        
        self.dropout = nn.Dropout(dropout)
        
        # 1x1 convolution for residual connection if channels differ
        self.residual = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else nn.Identity()
    
    def forward(self, x):
        residual = self.residual(x)
        
        out = self.conv1(x)
        out = self.norm1(out)
        out = F.relu(out)
        out = self.dropout(out)
        
        out = self.conv2(out)
        out = self.norm2(out)
        out = F.relu(out + residual)
        
        return out


class TCNBlock(nn.Module):
    """
    Temporal Convolutional Network block with exponentially increasing dilation.
    """
    def __init__(self, in_channels, hidden_channels, kernel_size=3, num_layers=4, dropout=0.1):
        super(TCNBlock, self).__init__()
        
        layers = []
        for i in range(num_layers):
            dilation = 2 ** i
            in_ch = in_channels if i == 0 else hidden_channels
            layers.append(ResidualBlock(in_ch, hidden_channels, kernel_size, dilation, dropout))
        
        self.network = nn.Sequential(*layers)
    
    def forward(self, x):
        return self.network(x)


class CNNModel(nn.Module):
    """
    CNN/TCN Model for Time Series Forecasting.
    
    Uses stacked dilated causal convolutions with residual connections.
    
    Parameters
    ----------
    num_timesteps_input : int
        Number of input timesteps (lookback window)
    num_timesteps_output : int
        Number of output timesteps (forecast horizon)
    num_features : int
        Number of input features
    hidden_size : int
        Number of hidden channels in convolutions
    num_layers : int
        Number of convolutional layers
    kernel_size : int
        Convolution kernel size
    dropout : float
        Dropout probability
    """
    def __init__(self, num_timesteps_input, num_timesteps_output, num_features=1,
                 hidden_size=64, num_layers=4, kernel_size=3, dropout=0.1):
        super(CNNModel, self).__init__()
        
        self.num_timesteps_input = num_timesteps_input
        self.num_timesteps_output = num_timesteps_output
        self.num_features = num_features
        self.hidden_size = hidden_size
        
        # Input projection
        self.input_proj = nn.Conv1d(num_features, hidden_size, 1)
        
        # TCN backbone
        self.tcn = TCNBlock(hidden_size, hidden_size, kernel_size, num_layers, dropout)
        
        # Output projection
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size * num_timesteps_input, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_timesteps_output)
        )
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, time=None, dec_time=None, mask=None):
        """
        Forward pass.
        
        Args:
            x: (batch, seq_len, features) or (batch, seq_len)
        
        Returns:
            output: (batch, horizon)
        """
        if x.dim() == 2:
            x = x.unsqueeze(-1)  # (batch, seq_len, 1)
        
        # Transpose for conv: (batch, features, seq_len)
        x = x.transpose(1, 2)
        
        # Input projection
        x = self.input_proj(x)
        
        # TCN backbone
        x = self.tcn(x)
        
        # Flatten and project to output
        x = x.reshape(x.shape[0], -1)
        x = self.dropout(x)
        output = self.output_proj(x)
        
        return output, time
    
    def initialize(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.data.fill_(0)
            elif isinstance(module, nn.BatchNorm1d):
                module.reset_parameters()


class SimpleCNN(nn.Module):
    """
    Simple CNN for time series forecasting.
    Uses standard (non-causal) convolutions with pooling.
    """
    def __init__(self, num_timesteps_input, num_timesteps_output, num_features=1,
                 hidden_size=64, num_layers=3, kernel_size=3, dropout=0.1):
        super(SimpleCNN, self).__init__()
        
        self.num_timesteps_input = num_timesteps_input
        self.num_timesteps_output = num_timesteps_output
        
        layers = []
        in_channels = num_features
        
        for i in range(num_layers):
            out_channels = hidden_size * (2 ** min(i, 2))  # Increase channels
            layers.extend([
                nn.Conv1d(in_channels, out_channels, kernel_size, padding=kernel_size//2),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(),
                nn.Dropout(dropout)
            ])
            in_channels = out_channels
        
        self.conv_layers = nn.Sequential(*layers)
        
        # Adaptive pooling to fixed size
        self.pool = nn.AdaptiveAvgPool1d(8)
        
        # Fully connected output
        self.fc = nn.Sequential(
            nn.Linear(in_channels * 8, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size, num_timesteps_output)
        )
    
    def forward(self, x, time=None, dec_time=None, mask=None):
        if x.dim() == 2:
            x = x.unsqueeze(-1)
        
        x = x.transpose(1, 2)  # (batch, features, seq_len)
        x = self.conv_layers(x)
        x = self.pool(x)
        x = x.reshape(x.shape[0], -1)
        output = self.fc(x)
        
        return output, time
    
    def initialize(self):
        for module in self.modules():
            if isinstance(module, (nn.Conv1d, nn.Linear)):
                init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.data.fill_(0)


# Aliases
CNN = CNNModel
TCN = CNNModel
