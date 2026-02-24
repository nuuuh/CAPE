"""
N-BEATS: Neural Basis Expansion Analysis for Interpretable Time Series Forecasting

Reference: 
- Oreshkin et al., "N-BEATS: Neural basis expansion analysis for interpretable time series forecasting" (ICLR 2020)
"""

import torch
import torch.nn as nn
import torch.nn.init as init
import numpy as np


class NBeatsBlock(nn.Module):
    """
    Basic building block of N-BEATS.
    Each block consists of a fully connected stack followed by basis expansion.
    """
    def __init__(self, input_size, theta_size, hidden_size, num_layers, basis_function):
        super(NBeatsBlock, self).__init__()
        self.input_size = input_size
        self.theta_size = theta_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.basis_function = basis_function
        
        # Fully connected layers
        layers = []
        layers.append(nn.Linear(input_size, hidden_size))
        layers.append(nn.ReLU())
        
        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_size, hidden_size))
            layers.append(nn.ReLU())
        
        self.fc = nn.Sequential(*layers)
        
        # Theta layers for backcast and forecast
        self.theta_b_fc = nn.Linear(hidden_size, theta_size, bias=False)
        self.theta_f_fc = nn.Linear(hidden_size, theta_size, bias=False)
    
    def forward(self, x):
        # x: (batch, input_size)
        h = self.fc(x)
        theta_b = self.theta_b_fc(h)
        theta_f = self.theta_f_fc(h)
        
        backcast = self.basis_function.backcast(theta_b)
        forecast = self.basis_function.forecast(theta_f)
        
        return backcast, forecast


class GenericBasis(nn.Module):
    """Generic basis function - learnable parameters."""
    def __init__(self, backcast_size, forecast_size, theta_size):
        super(GenericBasis, self).__init__()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size
        self.theta_size = theta_size
        
        self.backcast_fc = nn.Linear(theta_size, backcast_size)
        self.forecast_fc = nn.Linear(theta_size, forecast_size)
    
    def backcast(self, theta):
        return self.backcast_fc(theta)
    
    def forecast(self, theta):
        return self.forecast_fc(theta)


class TrendBasis(nn.Module):
    """Trend basis using polynomial expansion."""
    def __init__(self, backcast_size, forecast_size, degree=3):
        super(TrendBasis, self).__init__()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size
        self.degree = degree
        
        # Pre-compute polynomial basis
        backcast_time = torch.arange(backcast_size).float() / backcast_size
        forecast_time = torch.arange(forecast_size).float() / forecast_size
        
        self.register_buffer('backcast_basis', 
            torch.stack([backcast_time ** i for i in range(degree + 1)], dim=0))
        self.register_buffer('forecast_basis',
            torch.stack([forecast_time ** i for i in range(degree + 1)], dim=0))
    
    def backcast(self, theta):
        # theta: (batch, degree+1)
        return torch.einsum('bp,pt->bt', theta, self.backcast_basis)
    
    def forecast(self, theta):
        return torch.einsum('bp,pt->bt', theta, self.forecast_basis)


class SeasonalityBasis(nn.Module):
    """Seasonality basis using Fourier expansion."""
    def __init__(self, backcast_size, forecast_size, num_harmonics=4):
        super(SeasonalityBasis, self).__init__()
        self.backcast_size = backcast_size
        self.forecast_size = forecast_size
        self.num_harmonics = num_harmonics
        
        # Pre-compute Fourier basis
        backcast_time = torch.arange(backcast_size).float() / backcast_size * 2 * np.pi
        forecast_time = torch.arange(forecast_size).float() / forecast_size * 2 * np.pi
        
        backcast_basis = []
        forecast_basis = []
        for k in range(1, num_harmonics + 1):
            backcast_basis.append(torch.cos(k * backcast_time))
            backcast_basis.append(torch.sin(k * backcast_time))
            forecast_basis.append(torch.cos(k * forecast_time))
            forecast_basis.append(torch.sin(k * forecast_time))
        
        self.register_buffer('backcast_basis', torch.stack(backcast_basis, dim=0))
        self.register_buffer('forecast_basis', torch.stack(forecast_basis, dim=0))
    
    def backcast(self, theta):
        return torch.einsum('bp,pt->bt', theta, self.backcast_basis)
    
    def forecast(self, theta):
        return torch.einsum('bp,pt->bt', theta, self.forecast_basis)


class NBeatsStack(nn.Module):
    """Stack of N-BEATS blocks with shared basis type."""
    def __init__(self, input_size, forecast_size, hidden_size, num_layers, 
                 num_blocks, basis_type='generic', share_weights=False):
        super(NBeatsStack, self).__init__()
        self.input_size = input_size
        self.forecast_size = forecast_size
        self.num_blocks = num_blocks
        self.share_weights = share_weights
        
        # Determine theta size based on basis type
        if basis_type == 'trend':
            degree = 3
            theta_size = degree + 1
            basis_fn = TrendBasis(input_size, forecast_size, degree)
        elif basis_type == 'seasonality':
            num_harmonics = 4
            theta_size = 2 * num_harmonics
            basis_fn = SeasonalityBasis(input_size, forecast_size, num_harmonics)
        else:  # generic
            theta_size = hidden_size
            basis_fn = GenericBasis(input_size, forecast_size, theta_size)
        
        self.basis_function = basis_fn
        
        if share_weights:
            self.block = NBeatsBlock(input_size, theta_size, hidden_size, num_layers, basis_fn)
            self.blocks = None
        else:
            self.blocks = nn.ModuleList([
                NBeatsBlock(input_size, theta_size, hidden_size, num_layers, basis_fn)
                for _ in range(num_blocks)
            ])
    
    def forward(self, x):
        # x: (batch, input_size)
        residual = x
        forecast = 0
        
        for i in range(self.num_blocks):
            block = self.block if self.share_weights else self.blocks[i]
            backcast, block_forecast = block(residual)
            residual = residual - backcast
            forecast = forecast + block_forecast
        
        return residual, forecast


class NBeatsModel(nn.Module):
    """
    N-BEATS Model for Time Series Forecasting.
    
    Parameters
    ----------
    num_timesteps_input : int
        Number of input timesteps (lookback window)
    num_timesteps_output : int
        Number of output timesteps (forecast horizon)
    num_features : int
        Number of features per timestep (will be flattened)
    hidden_size : int
        Hidden layer size in FC layers
    num_layers : int
        Number of FC layers per block
    num_stacks : int
        Number of stacks (default: 2 for generic, 3 for interpretable)
    num_blocks : int
        Number of blocks per stack
    interpretable : bool
        If True, use trend + seasonality stacks; else use generic stacks
    dropout : float
        Dropout rate (applied after FC layers)
    """
    def __init__(self, num_timesteps_input, num_timesteps_output, num_features=1,
                 hidden_size=256, num_layers=4, num_stacks=2, num_blocks=3,
                 interpretable=False, dropout=0.1):
        super(NBeatsModel, self).__init__()
        
        self.num_timesteps_input = num_timesteps_input
        self.num_timesteps_output = num_timesteps_output
        self.num_features = num_features
        self.hidden_size = hidden_size
        
        # Flatten input
        input_size = num_timesteps_input * num_features
        
        if interpretable:
            # Interpretable architecture: trend + seasonality stacks
            self.stacks = nn.ModuleList([
                NBeatsStack(input_size, num_timesteps_output, hidden_size, num_layers, 
                           num_blocks, basis_type='trend'),
                NBeatsStack(input_size, num_timesteps_output, hidden_size, num_layers,
                           num_blocks, basis_type='seasonality'),
            ])
        else:
            # Generic architecture
            self.stacks = nn.ModuleList([
                NBeatsStack(input_size, num_timesteps_output, hidden_size, num_layers,
                           num_blocks, basis_type='generic')
                for _ in range(num_stacks)
            ])
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, time=None, dec_time=None, mask=None):
        """
        Forward pass.
        
        Args:
            x: (batch, num_timesteps_input, num_features) or (batch, num_timesteps_input)
        
        Returns:
            forecast: (batch, num_timesteps_output)
        """
        # Flatten input
        if x.dim() == 3:
            batch_size = x.shape[0]
            x = x.reshape(batch_size, -1)
        elif x.dim() == 2:
            batch_size = x.shape[0]
        
        x = self.dropout(x)
        
        residual = x
        forecast = 0
        
        for stack in self.stacks:
            residual, stack_forecast = stack(residual)
            forecast = forecast + stack_forecast
        
        return forecast, time
    
    def initialize(self):
        """Initialize model weights."""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                init.xavier_uniform_(module.weight)
                if module.bias is not None:
                    module.bias.data.fill_(0)


# Alias for compatibility
NBEATS = NBeatsModel
