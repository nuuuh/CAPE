"""
CAPE: Causal Autoregressive Predictor for Epidemics
An autoregressive model for next-token-prediction in epidemic time series.
Inspired by GPT-style language models but adapted for epidemic forecasting.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for token positions"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        return x + self.pe[:, :x.size(1), :]


class TCNBlock(nn.Module):
    """Temporal Convolutional Network block with causal padding"""
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout=0.1):
        super().__init__()
        padding = (kernel_size - 1) * dilation  # Causal padding
        self.conv = nn.Conv1d(in_channels, out_channels, kernel_size, 
                              padding=padding, dilation=dilation)
        self.crop = padding  # Amount to crop for causality
        self.dropout = nn.Dropout(dropout)
        self.relu = nn.ReLU()
        self.norm = nn.LayerNorm(out_channels)
        
        # Residual connection
        self.downsample = nn.Conv1d(in_channels, out_channels, 1) if in_channels != out_channels else None
    
    def forward(self, x):
        # x: [batch, channels, seq_len]
        out = self.conv(x)
        if self.crop > 0:
            out = out[:, :, :-self.crop]  # Remove future context
        out = self.relu(out)
        out = self.dropout(out)
        
        # Apply layer norm (need to transpose)
        out = out.transpose(1, 2)  # [batch, seq_len, channels]
        out = self.norm(out)
        out = out.transpose(1, 2)  # [batch, channels, seq_len]
        
        # Residual connection
        res = x if self.downsample is None else self.downsample(x)
        return out + res


class CausalLSTM(nn.Module):
    """LSTM with learnable initial hidden states"""
    def __init__(self, input_size, hidden_size, num_layers, dropout=0.1):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, 
                           batch_first=True, dropout=dropout if num_layers > 1 else 0)
        
        # Learnable initial states
        self.h0 = nn.Parameter(torch.randn(num_layers, 1, hidden_size) * 0.01)
        self.c0 = nn.Parameter(torch.randn(num_layers, 1, hidden_size) * 0.01)
    
    def forward(self, x, hidden=None):
        # x: [batch, seq_len, input_size]
        batch_size = x.size(0)
        
        if hidden is None:
            # Expand learnable initial states to batch size
            h0 = self.h0.expand(-1, batch_size, -1).contiguous()
            c0 = self.c0.expand(-1, batch_size, -1).contiguous()
            hidden = (h0, c0)
        
        out, hidden = self.lstm(x, hidden)
        return out, hidden


class TransformerDecoderBlock(nn.Module):
    """GPT-style Transformer Decoder Block with causal self-attention"""
    def __init__(self, hidden_size, num_heads, dropout=0.1, ff_ratio=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        
        # Multi-head self-attention
        self.self_attn = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        
        # Feedforward network
        self.ff = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * ff_ratio),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * ff_ratio, hidden_size),
            nn.Dropout(dropout)
        )
        
        # Layer normalization
        self.norm1 = nn.LayerNorm(hidden_size)
        self.norm2 = nn.LayerNorm(hidden_size)
        
        self.dropout = nn.Dropout(dropout)
    
    def forward(self, x, attn_mask=None):
        """
        Args:
            x: [batch, seq_len, hidden_size]
            attn_mask: Causal attention mask [seq_len, seq_len]
        """
        # Self-attention with residual connection
        attn_out, _ = self.self_attn(x, x, x, attn_mask=attn_mask, need_weights=False)
        x = self.norm1(x + self.dropout(attn_out))
        
        # Feedforward with residual connection
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        
        return x


class CausalTransformer(nn.Module):
    """Decoder-only Transformer with causal masking (GPT-style)"""
    def __init__(self, hidden_size, num_layers, num_heads, dropout=0.1, ff_ratio=4):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # Stack of transformer decoder blocks
        self.layers = nn.ModuleList([
            TransformerDecoderBlock(hidden_size, num_heads, dropout, ff_ratio)
            for _ in range(num_layers)
        ])
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(hidden_size)
        
        # Cache for causal mask
        self.register_buffer('causal_mask', None)
    
    def _get_causal_mask(self, seq_len, device):
        """Generate or retrieve cached causal attention mask"""
        if self.causal_mask is None or self.causal_mask.size(0) < seq_len:
            # Create causal mask: upper triangular matrix of -inf
            mask = torch.triu(torch.ones(seq_len, seq_len, device=device), diagonal=1)
            mask = mask.masked_fill(mask == 1, float('-inf'))
            self.register_buffer('causal_mask', mask)
        return self.causal_mask[:seq_len, :seq_len]
    
    def forward(self, x):
        """
        Args:
            x: [batch, seq_len, hidden_size]
        Returns:
            x: [batch, seq_len, hidden_size]
        """
        seq_len = x.size(1)
        attn_mask = self._get_causal_mask(seq_len, x.device)
        
        # Apply transformer layers
        for layer in self.layers:
            x = layer(x, attn_mask=attn_mask)
        
        # Final normalization
        x = self.final_norm(x)
        
        return x


class CAPE(nn.Module):
    """
    CAPE: Causal Autoregressive Predictor for Epidemics
    
    Architecture choices:
    - 'tcn': Temporal Convolutional Network (parallel, fast, good for patterns)
    - 'lstm': LSTM with learnable init (sequential, good for long-term memory)
    - 'hybrid': TCN + LSTM (best of both worlds)
    - 'transformer': Decoder-only Transformer with causal attention (GPT-style)
    """
    def __init__(self, token_size=4, hidden_size=128, num_layers=4, 
                 num_heads=4, dropout=0.1, max_tokens=512, 
                 architecture='transformer', ff_ratio=4):
        super().__init__()
        self.token_size = token_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.architecture = architecture
        self.hidden = hidden_size  # For compatibility with other models

        print(f"Initializing CAPE model with architecture: {architecture}")
        
        # Token embedding: project raw token values to hidden space
        self.token_embedding = nn.Linear(token_size, hidden_size)
        
        # Positional encoding
        self.pos_encoder = PositionalEncoding(hidden_size, max_len=max_tokens)
        
        # Architecture-specific components
        if architecture == 'tcn':
            # TCN with exponentially increasing dilation
            self.blocks = nn.ModuleList([
                TCNBlock(hidden_size, hidden_size, kernel_size=3, 
                        dilation=2**i, dropout=dropout)
                for i in range(num_layers)
            ])
        elif architecture == 'lstm':
            self.lstm = CausalLSTM(hidden_size, hidden_size, num_layers, dropout)
        elif architecture == 'transformer':
            # Decoder-only transformer (GPT-style)
            self.transformer = CausalTransformer(
                hidden_size, num_layers, num_heads, dropout, ff_ratio
            )
        elif architecture == 'hybrid':
            # First TCN layers for local patterns
            self.tcn_blocks = nn.ModuleList([
                TCNBlock(hidden_size, hidden_size, kernel_size=3, 
                        dilation=2**i, dropout=dropout)
                for i in range(num_layers // 2)
            ])
            # Then LSTM for long-range dependencies
            self.lstm = CausalLSTM(hidden_size, hidden_size, 
                                  num_layers - num_layers // 2, dropout)
        else:
            raise ValueError(f"Unknown architecture: {architecture}")
        
        # Output projection: hidden -> token_size
        self.output_proj = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, token_size)
        )
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights properly"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Conv1d):
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
    
    def forward(self, x, time=None, return_hidden=False):
        """
        Forward pass for next-token prediction
        
        Args:
            x: Input tokens [batch, num_tokens, token_size]
            time: Time information (optional) [batch, num_tokens, token_size]
            return_hidden: Whether to return hidden states
        
        Returns:
            predictions: Predicted next token [batch, num_tokens, token_size]
                        (for training: predict each position's next token)
            hidden (optional): Hidden representations
        """
        batch_size, num_tokens, _ = x.shape
        
        # Embed tokens to hidden space
        x = self.token_embedding(x)  # [batch, num_tokens, hidden_size]
        
        # Add positional encoding
        x = self.pos_encoder(x)
        
        # Apply architecture-specific processing
        if self.architecture == 'tcn':
            # TCN expects [batch, channels, seq_len]
            x = x.transpose(1, 2)  # [batch, hidden_size, num_tokens]
            for block in self.blocks:
                x = block(x)
            x = x.transpose(1, 2)  # [batch, num_tokens, hidden_size]
            
        elif self.architecture == 'lstm':
            x, _ = self.lstm(x)  # [batch, num_tokens, hidden_size]
            
        elif self.architecture == 'transformer':
            # Transformer decoder with causal attention
            x = self.transformer(x)  # [batch, num_tokens, hidden_size]
            
        elif self.architecture == 'hybrid':
            # First apply TCN
            x_tcn = x.transpose(1, 2)  # [batch, hidden_size, num_tokens]
            for block in self.tcn_blocks:
                x_tcn = block(x_tcn)
            x = x_tcn.transpose(1, 2)  # [batch, num_tokens, hidden_size]
            
            # Then apply LSTM
            x, _ = self.lstm(x)
        
        # Project to token space
        predictions = self.output_proj(x)  # [batch, num_tokens, token_size]

        if return_hidden:
            return predictions, x
        return predictions
    
    def predict_next_token(self, context_tokens, num_predictions=1, temperature=1.0):
        """
        Autoregressive generation: predict next tokens one by one
        
        Args:
            context_tokens: Historical tokens [batch, num_context_tokens, token_size]
            num_predictions: Number of future tokens to predict
            temperature: Sampling temperature (not used for deterministic prediction)
        
        Returns:
            predictions: Predicted future tokens [batch, num_predictions, token_size]
        """
        self.eval()
        with torch.no_grad():
            batch_size = context_tokens.size(0)
            predictions = []
            
            # Start with context
            current_sequence = context_tokens
            
            for _ in range(num_predictions):
                # Predict next token from current sequence
                pred = self.forward(current_sequence)  # [batch, seq_len, token_size]
                
                # Take the last prediction (next token)
                next_token = pred[:, -1:, :]  # [batch, 1, token_size]
                predictions.append(next_token)
                
                # Append to sequence for next iteration
                current_sequence = torch.cat([current_sequence, next_token], dim=1)
            
            # Concatenate all predictions
            predictions = torch.cat(predictions, dim=1)  # [batch, num_predictions, token_size]
            
        return predictions
    
    def compute_loss(self, predictions, targets, mask=None):
        """
        Compute autoregressive loss
        
        Args:
            predictions: Model predictions [batch, num_tokens, token_size]
            targets: Ground truth next tokens [batch, num_tokens, token_size]
            mask: Optional mask for valid positions [batch, num_tokens]
        
        Returns:
            loss: MSE loss between predictions and targets
        """
        if mask is not None:
            # Expand mask to match token_size dimension
            mask = mask.unsqueeze(-1).expand_as(predictions)
            loss = F.mse_loss(predictions * mask, targets * mask, reduction='sum')
            loss = loss / (mask.sum() + 1e-8)
        else:
            loss = F.mse_loss(predictions, targets)
        
        return loss


class CAPEForPretraining(nn.Module):
    """Wrapper for CAPE pretraining with representation head"""
    def __init__(self, cape_model, output_dim=None):
        super().__init__()
        self.cape = cape_model
        self.hidden = cape_model.hidden_size
        
        # Optional projection head for representation learning
        if output_dim is not None:
            self.projection_head = nn.Sequential(
                nn.Linear(cape_model.hidden_size, cape_model.hidden_size),
                nn.ReLU(),
                nn.Linear(cape_model.hidden_size, output_dim)
            )
        else:
            self.projection_head = None
    
    def forward(self, x, time=None, return_representation=False):
        predictions, hidden = self.cape(x, time, return_hidden=True)
        
        if return_representation:
            # Return mean-pooled representation
            representation = hidden.mean(dim=1)  # [batch, hidden_size]
            if self.projection_head is not None:
                representation = self.projection_head(representation)
            return predictions, representation
        
        return predictions


# Factory function for easy model creation
def create_cape_model(config):
    """Create CAPE model from config"""
    # Check if using compartmental model
    use_compartmental = getattr(config, 'use_compartmental', False)
    if use_compartmental:
        # Import compartmental model
        from .CAPE_Compartmental import CompartmentalCAPE
        
        # Get patch encoder type from config (transformer, lstm, tcn, hybrid)
        patch_encoder_type = getattr(config, 'patch_encoder_type', 'transformer')
        use_revin = getattr(config, 'use_revin', False)
        revin_affine = getattr(config, 'revin_affine', True)
        print(f"Creating CompartmentalCAPE model with patch_encoder_type={patch_encoder_type}, use_revin={use_revin}...")
        model = CompartmentalCAPE(
            input_size=config.token_size,
            hidden_size=config.hidden_size,
            num_layers=config.layers,
            num_heads=getattr(config, 'num_heads', 8),
            num_embeddings=getattr(config, 'num_embeddings', 6),
            dropout=config.dropout,
            max_tokens=getattr(config, 'max_tokens', 512),
            compartments=getattr(config, 'compartments', None),
            patch_encoder_type=patch_encoder_type,
            ff_ratio=getattr(config, 'ff_ratio', 4),
            use_revin=use_revin,
            revin_affine=revin_affine
        )
    else:
        model = CAPE(
            token_size=config.token_size,
            hidden_size=config.hidden_size,
            num_layers=config.layers,
            dropout=config.dropout,
            architecture=config.architecture  # Can be made configurable
        )
    
    return model
