"""
CAPE with Compartmental Epidemic Modeling
Sophisticated pretraining with learnable epidemic compartments (S, E, I, R, etc.)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
import numpy as np
from typing import Dict, List, Optional, Tuple, Callable
from scipy.ndimage import uniform_filter1d

# Import RevIN for optional reversible instance normalization
from model.layers.revin import RevIN

# Import ensemble strategies
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from src.ensemble_strategies import get_ensemble_strategy, ENSEMBLE_STRATEGIES
from scipy.ndimage import uniform_filter1d


# =============================================================================
# PREPROCESSING AND ENSEMBLE WRAPPERS
# Use imported strategies from cmp_foundation_models modules
# =============================================================================

def get_preprocessing_fn(name: str = 'smooth_light', **kwargs) -> Callable:
    """Get a preprocessing function by name.
    
    Args:
        name: Preprocessing strategy name ('none' or 'smooth_light')
        **kwargs: Additional arguments passed to smooth_light
    
    Returns:
        Function that takes tokens array and returns preprocessed array
    """
    if name == 'none':
        return lambda tokens: tokens
    if name == 'smooth_light':
        def _smooth(tokens, size=kwargs.get('size', 3), mode=kwargs.get('mode', 'nearest')):
            values = tokens.flatten()
            smoothed = uniform_filter1d(values, size=size, mode=mode)
            return smoothed.reshape(tokens.shape)
        return _smooth
    raise ValueError(f"Unknown preprocessing: {name}. Available: ['none', 'smooth_light']")


def preprocess_smooth_light(tokens: np.ndarray) -> np.ndarray:
    """Apply light smoothing to tokens. Default preprocessing for CAPE."""
    values = tokens.flatten()
    smoothed = uniform_filter1d(values, size=3, mode='nearest')
    return smoothed.reshape(tokens.shape)


# Default ensemble: max (best strategy based on analysis)
def ensemble_max(predictions: torch.Tensor) -> torch.Tensor:
    """Max ensemble - best overall strategy."""
    return get_ensemble_strategy('max')(predictions)


def ensemble_adaptive_percentile(predictions: torch.Tensor) -> torch.Tensor:
    """Adaptive percentile ensemble - legacy default."""
    return get_ensemble_strategy('adaptive_percentile')(predictions)


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


class PositionalEncoding(nn.Module):
    """Sinusoidal positional encoding for token positions"""
    def __init__(self, d_model, max_len=5000):
        super().__init__()
        self.d_model = d_model
        self.max_len = max_len
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)  # [1, max_len, d_model]
        self.register_buffer('pe', pe)

    def forward(self, x):
        # x: [batch, seq_len, d_model]
        seq_len = x.size(1)
        
        # If sequence is longer than max_len, dynamically generate positional encodings
        if seq_len > self.max_len:
            pe = torch.zeros(seq_len, self.d_model, device=x.device)
            position = torch.arange(0, seq_len, dtype=torch.float, device=x.device).unsqueeze(1)
            div_term = torch.exp(torch.arange(0, self.d_model, 2, device=x.device).float() * (-math.log(10000.0) / self.d_model))
            pe[:, 0::2] = torch.sin(position * div_term)
            pe[:, 1::2] = torch.cos(position * div_term)
            return x + pe.unsqueeze(0)
        
        return x + self.pe[:, :seq_len, :]


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


class CompartmentalCAPE(nn.Module):
    """
    CAPE with Compartmental Epidemic Modeling
    
    Key features:
    1. Learnable epidemic compartments (12 total: S, I, E, R, H, V, Q, D, P, W, A, C)
    2. Infectious (I) is the observed input time series
    3. Each compartment mapped to 6 embeddings via shared weights
    4. Self-attention on embeddings for next-token prediction
    5. Output dict with predictions for each compartment
    6. Loss weighting: I=1.0 (observed), others=0.5 by default
    """
    
    # Standard epidemic compartments (extended set)
    # S and I are required; others are optional
    COMPARTMENTS = ['S', 'I', 'E', 'R', 'H', 'V', 'Q', 'D', 'P', 'W', 'A', 'C']
    # S: Susceptible, I: Infectious (observed), E: Exposed, R: Recovered
    # H: Hospitalized, V: Vaccinated, Q: Quarantined, D: Deceased
    # P: Protected, W: Environmental reservoir, A: Asymptomatic, C: Chronic carriers
    
    def __init__(self, 
                 input_size=1,          # Size of input time series (I compartment)
                 hidden_size=128,        # Hidden dimension
                 num_layers=4,           # Number of transformer layers
                 num_heads=8,            # Number of attention heads
                 num_embeddings=6,       # Number of embeddings per compartment
                 dropout=0.1,
                 max_tokens=512,
                 compartments=None,      # List of compartments to use, default all
                 patch_encoder_type='transformer',  # 'transformer', 'tcn', 'lstm', 'hybrid'
                 ff_ratio=4,
                 use_revin=False,        # Whether to use RevIN (Reversible Instance Normalization)
                 revin_affine=True):     # Whether RevIN has learnable affine parameters
        super().__init__()
        
        # Configuration
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.num_embeddings = num_embeddings
        self.compartments = compartments if compartments is not None else self.COMPARTMENTS
        self.num_compartments = len(self.compartments)
        self.patch_encoder_type = patch_encoder_type
        self.use_revin = use_revin
        
        # RevIN layer (optional) - normalizes input, denormalizes output
        if use_revin:
            self.revin = RevIN(num_features=input_size, affine=revin_affine)
            print(f"Using RevIN with affine={revin_affine}")
        
        print(f"Initializing CompartmentalCAPE with compartments: {self.compartments}")
        print(f"Input size: {input_size}, Hidden size: {hidden_size}")
        print(f"Num embeddings per compartment: {num_embeddings}")
        print(f"Patch encoder type: {patch_encoder_type}")
        print(f"RevIN: {'enabled' if use_revin else 'disabled'}")
        
        # ==================== PART 1: ENCODER (FLEXIBLE) ====================
        # Initial embedding layer
        self.input_embedding = nn.Linear(input_size, hidden_size)
        
        # Positional encoding for patches
        self.patch_pos_encoder = PositionalEncoding(hidden_size, max_len=max_tokens)
        
        # Architecture-specific patch encoder (all are causal/autoregressive)
        if patch_encoder_type == 'transformer':
            # Decoder-only transformer (GPT-style, causal)
            self.patch_encoder = CausalTransformer(
                hidden_size, num_layers, num_heads, dropout, ff_ratio
            )
        elif patch_encoder_type == 'tcn':
            # TCN with causal padding
            self.patch_encoder_blocks = nn.ModuleList([
                TCNBlock(hidden_size, hidden_size, kernel_size=3, 
                        dilation=2**i, dropout=dropout)
                for i in range(num_layers)
            ])
        elif patch_encoder_type == 'lstm':
            # LSTM (inherently causal)
            self.patch_encoder = CausalLSTM(hidden_size, hidden_size, num_layers, dropout)
        elif patch_encoder_type == 'hybrid':
            # TCN + LSTM (both causal)
            self.patch_encoder_tcn = nn.ModuleList([
                TCNBlock(hidden_size, hidden_size, kernel_size=3, 
                        dilation=2**i, dropout=dropout)
                for i in range(num_layers // 2)
            ])
            self.patch_encoder_lstm = CausalLSTM(hidden_size, hidden_size, 
                                                 num_layers - num_layers // 2, dropout)
        else:
            raise ValueError(f"Unknown patch_encoder_type: {patch_encoder_type}")
        
        # ==================== PART 2: COMPARTMENTAL MODELING ====================
        # Learnable compartment type embeddings (used as queries for cross-attention)
        self.compartment_type_embedding = nn.Embedding(self.num_compartments, hidden_size)
        
        # Cross-attention: compartment embeddings attend to historical patch embeddings
        self.cross_attention = nn.MultiheadAttention(
            hidden_size, num_heads, dropout=dropout, batch_first=True
        )
        self.cross_attn_norm = nn.LayerNorm(hidden_size)
        
        # Feedforward after cross-attention
        self.cross_attn_ff = nn.Sequential(
            nn.Linear(hidden_size, hidden_size * 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_size * 2, hidden_size)
        )
        self.cross_attn_ff_norm = nn.LayerNorm(hidden_size)
        
        # Multi-view projection: map each compartment to multiple views
        self.multi_view_projection = nn.Linear(hidden_size, hidden_size * num_embeddings)
        
        # Positional encoding for multi-view embeddings
        self.pos_encoder = PositionalEncoding(hidden_size, max_len=max_tokens * num_embeddings * len(self.compartments))
        
        # Self-attention transformer on multi-view embeddings
        self.transformer = CausalTransformer(
            hidden_size, num_layers, num_heads, dropout, ff_ratio
        )
        
        # Output projections for each compartment
        # Each compartment gets its own prediction head
        self.output_heads = nn.ModuleDict({
            comp: nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 2),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(hidden_size * 2, input_size)
            ) for comp in self.compartments
        })
        
        # R(t) prediction head - predicts reproduction number from attention patterns
        # self.R_t_head = nn.Sequential(
        #     nn.Linear(input_size, hidden_size),
        #     nn.GELU(),
        #     nn.Dropout(dropout),
        #     nn.Linear(hidden_size, input_size),  # Predict R(t) for each time point
        #     nn.Softplus()  # Ensure positive R(t)
        # )

        self.R_t_head = nn.Sequential(
            nn.Linear(input_size, input_size),  # Predict R(t) for each time point
            nn.Softplus()  # Ensure positive R(t)
        )
        
        # ==================== LEARNABLE ENSEMBLE PARAMETERS ====================
        # Two-parameter learnable ensemble (Solution 4)
        # percentile_logit: sigmoid -> percentile in [0, 1], initialized at 0.7 (~70th percentile)
        # spread_scale_logit: sigmoid * 0.5 -> spread factor in [0, 0.5], initialized at 0
        self.percentile_logit = nn.Parameter(torch.tensor(0.85))  # sigmoid(0.85) ≈ 0.7
        self.spread_scale_logit = nn.Parameter(torch.tensor(0.0))  # sigmoid(0) = 0.5 -> 0.25 spread
        
        self._init_weights()
    
    def _init_weights(self):
        """Initialize weights properly"""
        for module in self.modules():
            if isinstance(module, nn.Linear):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
                if module.bias is not None:
                    nn.init.zeros_(module.bias)
            elif isinstance(module, nn.Embedding):
                nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def ensemble_learnable(self, predictions: torch.Tensor) -> torch.Tensor:
        """Learnable two-parameter ensemble strategy.
        
        Combines a learnable percentile with spread scaling:
        - percentile: controls base selection point (0=min, 0.5=median, 1=max)
        - spread_factor: adds fraction of (max - median) spread
        
        This can learn strategies like:
        - percentile=1.0, spread=0 → max
        - percentile=0.5, spread=0 → median  
        - percentile=1.0, spread=0.25 → max_plus_std (approximately)
        - percentile=0.9, spread=0.2 → beyond_max_20 style
        
        Args:
            predictions: Tensor of shape (num_masks, batch, seq, dim)
            
        Returns:
            Ensembled predictions of shape (batch, seq, dim)
        """
        n_masks = predictions.shape[0]
        
        # Get percentile (0 to 1)
        percentile = torch.sigmoid(self.percentile_logit)
        # Get spread factor (0 to 0.5)
        spread_factor = torch.sigmoid(self.spread_scale_logit) * 0.5
        
        # Sort predictions along mask dimension
        sorted_preds, _ = predictions.sort(dim=0)
        
        # Differentiable percentile interpolation
        idx_float = percentile * (n_masks - 1)
        idx_low = idx_float.floor().long().clamp(0, n_masks - 1)
        idx_high = (idx_low + 1).clamp(0, n_masks - 1)
        frac = idx_float - idx_low.float()
        
        # Interpolate between adjacent indices
        base = (1 - frac) * sorted_preds[idx_low] + frac * sorted_preds[idx_high]
        
        # Compute spread: max - median
        median_idx = n_masks // 2
        spread = sorted_preds[-1] - sorted_preds[median_idx]
        
        # Final prediction: base + spread_factor * spread
        return base + spread_factor * spread
    
    def get_ensemble_params(self) -> dict:
        """Get current learnable ensemble parameters (for logging)."""
        percentile = torch.sigmoid(self.percentile_logit).item()
        spread_factor = (torch.sigmoid(self.spread_scale_logit) * 0.5).item()
        return {
            'percentile': percentile,
            'spread_factor': spread_factor,
            'percentile_logit': self.percentile_logit.item(),
            'spread_scale_logit': self.spread_scale_logit.item()
        }

    def encode_patches(self, I_timeseries: torch.Tensor, apply_preprocessing: bool = False) -> torch.Tensor:
        """
        PART 1: Encode input patches into embeddings (CAUSAL/AUTOREGRESSIVE)
        
        Args:
            I_timeseries: Observed infectious compartment [batch, seq_len, input_size]
            apply_preprocessing: If True, apply smooth_light preprocessing to input
        
        Returns:
            patch_embeddings: [batch, seq_len, hidden_size]
        """
        # Apply preprocessing if requested (smooth_light: MA-3 smoothing)
        if apply_preprocessing:
            # Convert to numpy, apply smoothing, convert back
            device = I_timeseries.device
            dtype = I_timeseries.dtype
            x_np = I_timeseries.cpu().numpy()
            # Apply per-sample smoothing
            batch_size = x_np.shape[0]
            smoothed = np.stack([preprocess_smooth_light(x_np[i]) for i in range(batch_size)])
            I_timeseries = torch.from_numpy(smoothed).to(device=device, dtype=dtype)
        
        # Initial embedding
        x = self.input_embedding(I_timeseries)  # [batch, seq_len, hidden_size]
        
        # Add positional encoding
        x = self.patch_pos_encoder(x)
        
        # Apply architecture-specific encoder (all maintain causality)
        if self.patch_encoder_type == 'transformer':
            patch_embeddings = self.patch_encoder(x)  # Causal transformer
            
        elif self.patch_encoder_type == 'tcn':
            # TCN expects [batch, channels, seq_len]
            x = x.transpose(1, 2)  # [batch, hidden_size, seq_len]
            for block in self.patch_encoder_blocks:
                x = block(x)  # Causal convolution
            patch_embeddings = x.transpose(1, 2)  # [batch, seq_len, hidden_size]
            
        elif self.patch_encoder_type == 'lstm':
            patch_embeddings, _ = self.patch_encoder(x)  # Inherently causal
            
        elif self.patch_encoder_type == 'hybrid':
            # First apply TCN (causal)
            x_tcn = x.transpose(1, 2)  # [batch, hidden_size, seq_len]
            for block in self.patch_encoder_tcn:
                x_tcn = block(x_tcn)
            x = x_tcn.transpose(1, 2)  # [batch, seq_len, hidden_size]
            
            # Then apply LSTM (causal)
            patch_embeddings, _ = self.patch_encoder_lstm(x)
        
        return patch_embeddings
    
    def create_compartment_embeddings_with_cross_attention(
        self, 
        patch_embeddings: torch.Tensor,
        batch_size: int,
        seq_len: int,
        compartment_mask: torch.Tensor
    ) -> torch.Tensor:
        """
        Create compartment embeddings via cross-attention with causal masking.
        
        Args:
            patch_embeddings: [batch, seq_len, hidden_size]
            batch_size: Batch size
            seq_len: Sequence length
            compartment_mask: [batch, num_compartments] boolean mask for active compartments
        
        Returns:
            comp_embeddings: [batch, seq_len, num_compartments, hidden_size]
        """
        # Get all learnable compartment type embeddings as queries
        # [num_compartments, hidden_size]
        comp_type_embs = self.compartment_type_embedding.weight
        comp_queries = comp_type_embs.unsqueeze(0).unsqueeze(0).expand(batch_size, seq_len, -1, -1)
        
        # Reshape for batched cross-attention
        # Queries: each position's compartments [batch*seq_len, num_compartments, hidden_size]
        comp_queries_flat = comp_queries.reshape(batch_size * seq_len, self.num_compartments, self.hidden_size)
        patch_kv = patch_embeddings.unsqueeze(1).expand(-1, seq_len, -1, -1)
        
        # Reshape to [batch*seq_len, seq_len, hidden_size] for batched attention
        patch_kv_flat = patch_kv.reshape(batch_size * seq_len, seq_len, self.hidden_size)
        
        # Create causal mask: position i can only attend to positions [0:i+1]
        # Shape: [seq_len, seq_len] where mask[i,j] = True means position i cannot attend to j
        causal_mask = torch.triu(torch.ones(seq_len, seq_len, device=patch_embeddings.device, dtype=torch.bool), diagonal=1)
        
        # Expand to [batch*seq_len, num_compartments, seq_len]
        # The key insight: for cross-attention, at each position i in the sequence,
        # all num_compartments queries attend to the same set of historical patches [0:i+1]
        # So position 0 sees row 0 of causal_mask, position 1 sees row 1, etc.
        
        # Step 1: Expand for num_compartments: [seq_len, seq_len] -> [seq_len, num_compartments, seq_len]
        # Each row i is repeated for all compartments
        # Use repeat instead of expand to ensure memory is allocated
        causal_mask_per_comp = causal_mask.unsqueeze(1).repeat(1, self.num_compartments, 1)
        
        # Step 2: Repeat for batch size: [seq_len, num_compartments, seq_len] -> [batch*seq_len, num_compartments, seq_len]
        # Each of batch samples gets the same causal pattern
        causal_mask_final = causal_mask_per_comp.repeat(batch_size, 1, 1)
        
        # Convert boolean mask to float mask for attention (-inf for masked positions)
        causal_mask_float = causal_mask_final.float().masked_fill(causal_mask_final, float('-inf'))
        
        # Repeat for multi-head attention: [batch*seq_len, num_compartments, seq_len] -> [batch*seq_len*num_heads, num_compartments, seq_len]
        # PyTorch MultiheadAttention expects 3D mask to have shape (batch * num_heads, target_len, source_len)
        causal_mask_float = causal_mask_float.repeat_interleave(self.num_heads, dim=0)
        
        # Cross-attention: compartment queries at each position attend to historical patches only
        attn_out, _ = self.cross_attention(
            comp_queries_flat,      # [batch*seq_len, num_compartments, hidden_size]
            patch_kv_flat,          # [batch*seq_len, seq_len, hidden_size]
            patch_kv_flat,          # [batch*seq_len, seq_len, hidden_size]
            attn_mask=causal_mask_float,  # [batch*seq_len, num_compartments, seq_len]
            need_weights=False
        )
        
        # Reshape back to [batch, seq_len, num_compartments, hidden_size]
        attn_out = attn_out.reshape(batch_size, seq_len, self.num_compartments, self.hidden_size)
        
        # Residual connection + layer norm
        comp_embeddings = self.cross_attn_norm(comp_queries + attn_out)
        
        # Feedforward + residual + layer norm
        ff_out = self.cross_attn_ff(comp_embeddings)
        comp_embeddings = self.cross_attn_ff_norm(comp_embeddings + ff_out)
        
        # Apply compartment masking: zero out inactive compartments
        # [batch, 1, num_compartments, 1] to broadcast across seq_len and hidden_size
        mask_expanded = compartment_mask.unsqueeze(1).unsqueeze(-1).float()
        comp_embeddings = comp_embeddings * mask_expanded
        
        return comp_embeddings
    
    def create_multi_view_embeddings(self, 
                                    compartment_embeddings: torch.Tensor) -> torch.Tensor:
        """
        Create multi-view embeddings from compartment embeddings (vectorized)
        
        Args:
            compartment_embeddings: [batch, seq_len, num_compartments, hidden_size]
        
        Returns:
            multi_view_embeddings: [batch, seq_len, num_compartments, num_views, hidden_size]
        """
        batch, seq_len, num_compartments, _ = compartment_embeddings.shape
        all_views = self.multi_view_projection(compartment_embeddings).view(batch, seq_len, num_compartments, self.num_embeddings, -1)
        return all_views
    
    def forward(self, 
                I_timeseries: torch.Tensor, 
                compartment_mask: Optional[torch.Tensor] = None,
                compute_R_t: bool = False,
                return_hidden: bool = False,
                return_attention: bool = False,
                apply_preprocessing: bool = False) -> Dict[str, torch.Tensor]:
        """
        Forward pass with two-part architecture:
        PART 1: Encode patches (observed I) into embeddings (DECODER-ONLY/AUTOREGRESSIVE)
        PART 2: Cross-attention + multi-view + self-attention for compartmental modeling
        
        Args:
            I_timeseries: Observed infectious compartment [batch, seq_len, input_size]
            compartment_mask: Binary mask for active compartments [batch, num_compartments] or [num_compartments]
                            1 = active, 0 = masked out. If None, all compartments are used.
            compute_R_t: Whether to estimate R(t) time series from attention patterns
            return_hidden: Whether to return hidden states
            return_attention: Whether to return attention scores
        
        Returns:
            Dict mapping compartment name to predictions [batch, seq_len, input_size]
            If compute_R_t=True, includes 'R_t' key with reproduction number time series
            Masked compartments will have zero predictions
        """
        batch_size, seq_len, _ = I_timeseries.shape
        
        # Process compartment mask
        if compartment_mask is None:
            # Use all compartments for all samples
            compartment_mask = torch.ones(batch_size, self.num_compartments, 
                                         device=I_timeseries.device, dtype=torch.bool)
        elif compartment_mask.dim() == 1:
            # Same mask for all samples: [num_compartments] -> [batch, num_compartments]
            compartment_mask = compartment_mask.unsqueeze(0).expand(batch_size, -1)
        
        # ==================== RevIN: NORMALIZE INPUT ====================
        if self.use_revin:
            I_timeseries = self.revin(I_timeseries, mode='norm')
        
        # ==================== PART 1: ENCODER ====================
        # Encode input patches into embeddings (with optional preprocessing)
        patch_embeddings = self.encode_patches(I_timeseries, apply_preprocessing=apply_preprocessing)  # [batch, seq_len, hidden_size]

        # ==================== PART 2: COMPARTMENTAL MODELING ====================
        # Step 1: Create compartment embeddings via cross-attention (vectorized, with masking)
        compartment_embeddings = self.create_compartment_embeddings_with_cross_attention(
            patch_embeddings, batch_size, seq_len, compartment_mask
        )  # [batch, seq_len, num_compartments, hidden_size]
        
        # Step 2: Create multi-view embeddings (vectorized)
        multi_view_embs = self.create_multi_view_embeddings(compartment_embeddings)
        # [batch, seq_len, num_compartments, num_embeddings, hidden_size]

        # Step 3: Apply per-sample masking for transformer - VECTORIZED
        # Create mask for views based on active compartments per sample
        num_views = self.num_compartments * self.num_embeddings
        view_mask = compartment_mask.unsqueeze(-1).expand(-1, -1, self.num_embeddings)
        view_mask = view_mask.reshape(batch_size, num_views)
        view_mask_repeated = view_mask.unsqueeze(1).expand(-1, seq_len, -1).reshape(batch_size * seq_len, num_views)
        key_padding_mask = ~view_mask_repeated  # Invert: True = ignore this view
        
        # Step 4: Apply self-attention transformer per time step with masking
        multi_view_flat = multi_view_embs.reshape(batch_size * seq_len, num_views, self.hidden_size)
        
        # Add positional encoding for views (not time!)
        multi_view_flat = self.pos_encoder(multi_view_flat)
    
        # Apply transformer with key_padding_mask
        if compute_R_t or return_attention:
            contextualized_flat, attention_scores = self.transformer_with_attention(multi_view_flat, key_padding_mask)
        else:
            contextualized_flat = self.transformer_with_mask(multi_view_flat, key_padding_mask)
            attention_scores = None
        
        # Step 5: Reshape back and aggregate embeddings per compartment
        # Reshape: [batch, seq_len, num_compartments, num_embeddings, hidden_size]
        contextualized = contextualized_flat.reshape(batch_size, seq_len, self.num_compartments, self.num_embeddings, self.hidden_size)
        
        # Average over embeddings to get one vector per compartment: [batch, seq_len, num_compartments, hidden_size]
        compartment_output = contextualized.mean(dim=3)
        
        # Step 6: Generate predictions for each compartment
        predictions = {}
        for comp_idx, comp in enumerate(self.compartments):
            # Extract output for this compartment: [batch, seq_len, hidden_size]
            comp_out = compartment_output[:, :, comp_idx, :]
            # Apply compartment-specific output head: [batch, seq_len, input_size]
            raw_output = self.output_heads[comp](comp_out)
            # No clamping - let the model learn the full range of values
            # Apply RevIN denormalization if enabled
            if self.use_revin:
                raw_output = self.revin(raw_output, mode='denorm')
            predictions[comp] = raw_output
        
        # Step 7: Optionally compute R(t) from attention patterns
        if compute_R_t and attention_scores is not None:
            # Reshape contextualized for R(t) estimation: [batch, seq_len, num_views, hidden_size]
            contextualized_for_R_t = contextualized.reshape(batch_size, seq_len, num_views, self.hidden_size)
            
            R_t_pred = self.estimate_R_t_from_attention(
                attention_scores, 
                contextualized_for_R_t,
                batch_size,
                seq_len,
                num_views
            )
            predictions['R_t'] = R_t_pred
        
        # Step 8: Optional returns
        if return_attention:
            predictions['attention_scores'] = attention_scores
        
        if return_hidden:
            # Return compartment output embeddings as dict
            compartment_output_embeddings = {}
            for comp_idx, comp in enumerate(self.compartments):
                compartment_output_embeddings[comp] = compartment_output[:, :, comp_idx, :]
            return predictions, compartment_output_embeddings

        return predictions
    
    def compute_loss(self, 
                    predictions: Dict[str, torch.Tensor],
                    targets: Dict[str, torch.Tensor],
                    loss_weights: Optional[Dict[str, float]] = None,
                    compartment_mask: Optional[torch.Tensor] = None,
                    sequence_mask: Optional[torch.Tensor] = None,
                    mae_loss_weight: float = 0.0) -> Tuple[torch.Tensor, Dict[str, torch.Tensor]]:
        """Compute loss for all compartments (MSE + optional MAE)
        
        Args:
            predictions: Dict mapping compartment names to predicted tensors [B, T, D]
            targets: Dict mapping compartment names to target tensors [B, T, D]
            loss_weights: Optional weights for each compartment loss
            compartment_mask: [B, num_compartments] boolean mask indicating active compartments per sample
            sequence_mask: [B, T] boolean mask indicating valid (non-padded) time positions
            mae_loss_weight: Weight for MAE loss (0.0 = MSE only, 1.0 = equal MSE+MAE)
        """
        loss_weights = loss_weights or {}
        standard_comps = [c for c in predictions if c in targets and c in self.compartments]
        special_comps = [c for c in predictions if c in targets and c not in self.compartments]
        
        # If compartment_mask is None, create a default mask with all compartments active
        if compartment_mask is None:
            # Get batch size from first prediction
            first_pred = next(iter(predictions.values()))
            batch_size = first_pred.shape[0]
            device = first_pred.device
            compartment_mask = torch.ones(batch_size, len(self.compartments), dtype=torch.bool, device=device)
        
        losses, total_loss = {}, 0.0
        
        # Standard compartments - use MSE + optional MAE
        for comp in standard_comps:
            pred = predictions[comp]  # [B, T, D]
            targ = targets[comp]      # [B, T, D]
            
            # Get mask for this compartment
            comp_idx = self.compartments.index(comp)
            comp_mask = compartment_mask[:, comp_idx]  # [B]
            
            # Compute loss only for active samples
            if comp_mask.any():
                # Select only active samples
                pred_active = pred[comp_mask]  # [B_active, T, D]
                targ_active = targ[comp_mask]
                
                # Apply sequence mask if provided (to ignore padded positions)
                if sequence_mask is not None:
                    seq_mask_active = sequence_mask[comp_mask]  # [B_active, T]
                    # Expand mask to match tensor dimensions [B_active, T, D]
                    seq_mask_expanded = seq_mask_active.unsqueeze(-1).expand_as(pred_active)
                    
                    # Compute loss only on valid (non-padded) positions
                    if seq_mask_expanded.any():
                        diff = pred_active - targ_active
                        diff_squared = diff ** 2
                        # Masked MSE
                        mse_loss = (diff_squared * seq_mask_expanded).sum() / seq_mask_expanded.sum()
                        # Masked MAE
                        if mae_loss_weight > 0:
                            diff_abs = diff.abs()
                            mae_loss = (diff_abs * seq_mask_expanded).sum() / seq_mask_expanded.sum()
                            comp_loss = mse_loss + mae_loss_weight * mae_loss
                        else:
                            comp_loss = mse_loss
                    else:
                        comp_loss = torch.tensor(0.0, device=pred.device)
                else:
                    # No sequence mask - compute losses normally
                    mse_loss = F.mse_loss(pred_active, targ_active)
                    if mae_loss_weight > 0:
                        mae_loss = F.l1_loss(pred_active, targ_active)
                        comp_loss = mse_loss + mae_loss_weight * mae_loss
                    else:
                        comp_loss = mse_loss
                
                # import ipdb; ipdb.set_trace()
                
                losses[comp] = comp_loss
                total_loss += loss_weights.get(comp, 1.0) * comp_loss
            else:
                losses[comp] = torch.tensor(0.0, device=pred.device)
        
        # Special compartments (e.g., R_t)
        for comp in special_comps:
            pred = predictions[comp]
            targ = targets[comp]
            
            # Apply sequence mask if provided
            if sequence_mask is not None:
                seq_mask_expanded = sequence_mask.unsqueeze(-1).expand_as(pred)
                if seq_mask_expanded.any():
                    diff = pred - targ
                    diff_squared = diff ** 2
                    mse_loss = (diff_squared * seq_mask_expanded).sum() / seq_mask_expanded.sum()
                    if mae_loss_weight > 0:
                        diff_abs = diff.abs()
                        mae_loss = (diff_abs * seq_mask_expanded).sum() / seq_mask_expanded.sum()
                        losses[comp] = mse_loss + mae_loss_weight * mae_loss
                    else:
                        losses[comp] = mse_loss
                else:
                    losses[comp] = torch.tensor(0.0, device=pred.device)
            else:
                mse_loss = F.mse_loss(pred, targ)
                if mae_loss_weight > 0:
                    mae_loss = F.l1_loss(pred, targ)
                    losses[comp] = mse_loss + mae_loss_weight * mae_loss
                else:
                    losses[comp] = mse_loss
            
            total_loss += loss_weights.get(comp, 1.0) * losses[comp]
        
        return total_loss, losses
    
    def predict_next_token(self, 
                          I_context: torch.Tensor,
                          num_predictions: int = 1,
                          apply_postprocess: bool = True,
                          compartment_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Autoregressive prediction for multiple time steps.
        
        Args:
            I_context: Historical infectious compartment [batch, context_len, input_size]
            num_predictions: Number of future steps to predict
            apply_postprocess: Deprecated, kept for backward compatibility (ignored)
            compartment_mask: Binary mask for active compartments [num_compartments] or [batch, num_compartments]
                            1/True = active, 0/False = masked out. If None, all compartments are used.
        
        Returns:
            Dict of predictions for each compartment [batch, num_predictions, input_size]
        """
        self.eval()
        with torch.no_grad():
            current_I = I_context
            predictions_list = {comp: [] for comp in self.compartments}
            
            for _ in range(num_predictions):
                # Predict next step with compartment mask
                preds = self.forward(current_I, compartment_mask=compartment_mask)
                
                # Take last prediction for each compartment
                for comp, pred in preds.items():
                    next_token = pred[:, -1:, :]
                    predictions_list[comp].append(next_token)
                
                # Update I with prediction for next iteration
                if 'I' in preds:
                    next_I = preds['I'][:, -1:, :]
                    current_I = torch.cat([current_I, next_I], dim=1)
            
            # Concatenate predictions
            final_predictions = {}
            for comp, pred_list in predictions_list.items():
                if pred_list:  # If compartment was predicted
                    final_predictions[comp] = torch.cat(pred_list, dim=1)
            
            return final_predictions
    
    def transformer_with_mask(self, x, key_padding_mask=None):
        """
        Apply transformer with optional key_padding_mask (BIDIRECTIONAL attention)
        
        This is used for self-attention between compartment views at each time step.
        Views should be able to attend to ALL other active views (bidirectional),
        not just previous views (causal). The causal constraint is for time,
        which is handled by the patch encoder.
        
        Args:
            x: [batch, seq_len, hidden_size]
            key_padding_mask: [batch, seq_len] boolean mask, True = ignore
        
        Returns:
            output: [batch, seq_len, hidden_size]
        """
        # NO causal mask - bidirectional attention for compartment views
        # Only key_padding_mask is used to ignore inactive compartments
        
        # Apply transformer layers
        for layer in self.transformer.layers:
            attn_out, _ = layer.self_attn(x, x, x, attn_mask=None, key_padding_mask=key_padding_mask, need_weights=False)
            x = layer.norm1(x + layer.dropout(attn_out))
            
            # Feedforward
            ff_out = layer.ff(x)
            x = layer.norm2(x + ff_out)
        
        # Final normalization
        x = self.transformer.final_norm(x)
        
        return x
    
    def transformer_with_attention(self, x, key_padding_mask=None):
        """
        Apply transformer and return both output and attention scores (BIDIRECTIONAL)
        
        This is used for self-attention between compartment views at each time step.
        Views should be able to attend to ALL other active views (bidirectional).
        
        Args:
            x: [batch, seq_len, hidden_size]
            key_padding_mask: [batch, seq_len] boolean mask, True = ignore
        
        Returns:
            output: [batch, seq_len, hidden_size]
            attention_scores: List of attention weights from each layer [batch, num_heads, seq_len, seq_len]
        """
        # NO causal mask - bidirectional attention for compartment views
        
        attention_scores = []
        
        # Apply transformer layers and collect attention
        for layer in self.transformer.layers:
            # Get attention weights (need_weights=True)
            attn_out, attn_weights = layer.self_attn(x, x, x, attn_mask=None, key_padding_mask=key_padding_mask, need_weights=True, average_attn_weights=False)
            x = layer.norm1(x + layer.dropout(attn_out))
            
            # Feedforward
            ff_out = layer.ff(x)
            x = layer.norm2(x + ff_out)
            
            attention_scores.append(attn_weights)  # [batch, num_heads, seq_len, seq_len]
        
        # Final normalization
        x = self.transformer.final_norm(x)
        
        return x, attention_scores
    
    def estimate_R_t_from_attention(self, attention_scores, multi_view_embeddings, batch_size, seq_len, num_views):
        """
        Estimate R(t) time series from attention patterns with bounds
        
        FAST APPROACH:
        - Use attention scores to compute F and V matrices at each time point
        - Compute R(t) bounds: R_t_lower = σ_min(F) / σ_max(V), R_t_upper = σ_max(F) / σ_min(V)
        - Average bounds to get final R(t) estimate
        - No forward passes needed!
        
        Args:
            attention_scores: List of attention weights from each layer
            multi_view_embeddings: [batch, seq_len, num_views, hidden_size]
            batch_size: Batch size
            seq_len: Sequence length
            num_views: Number of views per compartment
        
        Returns:
            R_t: [batch, seq_len, input_size] - Reproduction number time series
        """
        # Extract I compartment views
        I_comp_idx = self.compartments.index('I')
        I_start_view = I_comp_idx * self.num_embeddings
        I_end_view = I_start_view + self.num_embeddings
        num_I_views = I_end_view - I_start_view
        
        # Extract I compartment embeddings: [batch, seq_len, num_I_views, hidden_size]
        I_embeddings = multi_view_embeddings[:, :, I_start_view:I_end_view, :]
        
        # Pre-process attention scores for F/V computation
        # Last layer attention: [batch*seq_len, num_heads, num_views, num_views]
        last_attn = attention_scores[-1]
        # Reshape and average heads: [batch, seq_len, num_views, num_views]
        attn_avg = last_attn.view(batch_size, seq_len, -1, num_views, num_views).mean(dim=2)
        # Extract I compartment attention: [batch, seq_len, num_I_views, num_I_views]
        attn_I = attn_avg[:, :, I_start_view:I_end_view, I_start_view:I_end_view]
        
        # Compute time-varying F and V matrices from attention patterns
        R_t_values = []
        
        for t in range(seq_len):
            # Get attention matrix at time t: [batch, num_I_views, num_I_views]
            A_t = attn_I[:, t, :, :]
            
            # F matrix: Transmission (off-diagonal flow)
            # We interpret off-diagonal attention as transmission between sub-compartments
            F_matrix = A_t.clone()
            diagonal_mask = torch.eye(num_I_views, device=A_t.device).bool().unsqueeze(0)
            F_matrix.masked_fill_(diagonal_mask, 0)
            F_matrix = F_matrix
            
            # V matrix: Transition/Removal (inverse of stability)
            # Diagonal attention represents stability (staying in state)
            # Removal rate ~ 1 / stability
            stability = torch.diagonal(A_t, dim1=-2, dim2=-1)  # [batch, num_I_views]
            removal_rates = 1.0 / (stability + 0.1)  # Avoid division by zero
            V_matrix = torch.diag_embed(removal_rates)
            
            # Compute R(t) bounds from singular values
            F_svd = torch.linalg.svdvals(F_matrix)  # [batch, num_I_views]
            V_svd = torch.linalg.svdvals(V_matrix)  # [batch, num_I_views]
            
            # R(t) bounds
            R_t_lower = F_svd.min(dim=1)[0] / (V_svd.max(dim=1)[0] + 1e-8)  # [batch]
            R_t_upper = F_svd.max(dim=1)[0] / (V_svd.min(dim=1)[0] + 1e-8)  # [batch]
            
            # Average of bounds as R(t) estimate
            R_t_value = (R_t_lower + R_t_upper) / 2.0  # [batch]
            R_t_values.append(R_t_value)
        
        # Stack R(t) values across time
        R_t_series = torch.stack(R_t_values, dim=1)  # [batch, seq_len]
        R_t_series_expanded = R_t_series.unsqueeze(-1).expand(-1, -1, self.input_size)  # [batch, seq_len, input_size]
        R_t_scaled = self.R_t_head(R_t_series_expanded)  # [batch, seq_len, input_size]
        
        return R_t_scaled


    def ensemble_predict(self,
                          I_context: torch.Tensor,
                          num_predictions: int = 1,
                          fixed_masks: Optional[List[torch.Tensor]] = None,
                          num_masks: int = 20,
                          mask_seed: int = 42,
                          num_input_tokens: Optional[int] = None,
                          apply_preprocessing: bool = True) -> Dict[str, torch.Tensor]:
        """
        Ensemble prediction with multiple compartmental masks and adaptive percentile aggregation.
        
        This method integrates preprocessing and ensemble directly into the forward process:
        1. If num_input_tokens is provided, treat I_context as INPUT CONTEXT (no labels):
           - Apply smooth_light on full input context (for proper historical smoothing)
           - Extract last num_input_tokens as the actual input
        2. Otherwise, apply preprocessing directly to I_context (backward compatible)
        3. Use fixed masks for reproducibility
        4. Collect predictions from all masks
        5. Apply adaptive_percentile ensemble to combine predictions
        
        IMPORTANT: I_context should NOT include label tokens to avoid leakage.
        
        Args:
            I_context: Historical context [batch, context_len, input_size]
                       If num_input_tokens is set, this is full INPUT context (no labels)
            num_predictions: Number of future steps to predict
            fixed_masks: Pre-generated fixed masks (recommended for reproducibility)
            num_masks: Number of masks to generate if fixed_masks not provided
            mask_seed: Random seed for generating masks if fixed_masks not provided
            num_input_tokens: If set, treat I_context as full input context and extract last N tokens
                             This enables proper smoothing with historical context
            apply_preprocessing: If True, apply smooth_light preprocessing (default: True)
        
        Returns:
            Dict of ensembled predictions for each compartment [batch, num_predictions, input_size]
        """
        self.eval()
        device = I_context.device
        dtype = I_context.dtype
        
        # Handle full context preprocessing
        if num_input_tokens is not None and apply_preprocessing:
            # I_context is INPUT CONTEXT (no labels): smooth it, then extract last num_input_tokens
            batch_size = I_context.shape[0]
            smoothed_contexts = []
            for i in range(batch_size):
                ctx_np = I_context[i].cpu().numpy()
                smoothed = preprocess_smooth_light(ctx_np)  # Smooth input context
                # Extract last num_input_tokens (context doesn't include labels, so just take last N)
                input_tokens = smoothed[-num_input_tokens:]
                smoothed_contexts.append(torch.from_numpy(input_tokens))
            I_context = torch.stack(smoothed_contexts).to(device=device, dtype=dtype)
            # Preprocessing is now done, don't do it again in forward
            do_preprocessing_in_forward = False
        else:
            do_preprocessing_in_forward = apply_preprocessing
        
        # Use provided fixed masks or generate new ones
        if fixed_masks is not None:
            masks = [m.to(device) for m in fixed_masks]
        else:
            # Generate masks (for backward compatibility)
            rng = np.random.RandomState(mask_seed)
            masks = []
            for _ in range(num_masks):
                mask = torch.zeros(self.num_compartments, dtype=torch.bool, device=device)
                mask[:2] = True  # S and I always active
                mask[2:] = torch.tensor(rng.rand(self.num_compartments - 2) < 0.5, device=device)
                masks.append(mask)
        
        with torch.no_grad():
            # Collect predictions from all masks
            all_mask_predictions = {comp: [] for comp in self.compartments}
            
            for mask in masks:
                current_I = I_context
                predictions_list = {comp: [] for comp in self.compartments}
                
                for step in range(num_predictions):
                    # Forward pass with preprocessing (only on first step to avoid double-smoothing)
                    preds = self.forward(
                        current_I, 
                        compartment_mask=mask,
                        apply_preprocessing=(do_preprocessing_in_forward and step == 0)
                    )
                    
                    # Take last prediction for each compartment
                    for comp, pred in preds.items():
                        if comp in self.compartments:
                            next_token = pred[:, -1:, :]
                            predictions_list[comp].append(next_token)
                    
                    # Update I with prediction for next iteration
                    if 'I' in preds:
                        next_I = preds['I'][:, -1:, :]
                        current_I = torch.cat([current_I, next_I], dim=1)
                
                # Concatenate predictions for this mask
                for comp, pred_list in predictions_list.items():
                    if pred_list:
                        mask_pred = torch.cat(pred_list, dim=1)  # [batch, num_predictions, input_size]
                        all_mask_predictions[comp].append(mask_pred)
            
            # Apply adaptive percentile ensemble for each compartment
            final_predictions = {}
            for comp, mask_preds in all_mask_predictions.items():
                if mask_preds:
                    # Stack: [num_masks, batch, num_predictions, input_size]
                    stacked = torch.stack(mask_preds, dim=0)
                    # Apply ensemble
                    ensembled = ensemble_adaptive_percentile(stacked)
                    final_predictions[comp] = ensembled
            
            return final_predictions

    def forecast(self,
                 x: torch.Tensor,
                 num_predictions: int = 1,
                 preprocessing: str = 'smooth_light',
                 ensemble: str = 'max',
                 num_masks: int = 20,
                 mask_seed: int = 42,
                 fixed_masks: Optional[List[torch.Tensor]] = None,
                 context: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Unified forecasting interface - like Chronos, takes raw input and returns predictions.
        
        This method provides a clean, simple interface that handles:
        1. Preprocessing (e.g., smooth_light with size=4)
        2. Multi-mask prediction collection
        3. Ensemble aggregation (e.g., max)
        
        Args:
            x: Input tensor [batch, num_input, input_size] - raw normalized tokens
            num_predictions: Number of future tokens to predict
            preprocessing: Preprocessing strategy name ('none', 'smooth_light', etc.)
            ensemble: Ensemble strategy name ('max', 'mean', 'adaptive_percentile', etc.)
            num_masks: Number of compartment masks to use
            mask_seed: Random seed for mask generation
            fixed_masks: Optional pre-generated masks for reproducibility
            context: Optional full historical context [batch, context_len, input_size]
                     If provided, preprocessing is applied to context, then last num_input tokens are extracted.
                     If None, preprocessing is applied directly to x.
        
        Returns:
            Predictions tensor [batch, num_predictions, input_size] (I compartment)
        """
        self.eval()
        device = x.device
        dtype = x.dtype
        batch_size = x.shape[0]
        num_input = x.shape[1]
        
        # Get preprocessing function (uses default size from cape_preprocessing.py)
        preprocess_fn = get_preprocessing_fn(preprocessing)
        
        # Get ensemble function (special handling for 'learnable')
        use_learnable_ensemble = (ensemble == 'learnable')
        if not use_learnable_ensemble:
            ensemble_fn = get_ensemble_strategy(ensemble)
        
        # Apply preprocessing
        if context is not None:
            # Apply preprocessing to full context, then extract last num_input tokens
            preprocessed_list = []
            for i in range(batch_size):
                ctx_np = context[i].cpu().numpy()
                preprocessed = preprocess_fn(ctx_np)
                # Extract last num_input tokens
                input_tokens = preprocessed[-num_input:]
                preprocessed_list.append(torch.from_numpy(input_tokens))
            x_preprocessed = torch.stack(preprocessed_list).to(device=device, dtype=dtype)
        else:
            # Apply preprocessing directly to input tokens
            preprocessed_list = []
            for i in range(batch_size):
                sample_np = x[i].cpu().numpy()
                preprocessed = preprocess_fn(sample_np)
                preprocessed_list.append(torch.from_numpy(preprocessed))
            x_preprocessed = torch.stack(preprocessed_list).to(device=device, dtype=dtype)
        
        # Generate or use fixed masks
        if fixed_masks is not None:
            masks = [m.to(device) for m in fixed_masks]
        else:
            rng = np.random.RandomState(mask_seed)
            masks = []
            for _ in range(num_masks):
                mask = torch.zeros(self.num_compartments, dtype=torch.bool, device=device)
                mask[:2] = True  # S and I always active
                mask[2:] = torch.tensor(rng.rand(self.num_compartments - 2) < 0.5, device=device)
                masks.append(mask)
        
        with torch.no_grad():
            # Collect predictions from all masks
            all_mask_preds = []  # Will be [num_masks, batch, num_predictions, input_size]
            
            for mask in masks:
                # Autoregressive prediction
                current_input = x_preprocessed
                step_preds = []
                
                for _ in range(num_predictions):
                    preds = self.forward(current_input, compartment_mask=mask)
                    
                    if 'I' in preds:
                        next_token = preds['I'][:, -1:, :]  # [batch, 1, input_size]
                        step_preds.append(next_token)
                        # Update input for next step
                        current_input = torch.cat([current_input, next_token], dim=1)
                
                if step_preds:
                    mask_pred = torch.cat(step_preds, dim=1)  # [batch, num_predictions, input_size]
                    all_mask_preds.append(mask_pred)
            
            # Stack: [num_masks, batch, num_predictions, input_size]
            stacked = torch.stack(all_mask_preds, dim=0)
            
            # Apply ensemble
            if use_learnable_ensemble:
                # Use built-in learnable ensemble (trains percentile_logit and spread_scale_logit)
                result = self.ensemble_learnable(stacked)  # [batch, num_predictions, input_size]
            else:
                # Apply ensemble PER SAMPLE (required for adaptive methods)
                ensembled_list = []
                for i in range(batch_size):
                    sample_preds = stacked[:, i:i+1, :, :]  # [num_masks, 1, seq, dim]
                    sample_ensembled = ensemble_fn(sample_preds)  # [1, seq, dim]
                    ensembled_list.append(sample_ensembled)
                
                result = torch.cat(ensembled_list, dim=0)  # [batch, num_predictions, input_size]
            
            return result

    def __call__(self, x: torch.Tensor, **kwargs) -> torch.Tensor:
        """
        Make model callable like Chronos: model(x) returns predictions.
        
        During training (when compartment_mask or compute_R_t are passed), 
        delegates to forward() for training mode.
        
        During inference, uses smooth_light preprocessing and max ensemble.
        For more control, use forecast() method directly.
        """
        # Training mode: delegate to forward() when training-specific args are passed
        training_args = {'compartment_mask'}
        if any(k in kwargs for k in training_args):
            return self.forward(x, **kwargs)
        
        # Inference mode: use forecast with ensemble
        return self.forecast(x, **kwargs)


def create_compartmental_cape(config):
    """Factory function to create CompartmentalCAPE from config"""
    return CompartmentalCAPE(
        input_size=config.token_size if hasattr(config, 'token_size') else 1,
        hidden_size=config.hidden_size if hasattr(config, 'hidden_size') else 128,
        num_layers=config.layers if hasattr(config, 'layers') else 4,
        num_heads=config.num_heads if hasattr(config, 'num_heads') else 8,
        num_embeddings=config.num_embeddings if hasattr(config, 'num_embeddings') else 6,
        dropout=config.dropout if hasattr(config, 'dropout') else 0.1,
        max_tokens=config.max_tokens if hasattr(config, 'max_tokens') else 512,
        compartments=config.compartments if hasattr(config, 'compartments') else None,
        patch_encoder_type=config.patch_encoder_type if hasattr(config, 'patch_encoder_type') else 'transformer',
        ff_ratio=config.ff_ratio if hasattr(config, 'ff_ratio') else 4
    )
