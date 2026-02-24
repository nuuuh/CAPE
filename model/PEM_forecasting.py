"""
PEM (Pre-trained Epidemic Model) for Downstream Forecasting
============================================================
This module provides PEM model variants for use with evaluate_unified.py.
PEM uses pre-trained encoder weights and adds a forecasting head.
"""

import torch
import torch.nn as nn
import os
import json
from typing import Dict, Optional


class Transpose(nn.Module):
    def __init__(self, *dims, contiguous=False): 
        super().__init__()
        self.dims, self.contiguous = dims, contiguous
    def forward(self, x):
        if self.contiguous: return x.transpose(*self.dims).contiguous()
        else: return x.transpose(*self.dims)


class PEMForForecasting(nn.Module):
    """
    PEM model for forecasting tasks.
    Uses pre-trained encoder and replaces reconstruction head with forecast head.
    """
    
    def __init__(self, patch_len: int, num_patches: int, horizon: int,
                 d_model: int = 256, n_layers: int = 6, n_heads: int = 8, 
                 dropout: float = 0.1, pretrain_path: str = None):
        super().__init__()
        self.patch_len = patch_len
        self.num_patches = num_patches
        self.horizon = horizon
        self.d_model = d_model
        
        # Import PatchTST components
        from model.TS_lib.Transformer_EncDec import Encoder, EncoderLayer
        from model.TS_lib.SelfAttention_Family import FullAttention, AttentionLayer
        from model.TS_lib.Embed import PatchEmbedding
        
        # Patch embedding
        self.patch_embedding = PatchEmbedding(d_model, patch_len, patch_len, 0, dropout)
        
        # Position embedding
        self.pos_embedding = nn.Parameter(torch.zeros(1, num_patches, d_model))
        nn.init.trunc_normal_(self.pos_embedding, std=0.02)
        
        # Transformer encoder
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(False, attention_dropout=dropout, output_attention=False),
                        d_model, n_heads
                    ),
                    d_model,
                    4 * d_model,
                    dropout=dropout,
                    activation='gelu'
                ) for _ in range(n_layers)
            ],
            norm_layer=nn.Sequential(Transpose(1, 2), nn.BatchNorm1d(d_model), Transpose(1, 2))
        )
        
        self.dropout_layer = nn.Dropout(dropout)
        
        # Forecasting head (replaces reconstruction head)
        self.forecast_head = nn.Sequential(
            nn.Linear(num_patches * d_model, d_model),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model, horizon)
        )
        
        # Load pre-trained weights if provided
        if pretrain_path and os.path.exists(pretrain_path):
            self._load_pretrained(pretrain_path)
    
    def _load_pretrained(self, path: str):
        """Load pre-trained encoder weights, handling size mismatches for pos_embedding."""
        checkpoint = torch.load(path, map_location='cpu')
        
        # Get encoder state dict
        if 'encoder_state_dict' in checkpoint:
            state_dict = checkpoint['encoder_state_dict']
        elif 'model_state_dict' in checkpoint:
            # Filter out reconstruction head
            state_dict = {k: v for k, v in checkpoint['model_state_dict'].items()
                         if not k.startswith('reconstruction_head')}
        else:
            state_dict = checkpoint
        
        # Handle pos_embedding size mismatch (pretrained with different num_patches)
        if 'pos_embedding' in state_dict:
            pretrain_pos_emb = state_dict['pos_embedding']  # [1, pretrain_num_patches, d_model]
            pretrain_num_patches = pretrain_pos_emb.shape[1]
            current_num_patches = self.num_patches
            
            if pretrain_num_patches != current_num_patches:
                print(f"  Adapting pos_embedding: {pretrain_num_patches} -> {current_num_patches} patches")
                # Interpolate position embeddings to match current num_patches
                # Shape: [1, pretrain_num_patches, d_model] -> [1, current_num_patches, d_model]
                pretrain_pos_emb = pretrain_pos_emb.permute(0, 2, 1)  # [1, d_model, pretrain_num_patches]
                adapted_pos_emb = torch.nn.functional.interpolate(
                    pretrain_pos_emb, 
                    size=current_num_patches, 
                    mode='linear', 
                    align_corners=False
                )
                adapted_pos_emb = adapted_pos_emb.permute(0, 2, 1)  # [1, current_num_patches, d_model]
                state_dict['pos_embedding'] = adapted_pos_emb
        
        # Filter out keys that have size mismatches we can't adapt (like forecast_head)
        model_state = self.state_dict()
        filtered_state_dict = {}
        for k, v in state_dict.items():
            if k in model_state:
                if v.shape == model_state[k].shape:
                    filtered_state_dict[k] = v
                else:
                    print(f"  Skipping {k}: shape mismatch {v.shape} vs {model_state[k].shape}")
            else:
                # Key not in current model (e.g., reconstruction_head), skip it
                pass
        
        # Load weights (strict=False to allow missing forecast_head)
        missing, unexpected = self.load_state_dict(filtered_state_dict, strict=False)
        
        if missing:
            # Only warn about truly missing keys (forecast_head is expected)
            actual_missing = [k for k in missing if 'forecast_head' not in k]
            if actual_missing:
                print(f"  Warning: Missing keys: {actual_missing}")
        
        print(f"  Loaded pre-trained encoder from {path}")
    
    def forward(self, x: torch.Tensor, time=None, dec_time=None, mask=None) -> torch.Tensor:
        """
        Forward pass for forecasting.
        
        Args:
            x: [batch, seq_len, 1] or [batch, num_patches, patch_len]
        Returns:
            forecast: [batch, horizon]
        """
        # Handle [batch, seq_len, 1] input format from evaluate_unified.py
        if x.dim() == 3 and x.size(-1) == 1:
            x = x.squeeze(-1)  # [batch, seq_len]
            batch_size, seq_len = x.shape
            num_patches = seq_len // self.patch_len
            x = x[:, :num_patches * self.patch_len].reshape(batch_size, num_patches, self.patch_len)
        
        # Patch embedding
        x = self.patch_embedding(x)  # [batch, num_patches, d_model]
        
        # Add position embedding
        x = x + self.pos_embedding[:, :x.size(1), :]
        
        # Dropout
        x = self.dropout_layer(x)
        
        # Transformer encoder
        x, _ = self.encoder(x)  # [batch, num_patches, d_model]
        
        # Flatten and project to horizon
        x = x.flatten(1)  # [batch, num_patches * d_model]
        forecast = self.forecast_head(x)  # [batch, horizon]
        
        return forecast


def create_pem_model(input_len: int, output_len: int, params: Dict, 
                     device: str = 'cuda', pretrain_path: str = None) -> PEMForForecasting:
    """
    Factory function to create PEM model for forecasting.
    
    Args:
        input_len: Input sequence length (will be divided into patches)
        output_len: Forecast horizon
        params: Model parameters (lr, etc.)
        device: Device to use
        pretrain_path: Path to pre-trained checkpoint
    
    Returns:
        PEMForForecasting model
    """
    patch_len = params.get('patch_len', 4)
    num_patches = input_len // patch_len
    
    # Load config from pretrain dir if available
    d_model = params.get('d_model', 256)
    n_layers = params.get('n_layers', 6)
    n_heads = params.get('n_heads', 8)
    dropout = params.get('dropout', 0.1)
    
    if pretrain_path and os.path.exists(pretrain_path):
        config_path = os.path.join(os.path.dirname(pretrain_path), 'model_config.json')
        if os.path.exists(config_path):
            with open(config_path) as f:
                cfg = json.load(f)
            d_model = cfg.get('d_model', d_model)
            n_layers = cfg.get('n_layers', n_layers)
            n_heads = cfg.get('n_heads', n_heads)
            dropout = cfg.get('dropout', dropout)
            patch_len = cfg.get('patch_len', patch_len)
            num_patches = input_len // patch_len
    
    model = PEMForForecasting(
        patch_len=patch_len,
        num_patches=num_patches,
        horizon=output_len,
        d_model=d_model,
        n_layers=n_layers,
        n_heads=n_heads,
        dropout=dropout,
        pretrain_path=pretrain_path
    )
    
    return model.to(device)
