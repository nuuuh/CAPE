#!/usr/bin/env python
"""
CAPE Shared Utilities for Online Evaluation
=============================================
Core functions for data loading, model loading, prediction collection,
and ensemble evaluation.

Used by strategy_online.py for rolling-fold evaluation.
"""

import os
import sys
import json
import warnings
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings('ignore')

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from model.CAPE_Compartmental import CompartmentalCAPE, preprocess_smooth_light

from ensemble_strategies import get_ensemble_strategy, AVAILABLE_STRATEGIES


def setup_seed(seed=15):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# FIXED COMPARTMENTAL MASKS
# =============================================================================

def generate_fixed_masks(num_masks, num_compartments=12, seed=42):
    """Generate a fixed set of compartmental masks on CPU."""
    rng = np.random.RandomState(seed)
    masks = []
    for _ in range(num_masks):
        mask = torch.zeros(num_compartments, dtype=torch.bool)
        mask[:2] = True  # S and I always on
        mask[2:] = torch.tensor(rng.rand(num_compartments - 2) < 0.5)
        masks.append(mask)
    return masks


_FIXED_MASKS_CACHE = {}
_LOADED_MASKS_FROM_CHECKPOINT = None


def get_fixed_masks(num_masks, device='cpu', seed=42):
    """Get fixed masks, moving to the specified device."""
    global _LOADED_MASKS_FROM_CHECKPOINT
    if _LOADED_MASKS_FROM_CHECKPOINT is not None:
        loaded = _LOADED_MASKS_FROM_CHECKPOINT
        if len(loaded) >= num_masks:
            return [mask.to(device) for mask in loaded[:num_masks]]
        print(f"  Warning: Checkpoint has {len(loaded)} masks, but {num_masks} requested.")

    cache_key = (num_masks, seed)
    if cache_key not in _FIXED_MASKS_CACHE:
        _FIXED_MASKS_CACHE[cache_key] = generate_fixed_masks(num_masks, seed=seed)
    return [mask.to(device) for mask in _FIXED_MASKS_CACHE[cache_key]]


def load_masks_from_checkpoint(pretrain_path):
    """Load masks from checkpoint if available."""
    global _LOADED_MASKS_FROM_CHECKPOINT
    try:
        checkpoint = torch.load(pretrain_path, map_location='cpu', weights_only=False)
        if 'fixed_masks' in checkpoint:
            _LOADED_MASKS_FROM_CHECKPOINT = checkpoint['fixed_masks']
            num_masks = len(_LOADED_MASKS_FROM_CHECKPOINT)
            mask_seed = checkpoint.get('mask_seed', 'unknown')
            print(f"  Loaded {num_masks} masks from checkpoint (seed={mask_seed})")
            return True
    except Exception as e:
        print(f"  Warning: Could not load masks from checkpoint: {e}")
    return False


# =============================================================================
# DATA LOADING
# =============================================================================

class TokenDataset(Dataset):
    def __init__(self, tokens: np.ndarray, num_input: int, num_output: int):
        total = num_input + num_output
        self.samples = [(tokens[i:i+num_input], tokens[i+num_input:i+total])
                        for i in range(len(tokens) - total + 1)]

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        inp, out = self.samples[idx]
        return {'input': torch.tensor(inp, dtype=torch.float32),
                'label': torch.tensor(out, dtype=torch.float32)}


def load_data(data_path, disease, token_size=4, train_rate=0.3, valid_rate=0.1):
    data = torch.load(data_path, weights_only=False)[disease]
    total = {}
    for values in data.values():
        for w, t in enumerate(values[0][1]):
            total[t.item()] = total.get(t.item(), 0) + int(values[0][0][w].item())

    values = np.array([total[k] for k in sorted(total)], dtype=np.float32)
    scaler = StandardScaler()
    values_norm = scaler.fit_transform(values.reshape(-1, 1)).flatten()

    num_tokens = len(values_norm) // token_size
    tokens = values_norm[:num_tokens * token_size].reshape(num_tokens, token_size)
    train_end = int(num_tokens * train_rate)
    valid_end = int(num_tokens * (train_rate + valid_rate))

    return tokens[:train_end], tokens[train_end:valid_end], tokens[valid_end:], scaler


def load_pretrained_cape(pretrain_path, device):
    checkpoint = torch.load(pretrain_path, map_location=device, weights_only=False)
    config_path = os.path.join(os.path.dirname(pretrain_path), 'model_config.json')
    cfg = json.load(open(config_path)) if os.path.exists(config_path) else {}

    model = CompartmentalCAPE(
        input_size=cfg.get('token_size', 4),
        hidden_size=cfg.get('hidden_dim', cfg.get('hidden_size', 256)),
        num_layers=cfg.get('num_layers', 6),
        num_heads=cfg.get('num_heads', 4),
        num_embeddings=cfg.get('num_embeddings', 6),
        patch_encoder_type=cfg.get('patch_encoder_type', 'transformer')
    )

    state = {k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()}
    model.load_state_dict(state, strict=False)
    load_masks_from_checkpoint(pretrain_path)
    return model.to(device)


# =============================================================================
# PREDICTION COLLECTION & EVALUATION
# =============================================================================

def collect_predictions(model, dataloader, num_output, num_masks, device,
                        config=None, use_fixed_masks=True):
    """Collect predictions from multiple masks.

    Returns:
        all_preds: [num_masks, total_samples, seq, features]
        all_targets: [total_samples, seq, features]
    """
    model.eval()
    if use_fixed_masks:
        fixed_masks = get_fixed_masks(num_masks, device=device)

    all_preds, all_targets = [], []
    with torch.no_grad():
        for batch in dataloader:
            x, y = batch['input'].to(device), batch['label'].to(device)
            ensemble = []
            for i in range(num_masks):
                mask = fixed_masks[i] if use_fixed_masks else _random_mask(device)
                preds = model.predict_next_token(x, num_predictions=num_output,
                                                 apply_postprocess=True, compartment_mask=mask)
                ensemble.append(preds['I'])
            all_preds.append(torch.stack(ensemble, dim=0))
            all_targets.append(y)

    return torch.cat(all_preds, dim=1), torch.cat(all_targets, dim=0)


def _random_mask(device):
    mask = torch.zeros(12, device=device, dtype=torch.bool)
    mask[:2] = True
    mask[2:] = torch.rand(10, device=device) < 0.5
    return mask


def evaluate_ensemble(ensemble_fn, preds, targets, metrics_list=None):
    """Evaluate an ensemble function. Uses LAST output token."""
    combined = ensemble_fn(preds)
    pred_np = combined[:, -1:, :].cpu().numpy()
    target_np = targets[:, -1:, :].cpu().numpy()

    if metrics_list is not None:
        try:
            from strategy_online import compute_selected_metrics
        except ImportError:
            from src.strategy_online import compute_selected_metrics
        if preds.dim() == 4:
            last_input = preds[0, :, -1:, :].cpu().numpy()
        else:
            last_input = None
        return compute_selected_metrics(pred_np, target_np, metrics_list, last_input)
    else:
        mse = float(np.mean((pred_np - target_np) ** 2))
        mae = float(np.mean(np.abs(pred_np - target_np)))
        return mse, mae
