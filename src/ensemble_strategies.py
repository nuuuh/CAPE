"""
Ensemble Strategies for CAPE Compartmental Mask Predictions.

This module provides:
1. BEST_STRATEGIES: Non-learnable strategies (model predictions only)
2. BEST_OBS_AWARE: Non-learnable observation-aware strategies (model + observations)
3. LEARNABLE_ENSEMBLES: Learnable ensemble modules (model predictions only)

All strategies combine predictions from multiple compartment mask configurations.
"""

import torch
import torch.nn as nn
from typing import Callable, Dict


# =============================================================================
# NON-LEARNABLE BASE STRATEGIES
# =============================================================================
# All take predictions [n_masks, batch, seq, dim] and return [batch, seq, dim].

def ensemble_percentile(predictions: torch.Tensor, percentile: float = 75.0) -> torch.Tensor:
    """Return specific percentile of predictions."""
    k = int(predictions.shape[0] * percentile / 100.0)
    k = max(0, min(k, predictions.shape[0] - 1))
    sorted_preds, _ = predictions.sort(dim=0)
    return sorted_preds[k]


def ensemble_median_max_blend(predictions: torch.Tensor, blend: float = 0.5) -> torch.Tensor:
    """Blend of median (robust) and max (optimistic)."""
    median_pred = predictions.median(dim=0).values
    max_pred = predictions.max(dim=0).values
    return (1 - blend) * median_pred + blend * max_pred


def ensemble_upper_half_mean(predictions: torch.Tensor) -> torch.Tensor:
    """Mean of predictions above median."""
    n_masks = predictions.shape[0]
    median_idx = n_masks // 2
    sorted_preds, _ = predictions.sort(dim=0)
    return sorted_preds[median_idx:].mean(dim=0)


def ensemble_biased_mean(predictions: torch.Tensor, bias_factor: float = 0.5) -> torch.Tensor:
    """Bias-corrected mean: mean + bias_factor * (max - mean)."""
    mean_pred = predictions.mean(dim=0)
    max_pred = predictions.max(dim=0).values
    return mean_pred + bias_factor * (max_pred - mean_pred)


def ensemble_optimistic_weighted(predictions: torch.Tensor, optimism: float = 2.0) -> torch.Tensor:
    """Weight higher predictions more than lower ones."""
    n_masks = predictions.shape[0]
    sorted_preds, _ = predictions.sort(dim=0)
    ranks = torch.arange(n_masks, device=predictions.device, dtype=torch.float32)
    weights = (ranks + 1) ** optimism
    weights = (weights / weights.sum()).view(-1, 1, 1, 1)
    return (sorted_preds * weights).sum(dim=0)


def ensemble_robust_optimal(predictions: torch.Tensor) -> torch.Tensor:
    """Robust optimal: weighted combination tuned for epidemic forecasting."""
    n_masks = predictions.shape[0]
    sorted_preds, _ = predictions.sort(dim=0)
    mean_pred = predictions.mean(dim=0)
    median_pred = predictions.median(dim=0).values
    p75 = sorted_preds[(3 * n_masks) // 4]
    max_pred = predictions.max(dim=0).values
    return 0.30 * p75 + 0.30 * max_pred + 0.25 * mean_pred + 0.15 * median_pred


def ensemble_adaptive_percentile(predictions: torch.Tensor) -> torch.Tensor:
    """Adaptively choose percentile based on prediction agreement."""
    n_masks = predictions.shape[0]
    sorted_preds, _ = predictions.sort(dim=0)
    mean_pred = predictions.mean(dim=0)
    std_pred = predictions.std(dim=0)
    cv = (std_pred / (mean_pred.abs() + 1e-8)).mean()
    target_percentile = 0.5 + 0.4 * torch.sigmoid(cv * 3 - 1.5)
    idx = target_percentile * (n_masks - 1)
    idx_low = int(idx.floor().item())
    idx_high = min(idx_low + 1, n_masks - 1)
    frac = idx - idx_low
    return (1 - frac) * sorted_preds[idx_low] + frac * sorted_preds[idx_high]


def ensemble_trimmed_upper(predictions: torch.Tensor) -> torch.Tensor:
    """Trimmed mean of upper 60% of predictions."""
    n_masks = predictions.shape[0]
    trim_low = max(1, int(n_masks * 0.4))
    sorted_preds, _ = predictions.sort(dim=0)
    return sorted_preds[trim_low:].mean(dim=0)


def ensemble_quantile_blend_60_80(predictions: torch.Tensor) -> torch.Tensor:
    """Blend of 60th and 80th percentiles."""
    n_masks = predictions.shape[0]
    sorted_preds, _ = predictions.sort(dim=0)
    idx_60 = min(int(n_masks * 0.6), n_masks - 1)
    idx_80 = min(int(n_masks * 0.8), n_masks - 1)
    return 0.6 * sorted_preds[idx_60] + 0.4 * sorted_preds[idx_80]


def ensemble_shrinkage_to_upper(predictions: torch.Tensor, shrink: float = 0.5) -> torch.Tensor:
    """Shrink predictions toward the upper quartile mean."""
    n_masks = predictions.shape[0]
    sorted_preds, _ = predictions.sort(dim=0)
    q3_start = (3 * n_masks) // 4
    upper_target = sorted_preds[q3_start:].mean(dim=0)
    current_mean = predictions.mean(dim=0)
    return current_mean + shrink * (upper_target - current_mean)


def ensemble_variance_adaptive_blend(predictions: torch.Tensor) -> torch.Tensor:
    """Blend strategy based on per-position variance."""
    median_pred = predictions.median(dim=0).values
    max_pred = predictions.max(dim=0).values
    var_pred = predictions.var(dim=0)
    var_min, var_max = var_pred.min(), var_pred.max()
    var_norm = (var_pred - var_min) / (var_max - var_min + 1e-8)
    return var_norm * median_pred + (1 - var_norm) * max_pred


def ensemble_max_plus_std(predictions: torch.Tensor, factor: float = 0.5) -> torch.Tensor:
    """Max prediction plus a factor of the standard deviation."""
    max_pred = predictions.max(dim=0).values
    std_pred = predictions.std(dim=0)
    return max_pred + factor * std_pred


def ensemble_beyond_max(predictions: torch.Tensor, extend: float = 0.2) -> torch.Tensor:
    """Go beyond max by a percentage of the prediction range."""
    max_pred = predictions.max(dim=0).values
    min_pred = predictions.min(dim=0).values
    return max_pred + extend * (max_pred - min_pred)


# =============================================================================
# OBSERVATION-AWARE STRATEGIES (Consensus-Weighted Residual Correction)
# =============================================================================

def ensemble_consensus_residual(predictions: torch.Tensor, last_obs: torch.Tensor = None,
                                 exp_smooth: torch.Tensor = None,
                                 base_strength: float = 0.3,
                                 **kwargs) -> torch.Tensor:
    """
    Consensus-Weighted Residual Correction.

    pred = obs + strength * consensus * median(residuals)

    Where residuals = predictions - obs, and consensus = exp(-CV(residuals)).
    When masks strongly agree -> apply full correction.
    When masks disagree -> shrink toward observation.
    """
    n_masks, batch, seq, dim = predictions.shape

    anchor = last_obs if last_obs is not None else exp_smooth
    if anchor is None:
        return predictions.mean(dim=0)

    anchor_expanded = anchor.unsqueeze(1).expand(-1, seq, -1)  # [batch, seq, dim]

    # Compute residuals from each mask
    residuals = predictions - anchor_expanded.unsqueeze(0)  # [n_masks, batch, seq, dim]

    # Aggregate residuals using median (robust to outliers)
    median_residual = residuals.median(dim=0).values  # [batch, seq, dim]

    # Measure consensus via coefficient of variation
    residual_std = residuals.std(dim=0)  # [batch, seq, dim]
    residual_magnitude = median_residual.abs() + 1e-8
    cv = residual_std / residual_magnitude
    consensus = torch.exp(-cv)

    # Apply correction with consensus-modulated strength
    effective_strength = base_strength * consensus
    return anchor_expanded + effective_strength * median_residual


# =============================================================================
# LEARNABLE ENSEMBLE MODULES
# =============================================================================

class LearnableEnsemble(nn.Module):
    """Learnable weighted ensemble with trainable weights per mask."""
    def __init__(self, n_masks: int, **kwargs):
        super().__init__()
        self.weights = nn.Parameter(torch.ones(n_masks) / n_masks)

    def forward(self, predictions: torch.Tensor, last_obs=None, exp_smooth=None, **kwargs) -> torch.Tensor:
        weights = torch.softmax(self.weights, dim=0).view(-1, 1, 1, 1)
        return (predictions * weights).sum(dim=0)


class AttentionEnsemble(nn.Module):
    """Attention-based ensemble - learns to weight masks based on their predictions."""
    def __init__(self, hidden_dim: int = 64, **kwargs):
        super().__init__()
        self.query = nn.Linear(1, hidden_dim)
        self.key = nn.Linear(1, hidden_dim)
        self.scale = hidden_dim ** 0.5

    def forward(self, predictions: torch.Tensor, last_obs=None, exp_smooth=None, **kwargs) -> torch.Tensor:
        n_masks, batch, seq, dim = predictions.shape
        mask_features = predictions.mean(dim=(2, 3)).view(n_masks * batch, 1)
        q = self.query(mask_features).view(n_masks, batch, -1)
        k = self.key(mask_features).view(n_masks, batch, -1)
        scores = torch.einsum('nbh,mbh->nmb', q, k) / self.scale
        attn = torch.softmax(scores.mean(dim=1), dim=0).view(n_masks, batch, 1, 1)
        return (predictions * attn).sum(dim=0)


class GatedEnsemble(nn.Module):
    """Gated ensemble - learns gates for each mask based on input statistics."""
    def __init__(self, n_masks: int, input_dim: int = 4, **kwargs):
        super().__init__()
        self.gate = nn.Sequential(nn.Linear(input_dim, n_masks), nn.Softmax(dim=-1))

    def forward(self, predictions: torch.Tensor, last_obs=None, exp_smooth=None, **kwargs) -> torch.Tensor:
        n_masks, batch, seq, dim = predictions.shape
        input_stats = torch.stack([
            predictions.mean(dim=(0, 2, 3)),
            predictions.std(dim=(0, 2, 3)),
            predictions.min(dim=0).values.min(dim=1).values.min(dim=1).values,
            predictions.max(dim=0).values.max(dim=1).values.max(dim=1).values,
        ], dim=-1)
        weights = self.gate(input_stats).t().view(n_masks, batch, 1, 1)
        return (predictions * weights).sum(dim=0)


# =============================================================================
# STRATEGY REGISTRIES
# =============================================================================

def _wrap_strategy(fn):
    """Wrap a strategy function to accept unified signature."""
    def wrapped(predictions, last_obs=None, exp_smooth=None, **kwargs):
        return fn(predictions)
    return wrapped


def _wrap_strategy_with_params(fn, **default_kwargs):
    """Wrap a strategy function with default parameters."""
    def wrapped(predictions, last_obs=None, exp_smooth=None, **kwargs):
        return fn(predictions, **default_kwargs)
    return wrapped


# Non-learnable base strategies (model predictions only)
BEST_STRATEGIES = [
    'beyond_max_20', 'max_plus_std', 'max_plus_std_100', 'variance_adaptive_blend',
    'percentile_75', 'median_max_blend', 'upper_half_mean', 'biased_mean_50',
    'optimistic_weighted', 'robust_optimal', 'adaptive_percentile', 'trimmed_upper',
    'quantile_blend_60_80', 'shrinkage_upper_50',
]

_BASE_STRATEGIES: Dict[str, Callable] = {
    'beyond_max_20': _wrap_strategy_with_params(ensemble_beyond_max, extend=0.2),
    'max_plus_std': _wrap_strategy_with_params(ensemble_max_plus_std, factor=0.5),
    'max_plus_std_100': _wrap_strategy_with_params(ensemble_max_plus_std, factor=1.0),
    'variance_adaptive_blend': _wrap_strategy(ensemble_variance_adaptive_blend),
    'percentile_75': _wrap_strategy_with_params(ensemble_percentile, percentile=75.0),
    'median_max_blend': _wrap_strategy_with_params(ensemble_median_max_blend, blend=0.5),
    'upper_half_mean': _wrap_strategy(ensemble_upper_half_mean),
    'biased_mean_50': _wrap_strategy_with_params(ensemble_biased_mean, bias_factor=0.5),
    'optimistic_weighted': _wrap_strategy_with_params(ensemble_optimistic_weighted, optimism=2.0),
    'robust_optimal': _wrap_strategy(ensemble_robust_optimal),
    'adaptive_percentile': _wrap_strategy(ensemble_adaptive_percentile),
    'trimmed_upper': _wrap_strategy(ensemble_trimmed_upper),
    'quantile_blend_60_80': _wrap_strategy(ensemble_quantile_blend_60_80),
    'shrinkage_upper_50': _wrap_strategy_with_params(ensemble_shrinkage_to_upper, shrink=0.5),
}

# Non-learnable observation-aware strategies
BEST_OBS_AWARE = [
    'residual_correction_20', 'residual_correction_30', 'residual_correction_40',
]

_OBS_STRATEGIES: Dict[str, Callable] = {
    'residual_correction_20': lambda p, last_obs=None, exp_smooth=None, **kw: ensemble_consensus_residual(p, last_obs, exp_smooth, base_strength=0.2),
    'residual_correction_30': lambda p, last_obs=None, exp_smooth=None, **kw: ensemble_consensus_residual(p, last_obs, exp_smooth, base_strength=0.3),
    'residual_correction_40': lambda p, last_obs=None, exp_smooth=None, **kw: ensemble_consensus_residual(p, last_obs, exp_smooth, base_strength=0.4),
}

# Learnable ensemble modules
LEARNABLE_ENSEMBLES: Dict[str, type] = {
    'learnable': LearnableEnsemble,
    'attention': AttentionEnsemble,
    'gated': GatedEnsemble,
}

# Combined registries
ENSEMBLE_STRATEGIES: Dict[str, Callable] = {**_BASE_STRATEGIES, **_OBS_STRATEGIES}
AVAILABLE_STRATEGIES = BEST_STRATEGIES + BEST_OBS_AWARE
AVAILABLE_LEARNABLE = list(LEARNABLE_ENSEMBLES.keys())


# =============================================================================
# API FUNCTIONS
# =============================================================================

def get_ensemble_strategy(name: str) -> Callable:
    """Get ensemble strategy function by name."""
    if name not in ENSEMBLE_STRATEGIES:
        raise ValueError(f"Unknown strategy: {name}. Available: {list(ENSEMBLE_STRATEGIES.keys())}")
    return ENSEMBLE_STRATEGIES[name]


def get_learnable_ensemble(name: str, **kwargs) -> nn.Module:
    """Get learnable ensemble module by name."""
    if name not in LEARNABLE_ENSEMBLES:
        raise ValueError(f"Unknown learnable: {name}. Available: {list(LEARNABLE_ENSEMBLES.keys())}")
    return LEARNABLE_ENSEMBLES[name](**kwargs)
