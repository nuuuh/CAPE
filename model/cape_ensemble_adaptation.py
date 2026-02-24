"""
CAPE Ensemble Adaptation
========================
Combines multiple CAPE-specific adaptation strategies and selects the best one
or creates an ensemble. This is the final integrated solution.

Key strategies combined:
1. Mask selection (top-K masks based on validation performance)
2. Online bias correction (adapts as predictions are made)
3. Trend-following (for trending data)
4. Uncertainty-weighted ensemble (trust low-uncertainty predictions more)
"""

import numpy as np
import torch
import torch.nn as nn
from collections import deque
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass, field
from sklearn.metrics import mean_squared_error
from tqdm import tqdm


@dataclass
class EnsembleAdaptConfig:
    """Configuration for ensemble adaptation."""
    num_masks: int = 50  # More masks for better selection
    top_k_masks: int = 15  # Keep top K performing masks
    memory_size: int = 30
    bias_momentum: float = 0.95  # Exponential smoothing for bias
    uncertainty_threshold: float = 0.3  # When to use uncertainty weighting
    use_trend_adjustment: bool = True
    trend_weight: float = 0.15  # Weight for trend continuation


class CAPEEnsembleAdapter:
    """
    Ensemble adapter combining multiple CAPE-specific strategies.
    """
    
    def __init__(self, model, device, config: EnsembleAdaptConfig = None):
        self.model = model
        self.device = device
        self.config = config or EnsembleAdaptConfig()
        
        # Generate diverse masks
        torch.manual_seed(42)
        self.all_masks = []
        for _ in range(self.config.num_masks):
            mask = torch.zeros(12, device=device, dtype=torch.bool)
            mask[:2] = True  # S, I always on
            # Vary the probability of other compartments
            prob = 0.3 + 0.4 * torch.rand(1).item()  # Random between 0.3 and 0.7
            mask[2:] = torch.rand(10, device=device) > prob
            self.all_masks.append(mask)
        
        # Selected best masks (after calibration)
        self.selected_masks = None
        self.mask_weights = None
        
        # Online tracking
        self.bias_estimate = 0.0
        self.error_history = deque(maxlen=self.config.memory_size)
        self.pred_history = deque(maxlen=self.config.memory_size)
        self.target_history = deque(maxlen=self.config.memory_size)
        
        # Trend tracking
        self.recent_trend = 0.0
        
    def _get_predictions_for_masks(self, x: torch.Tensor, num_output: int, 
                                    masks: List) -> np.ndarray:
        """Get predictions from a list of masks."""
        all_preds = []
        
        with torch.no_grad():
            for mask in masks:
                curr = x.clone()
                preds = []
                for _ in range(num_output):
                    out = self.model(curr, compartment_mask=mask)
                    next_tok = out['I'][:, -1:, :]
                    preds.append(next_tok)
                    curr = torch.cat([curr[:, 1:, :], next_tok], dim=1)
                
                pred = torch.cat(preds, dim=1).cpu().numpy()
                all_preds.append(pred)
        
        return np.stack(all_preds, axis=0)  # (num_masks, batch, num_output, token_size)
    
    def calibrate(self, valid_loader, num_output: int = 1):
        """
        Calibrate on validation data to select best masks and initial parameters.
        """
        self.model.eval()
        print("  [Ensemble] Calibrating with diverse mask pool...")
        
        # Collect all predictions for each mask
        mask_predictions = [[] for _ in range(len(self.all_masks))]
        all_targets = []
        all_inputs = []
        
        for batch in valid_loader:
            x, y = batch['input'].to(self.device), batch['label']
            all_inputs.append(x.cpu().numpy())
            all_targets.append(y.numpy())
            
            preds = self._get_predictions_for_masks(x, num_output, self.all_masks)
            for i in range(len(self.all_masks)):
                mask_predictions[i].append(preds[i])
        
        targets = np.concatenate(all_targets, axis=0)
        inputs = np.concatenate(all_inputs, axis=0)
        
        for i in range(len(self.all_masks)):
            mask_predictions[i] = np.concatenate(mask_predictions[i], axis=0)
        
        # Compute MSE for each mask
        mask_mses = []
        for i in range(len(self.all_masks)):
            mse = mean_squared_error(targets.flatten(), mask_predictions[i].flatten())
            mask_mses.append(mse)
        
        mask_mses = np.array(mask_mses)
        
        # Select top K masks
        sorted_indices = np.argsort(mask_mses)
        self.selected_masks = [self.all_masks[i] for i in sorted_indices[:self.config.top_k_masks]]
        selected_mses = mask_mses[sorted_indices[:self.config.top_k_masks]]
        
        # Compute weights (inverse MSE with temperature)
        temperature = 0.5  # Lower = more weight to best masks
        inv_mses = 1.0 / (selected_mses + 1e-6)
        self.mask_weights = np.exp(inv_mses / temperature)
        self.mask_weights /= self.mask_weights.sum()
        
        print(f"  [Ensemble] Selected {len(self.selected_masks)} masks")
        print(f"  [Ensemble] MSE range: {selected_mses[0]:.4f} - {selected_mses[-1]:.4f}")
        print(f"  [Ensemble] Weight range: {self.mask_weights.min():.4f} - {self.mask_weights.max():.4f}")
        
        # Initialize bias from validation errors
        all_preds = self._get_predictions_for_masks(
            torch.tensor(inputs, device=self.device), num_output, self.selected_masks
        )
        
        # Weighted ensemble prediction
        weighted_pred = np.zeros_like(all_preds[0])
        for i, w in enumerate(self.mask_weights):
            weighted_pred += w * all_preds[i]
        
        # Compute initial bias
        initial_error = targets - weighted_pred
        self.bias_estimate = initial_error.mean()
        print(f"  [Ensemble] Initial bias: {self.bias_estimate:.4f}")
        
        # Analyze trend in validation
        if len(inputs) > 1:
            input_means = [inp.flatten()[-4:].mean() for inp in inputs]
            trend = np.diff(input_means).mean()
            self.recent_trend = trend
            print(f"  [Ensemble] Detected trend: {self.recent_trend:.4f}")
    
    def predict(self, x: torch.Tensor, num_output: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make prediction using calibrated ensemble.
        """
        batch_size = x.shape[0]
        
        # Get predictions from selected masks
        all_preds = self._get_predictions_for_masks(x, num_output, self.selected_masks)
        
        # Weighted ensemble
        weighted_pred = np.zeros((batch_size, all_preds.shape[2], all_preds.shape[3]))
        for i, w in enumerate(self.mask_weights):
            weighted_pred += w * all_preds[i]
        
        # Uncertainty from mask variance
        uncertainty = all_preds.std(axis=0)
        
        # Apply bias correction
        weighted_pred = weighted_pred + self.bias_estimate
        
        # Apply trend adjustment for low-uncertainty predictions
        if self.config.use_trend_adjustment:
            x_np = x.cpu().numpy()
            for b in range(batch_size):
                # Compute local trend from input
                inp = x_np[b]
                if len(inp) >= 2:
                    local_trend = (inp[-1] - inp[-2]).mean()
                    
                    # Weight trend by inverse uncertainty
                    sample_uncertainty = uncertainty[b].mean()
                    if sample_uncertainty < self.config.uncertainty_threshold:
                        trend_adj = local_trend * self.config.trend_weight
                        weighted_pred[b] = weighted_pred[b] + trend_adj
        
        return weighted_pred, uncertainty
    
    def predict_and_update(self, x: torch.Tensor, y: np.ndarray, 
                            num_output: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make prediction and update online parameters.
        """
        pred, uncertainty = self.predict(x, num_output)
        
        # Update bias with exponential smoothing
        error = y - pred
        batch_error = error.mean()
        
        self.bias_estimate = (
            self.config.bias_momentum * self.bias_estimate + 
            (1 - self.config.bias_momentum) * batch_error
        )
        
        # Track for analysis
        self.error_history.append(batch_error)
        
        return pred, uncertainty


def evaluate_with_ensemble_adaptation(model, valid_loader, test_loader,
                                       num_output: int, device: str,
                                       config: EnsembleAdaptConfig = None):
    """
    Evaluate CAPE with ensemble adaptation.
    """
    config = config or EnsembleAdaptConfig()
    
    adapter = CAPEEnsembleAdapter(model, device, config)
    
    # Calibrate
    print("  [Ensemble] Calibrating on validation data...")
    adapter.calibrate(valid_loader, num_output)
    
    # Evaluate with online updates
    print("  [Ensemble] Evaluating on test data...")
    all_preds, all_stds, all_targets = [], [], []
    
    for batch in tqdm(test_loader, desc="Testing"):
        x, y = batch['input'].to(device), batch['label'].numpy()
        
        # Predict first (before seeing target)
        pred, std = adapter.predict(x, num_output)
        
        all_preds.append(pred)
        all_stds.append(std)
        all_targets.append(y)
        
        # Update with true target
        adapter.predict_and_update(x, y, num_output)
    
    preds = np.concatenate(all_preds)
    stds = np.concatenate(all_stds)
    targets = np.concatenate(all_targets)
    
    mse = mean_squared_error(targets.flatten(), preds.flatten())
    mae = np.mean(np.abs(targets.flatten() - preds.flatten()))
    
    print(f"  [Ensemble] Final bias: {adapter.bias_estimate:.4f}")
    
    return preds, stds, targets, {'mse': mse, 'mae': mae}
