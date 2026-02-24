"""
CAPE Online Adaptation
======================
This method adapts CAPE's predictions in an online manner by learning from
each prediction error and adjusting future predictions.

Key CAPE-specific features:
1. Mask selection adapts based on recent error feedback
2. Uses uncertainty to modulate adaptation rate
3. Learns pattern-specific corrections online

CHRONOS cannot use this because:
- CHRONOS has a fixed model with no mask ensemble
- Cannot select different "views" of the prediction
- Cannot use per-mask error feedback
"""

import numpy as np
import torch
import torch.nn as nn
from collections import deque
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass
from sklearn.metrics import mean_squared_error
from tqdm import tqdm


@dataclass
class OnlineAdaptConfig:
    """Configuration for online adaptation."""
    num_masks: int = 30
    memory_size: int = 20  # How many recent samples to remember
    adapt_rate: float = 0.1  # Learning rate for adaptation
    use_mask_reweighting: bool = True  # Adapt mask weights online
    use_bias_correction: bool = True  # Correct systematic bias


class CAPEOnlineAdapter:
    """
    Online adaptation for CAPE predictions.
    
    Adapts by:
    1. Tracking recent prediction errors
    2. Reweighting masks based on their performance
    3. Applying a learned bias correction
    """
    
    def __init__(self, model, device, config: OnlineAdaptConfig = None):
        self.model = model
        self.device = device
        self.config = config or OnlineAdaptConfig()
        
        # Memory for online learning
        self.error_memory = deque(maxlen=self.config.memory_size)
        self.prediction_memory = deque(maxlen=self.config.memory_size)
        self.target_memory = deque(maxlen=self.config.memory_size)
        
        # Mask performance tracking
        self.mask_errors = {}  # mask_idx -> list of errors
        self.mask_weights = np.ones(self.config.num_masks) / self.config.num_masks
        
        # Bias correction
        self.bias_estimate = 0.0
        self.bias_momentum = 0.9
        
        # Fixed masks
        torch.manual_seed(42)
        self.fixed_masks = []
        for _ in range(self.config.num_masks):
            mask = torch.zeros(12, device=device, dtype=torch.bool)
            mask[:2] = True
            mask[2:] = torch.rand(10, device=device) > 0.5
            self.fixed_masks.append(mask)
    
    def _get_mask_predictions(self, x: torch.Tensor, num_output: int) -> np.ndarray:
        """Get predictions from all masks."""
        all_preds = []
        
        with torch.no_grad():
            for mask in self.fixed_masks:
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
    
    def predict_and_update(self, x: torch.Tensor, y: torch.Tensor, 
                            num_output: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make prediction and update based on true target.
        
        This is for "online" evaluation where we see the true label after prediction.
        
        Args:
            x: Input (batch, num_input, token_size)
            y: True target (batch, num_output, token_size)
            num_output: Number of output tokens
            
        Returns:
            prediction: Adapted prediction
            uncertainty: Prediction uncertainty
        """
        self.model.eval()
        batch_size = x.shape[0]
        
        # Get predictions from all masks
        all_preds = self._get_mask_predictions(x, num_output)  # (num_masks, batch, num_output, token_size)
        
        # Apply mask weights
        weighted_preds = np.zeros((batch_size, all_preds.shape[2], all_preds.shape[3]))
        for i, w in enumerate(self.mask_weights):
            weighted_preds += w * all_preds[i]
        
        # Apply bias correction
        if self.config.use_bias_correction:
            weighted_preds = weighted_preds + self.bias_estimate
        
        # Uncertainty
        uncertainty = all_preds.std(axis=0)
        
        # Update based on true target (if available)
        if y is not None:
            y_np = y.numpy() if isinstance(y, torch.Tensor) else y
            
            for b in range(batch_size):
                # Compute error
                error = y_np[b] - weighted_preds[b]
                self.error_memory.append(error.mean())
                
                # Update bias estimate
                if self.config.use_bias_correction:
                    self.bias_estimate = (
                        self.bias_momentum * self.bias_estimate + 
                        (1 - self.bias_momentum) * error.mean()
                    )
                
                # Update mask weights based on individual mask performance
                if self.config.use_mask_reweighting:
                    for mask_idx in range(len(self.fixed_masks)):
                        mask_pred = all_preds[mask_idx, b]
                        mask_error = np.abs(y_np[b] - mask_pred).mean()
                        
                        if mask_idx not in self.mask_errors:
                            self.mask_errors[mask_idx] = deque(maxlen=self.config.memory_size)
                        self.mask_errors[mask_idx].append(mask_error)
                    
                    # Recompute mask weights
                    avg_errors = []
                    for mask_idx in range(len(self.fixed_masks)):
                        if mask_idx in self.mask_errors and len(self.mask_errors[mask_idx]) > 0:
                            avg_errors.append(np.mean(self.mask_errors[mask_idx]))
                        else:
                            avg_errors.append(1.0)
                    
                    avg_errors = np.array(avg_errors)
                    # Inverse error weighting with softmax
                    inv_errors = 1.0 / (avg_errors + 1e-6)
                    self.mask_weights = inv_errors / inv_errors.sum()
        
        return weighted_preds, uncertainty
    
    def predict_only(self, x: torch.Tensor, num_output: int) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make prediction without update (for final evaluation).
        """
        self.model.eval()
        batch_size = x.shape[0]
        
        all_preds = self._get_mask_predictions(x, num_output)
        
        # Apply current mask weights
        weighted_preds = np.zeros((batch_size, all_preds.shape[2], all_preds.shape[3]))
        for i, w in enumerate(self.mask_weights):
            weighted_preds += w * all_preds[i]
        
        # Apply bias correction
        if self.config.use_bias_correction:
            weighted_preds = weighted_preds + self.bias_estimate
        
        uncertainty = all_preds.std(axis=0)
        
        return weighted_preds, uncertainty


def evaluate_with_online_adaptation(model, valid_loader, test_loader,
                                     num_output: int, device: str,
                                     config: OnlineAdaptConfig = None):
    """
    Evaluate CAPE with online adaptation.
    
    Uses validation set for initial "burn-in" phase to learn initial parameters,
    then continues adapting on test set.
    
    Args:
        model: CAPE model
        valid_loader: Validation data for initial adaptation
        test_loader: Test data for evaluation
        num_output: Number of output tokens
        device: Device
        config: Online adaptation config
        
    Returns:
        preds, stds, targets, metrics
    """
    config = config or OnlineAdaptConfig()
    
    adapter = CAPEOnlineAdapter(model, device, config)
    
    # Burn-in on validation set
    print("  [Online] Burn-in phase on validation data...")
    for batch in tqdm(valid_loader, desc="Burn-in"):
        x, y = batch['input'].to(device), batch['label']
        adapter.predict_and_update(x, y, num_output)
    
    print(f"  [Online] After burn-in: bias={adapter.bias_estimate:.4f}")
    print(f"  [Online] Top 5 mask weights: {sorted(adapter.mask_weights)[-5:]}")
    
    # Evaluate on test set with continued adaptation
    print("  [Online] Evaluating on test data with continued adaptation...")
    all_preds, all_stds, all_targets = [], [], []
    
    for batch in tqdm(test_loader, desc="Testing"):
        x, y = batch['input'].to(device), batch['label']
        
        # Make prediction (before seeing target)
        pred, std = adapter.predict_only(x, num_output)
        
        all_preds.append(pred)
        all_stds.append(std)
        all_targets.append(y.numpy())
        
        # Update with true target
        adapter.predict_and_update(x, y, num_output)
    
    preds = np.concatenate(all_preds)
    stds = np.concatenate(all_stds)
    targets = np.concatenate(all_targets)
    
    mse = mean_squared_error(targets.flatten(), preds.flatten())
    mae = np.mean(np.abs(targets.flatten() - preds.flatten()))
    
    print(f"  [Online] Final bias: {adapter.bias_estimate:.4f}")
    
    return preds, stds, targets, {'mse': mse, 'mae': mae}
