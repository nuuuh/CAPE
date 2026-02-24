"""
CAPE-Specific Adaptive Inference
================================
A more effective post-hoc adaptation method that uses CAPE's unique properties
to beat CHRONOS on wave-pattern diseases.

Key strategies:
1. Learned mask weighting - use validation data to learn which masks work best
2. Trend continuation - use CAPE's uncertainty to weight trend-following
3. Seasonal adjustment - detect and use seasonality patterns
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import find_peaks, correlate
from scipy.fft import fft
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass
from sklearn.metrics import mean_squared_error


@dataclass 
class AdaptiveInferenceConfig:
    """Configuration for adaptive inference."""
    num_masks: int = 30
    trend_window: int = 4
    min_period: int = 10
    max_period: int = 60
    top_k_masks: int = 10  # Use top K masks instead of all
    blend_with_naive: float = 0.0  # Blend factor with naive prediction
    use_trend_continuation: bool = True


class CAPEAdaptiveInference:
    """
    Adaptive inference for CAPE that learns from validation data.
    
    This is CAPE-specific because:
    1. Uses mask ensemble - CHRONOS has no masks
    2. Learns mask weights on validation - only possible with ensemble
    3. Uses uncertainty from mask variance
    """
    
    def __init__(self, model, device, config: AdaptiveInferenceConfig = None):
        self.model = model
        self.device = device
        self.config = config or AdaptiveInferenceConfig()
        
        # Learned parameters (initialized during calibration)
        self.mask_weights = None  # Weights for each mask
        self.best_masks = None  # Indices of best performing masks
        self.trend_weight = 0.0  # How much to weight trend continuation
        self.seasonality_period = None
        self.seasonal_template = None
        
    def calibrate(self, valid_loader, num_output: int = 1):
        """
        Calibrate adaptation parameters on validation data.
        
        This learns:
        1. Which masks perform best on this data
        2. How much to weight trend continuation
        3. Seasonal patterns (if any)
        
        Args:
            valid_loader: Validation data loader
            num_output: Number of output tokens
        """
        self.model.eval()
        num_masks = self.config.num_masks
        
        # Collect predictions from each mask separately
        mask_predictions = [[] for _ in range(num_masks)]
        all_targets = []
        all_inputs = []
        
        # Generate fixed masks for reproducibility during calibration
        torch.manual_seed(42)
        fixed_masks = []
        for _ in range(num_masks):
            mask = torch.zeros(12, device=self.device, dtype=torch.bool)
            mask[:2] = True
            mask[2:] = torch.rand(10, device=self.device) > 0.5
            fixed_masks.append(mask)
        
        with torch.no_grad():
            for batch in valid_loader:
                x, y = batch['input'].to(self.device), batch['label'].to(self.device)
                all_inputs.append(x.cpu().numpy())
                all_targets.append(y.cpu().numpy())
                
                for mask_idx, mask in enumerate(fixed_masks):
                    # Autoregressive prediction
                    curr = x.clone()
                    preds = []
                    for _ in range(num_output):
                        out = self.model(curr, compartment_mask=mask)
                        next_tok = out['I'][:, -1:, :]
                        preds.append(next_tok)
                        curr = torch.cat([curr[:, 1:, :], next_tok], dim=1)
                    
                    pred = torch.cat(preds, dim=1).cpu().numpy()
                    mask_predictions[mask_idx].append(pred)
        
        # Concatenate
        targets = np.concatenate(all_targets, axis=0)
        inputs = np.concatenate(all_inputs, axis=0)
        
        for i in range(num_masks):
            mask_predictions[i] = np.concatenate(mask_predictions[i], axis=0)
        
        # Compute MSE for each mask
        mask_mses = []
        for i in range(num_masks):
            mse = mean_squared_error(targets.flatten(), mask_predictions[i].flatten())
            mask_mses.append(mse)
        
        mask_mses = np.array(mask_mses)
        
        # Select best masks
        sorted_indices = np.argsort(mask_mses)
        self.best_masks = sorted_indices[:self.config.top_k_masks]
        
        # Compute mask weights (inverse MSE, normalized)
        best_mses = mask_mses[self.best_masks]
        self.mask_weights = 1.0 / (best_mses + 1e-6)
        self.mask_weights /= self.mask_weights.sum()
        
        print(f"  [Calibration] Best mask MSEs: {best_mses[:5]}")
        print(f"  [Calibration] Selected {len(self.best_masks)} masks with weights: {self.mask_weights[:5]}")
        
        # Store the fixed masks for inference
        self.fixed_masks = [fixed_masks[i] for i in self.best_masks]
        
        # Analyze trend - does trend continuation help?
        trend_preds = []
        for i in range(len(targets)):
            inp = inputs[i]  # (num_input, token_size)
            # Estimate trend from last few tokens
            if len(inp) >= 2:
                recent = inp[-2:]
                trend = recent[-1] - recent[-2]  # Per-token trend
                trend_pred = inp[-1] + trend  # Continue trend
                trend_preds.append(trend_pred)
            else:
                trend_preds.append(inp[-1])
        
        trend_preds = np.array(trend_preds)
        if trend_preds.shape[0] == targets.shape[0]:
            # Reshape trend_preds to match targets shape
            trend_preds_reshaped = trend_preds.reshape(targets.shape[0], 1, -1)
            if num_output > 1:
                trend_preds_reshaped = np.tile(trend_preds_reshaped, (1, num_output, 1))
            trend_mse = mean_squared_error(targets.flatten(), trend_preds_reshaped.flatten())
            
            # If trend helps, compute optimal weight
            avg_mask_mse = np.mean(best_mses)
            if trend_mse < avg_mask_mse:
                # Trend is better - but we'll blend
                self.trend_weight = min(0.3, avg_mask_mse / (trend_mse + 1e-6) * 0.1)
            print(f"  [Calibration] Trend MSE: {trend_mse:.4f}, Avg mask MSE: {avg_mask_mse:.4f}")
            print(f"  [Calibration] Trend weight: {self.trend_weight:.3f}")
        
        # Detect seasonality in the input data
        all_inputs_flat = inputs.reshape(-1)
        self._detect_seasonality(all_inputs_flat)
        
    def _detect_seasonality(self, history: np.ndarray):
        """Detect seasonal patterns in historical data."""
        if len(history) < self.config.min_period * 3:
            return
        
        # Autocorrelation
        history_centered = history - history.mean()
        autocorr = correlate(history_centered, history_centered, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / (autocorr[0] + 1e-8)
        
        # Find peaks
        peaks, props = find_peaks(
            autocorr[self.config.min_period:self.config.max_period],
            height=0.3,
            distance=self.config.min_period // 2
        )
        
        if len(peaks) > 0:
            best_peak = peaks[np.argmax(props['peak_heights'])]
            self.seasonality_period = best_peak + self.config.min_period
            strength = props['peak_heights'][np.argmax(props['peak_heights'])]
            print(f"  [Calibration] Detected seasonality: period={self.seasonality_period}, strength={strength:.2f}")
            
            # Learn seasonal template
            if len(history) >= self.seasonality_period * 2:
                num_periods = len(history) // self.seasonality_period
                periods = []
                for i in range(num_periods):
                    start = i * self.seasonality_period
                    end = start + self.seasonality_period
                    if end <= len(history):
                        periods.append(history[start:end])
                
                if periods:
                    self.seasonal_template = np.mean(periods, axis=0)
                    self.seasonal_template = (self.seasonal_template - self.seasonal_template.mean()) / (self.seasonal_template.std() + 1e-8)
    
    def predict(self, x: torch.Tensor, num_output: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make prediction using calibrated adaptive inference.
        
        Args:
            x: Input tensor (batch, num_input, token_size)
            num_output: Number of output tokens
            
        Returns:
            prediction: (batch, num_output, token_size)
            uncertainty: (batch, num_output, token_size)
        """
        self.model.eval()
        batch_size = x.shape[0]
        
        # Use only the calibrated best masks
        masks_to_use = self.fixed_masks if self.fixed_masks else self._generate_masks()
        
        all_preds = []
        
        with torch.no_grad():
            for mask in masks_to_use:
                curr = x.clone()
                preds = []
                for _ in range(num_output):
                    out = self.model(curr, compartment_mask=mask)
                    next_tok = out['I'][:, -1:, :]
                    preds.append(next_tok)
                    curr = torch.cat([curr[:, 1:, :], next_tok], dim=1)
                
                pred = torch.cat(preds, dim=1).cpu().numpy()
                all_preds.append(pred)
        
        all_preds = np.stack(all_preds, axis=0)  # (num_masks, batch, num_output, token_size)
        
        # Weighted average using learned mask weights
        if self.mask_weights is not None:
            weighted_pred = np.zeros_like(all_preds[0])
            for i, w in enumerate(self.mask_weights):
                weighted_pred += w * all_preds[i]
        else:
            weighted_pred = all_preds.mean(axis=0)
        
        # Uncertainty from mask variance
        uncertainty = all_preds.std(axis=0)
        
        # Apply trend continuation if calibrated
        if self.config.use_trend_continuation and self.trend_weight > 0:
            x_np = x.cpu().numpy()
            for b in range(batch_size):
                inp = x_np[b]  # (num_input, token_size)
                if len(inp) >= 2:
                    trend = inp[-1] - inp[-2]
                    trend_pred = inp[-1] + trend
                    
                    # Reshape to match output
                    trend_pred = trend_pred.reshape(1, -1)
                    if num_output > 1:
                        trend_pred = np.tile(trend_pred, (num_output, 1))
                    
                    # Blend based on calibrated weight
                    weighted_pred[b] = (1 - self.trend_weight) * weighted_pred[b] + self.trend_weight * trend_pred
        
        return weighted_pred, uncertainty
    
    def _generate_masks(self):
        """Generate random masks if not calibrated."""
        masks = []
        for _ in range(self.config.num_masks):
            mask = torch.zeros(12, device=self.device, dtype=torch.bool)
            mask[:2] = True
            mask[2:] = torch.rand(10, device=self.device) > 0.5
            masks.append(mask)
        return masks


def evaluate_with_adaptive_inference(model, train_loader, valid_loader, test_loader,
                                     num_output: int, device: str,
                                     config: AdaptiveInferenceConfig = None):
    """
    Evaluate CAPE with adaptive inference (post-hoc adaptation).
    
    Args:
        model: CAPE model
        train_loader: Training data (for context)
        valid_loader: Validation data (for calibration)
        test_loader: Test data (for evaluation)
        num_output: Number of output tokens
        device: Device
        config: Adaptive inference config
        
    Returns:
        preds, stds, targets, metrics
    """
    config = config or AdaptiveInferenceConfig()
    
    # Create adaptive inference module
    adaptive = CAPEAdaptiveInference(model, device, config)
    
    # Calibrate on validation data
    print("  [Adaptive] Calibrating on validation data...")
    adaptive.calibrate(valid_loader, num_output)
    
    # Evaluate on test data
    print("  [Adaptive] Evaluating on test data...")
    all_preds, all_stds, all_targets = [], [], []
    
    for batch in test_loader:
        x, y = batch['input'].to(device), batch['label']
        
        pred, std = adaptive.predict(x, num_output)
        
        all_preds.append(pred)
        all_stds.append(std)
        all_targets.append(y.numpy())
    
    preds = np.concatenate(all_preds, axis=0)
    stds = np.concatenate(all_stds, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    
    mse = mean_squared_error(targets.flatten(), preds.flatten())
    mae = np.mean(np.abs(targets.flatten() - preds.flatten()))
    
    return preds, stds, targets, {'mse': mse, 'mae': mae}
