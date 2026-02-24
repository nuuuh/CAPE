"""
CAPE Residual Correction Adapter
================================
This adapter learns a correction model on validation data to fix systematic
biases in CAPE's predictions. Only possible for CAPE due to its mask ensemble.

Key idea:
1. Generate CAPE predictions on validation set
2. Learn a correction model (residual predictor) from errors
3. Apply correction to test predictions

This is CAPE-specific because:
- Uses mask ensemble variance as a feature
- Learns mask-specific corrections
- Uses compartmental outputs as features (if available)
"""

import numpy as np
import torch
import torch.nn as nn
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error
from typing import Tuple, Dict, Optional, List
from dataclasses import dataclass
from tqdm import tqdm


@dataclass
class ResidualCorrectionConfig:
    """Configuration for residual correction."""
    num_masks: int = 30
    correction_alpha: float = 1.0  # Ridge regularization
    use_trend_features: bool = True
    use_uncertainty_features: bool = True
    max_correction: float = 0.5  # Max correction as fraction of prediction


class CAPEResidualCorrector:
    """
    Learn and apply corrections to CAPE predictions.
    
    Features used for correction:
    1. Input statistics (mean, std, trend)
    2. CAPE prediction variance (uncertainty)
    3. Mask-specific predictions
    4. Seasonal position
    """
    
    def __init__(self, model, device, config: ResidualCorrectionConfig = None):
        self.model = model
        self.device = device
        self.config = config or ResidualCorrectionConfig()
        
        self.correction_model = None
        self.feature_scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        
        # Fixed masks for reproducibility
        torch.manual_seed(42)
        self.fixed_masks = []
        for _ in range(self.config.num_masks):
            mask = torch.zeros(12, device=device, dtype=torch.bool)
            mask[:2] = True
            mask[2:] = torch.rand(10, device=device) > 0.5
            self.fixed_masks.append(mask)
    
    def _extract_features(self, x: np.ndarray, mask_preds: np.ndarray, mask_std: np.ndarray) -> np.ndarray:
        """
        Extract features for correction model.
        
        Args:
            x: Input sequence (num_input, token_size)
            mask_preds: Predictions from each mask (num_masks, num_output, token_size)
            mask_std: Uncertainty (num_output, token_size)
            
        Returns:
            features: Feature vector for this sample
        """
        features = []
        
        # Input statistics
        features.append(x.mean())
        features.append(x.std())
        features.append(x[-1].mean())  # Last token mean
        
        # Trend features
        if self.config.use_trend_features and len(x) >= 2:
            trend = (x[-1] - x[-2]).mean()
            features.append(trend)
            
            # Acceleration
            if len(x) >= 3:
                accel = ((x[-1] - x[-2]) - (x[-2] - x[-3])).mean()
                features.append(accel)
            else:
                features.append(0.0)
        else:
            features.extend([0.0, 0.0])
        
        # Uncertainty features
        if self.config.use_uncertainty_features:
            features.append(mask_std.mean())
            features.append(mask_std.max())
            features.append(mask_std.min())
        else:
            features.extend([0.0, 0.0, 0.0])
        
        # Mask prediction statistics
        features.append(mask_preds.mean())
        features.append(mask_preds.std())
        
        # Ratio of prediction to input
        if abs(x[-1].mean()) > 1e-6:
            features.append(mask_preds.mean() / x[-1].mean())
        else:
            features.append(1.0)
        
        return np.array(features, dtype=np.float32)
    
    def _get_mask_predictions(self, x: torch.Tensor, num_output: int) -> Tuple[np.ndarray, np.ndarray]:
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
        
        all_preds = np.stack(all_preds, axis=0)  # (num_masks, batch, num_output, token_size)
        mean_pred = all_preds.mean(axis=0)
        std_pred = all_preds.std(axis=0)
        
        return all_preds, mean_pred, std_pred
    
    def calibrate(self, valid_loader, num_output: int = 1):
        """
        Learn correction model on validation data.
        
        Args:
            valid_loader: Validation data loader
            num_output: Number of output tokens
        """
        self.model.eval()
        
        all_features = []
        all_residuals = []
        
        print("  [Residual] Collecting validation predictions...")
        
        for batch in tqdm(valid_loader, desc="Calibrating"):
            x, y = batch['input'].to(self.device), batch['label'].numpy()
            batch_size = x.shape[0]
            
            # Get mask predictions
            all_preds, mean_pred, std_pred = self._get_mask_predictions(x, num_output)
            x_np = x.cpu().numpy()
            
            for i in range(batch_size):
                # Extract features
                features = self._extract_features(
                    x_np[i],
                    all_preds[:, i],
                    std_pred[i]
                )
                all_features.append(features)
                
                # Compute residual (error)
                residual = y[i].flatten() - mean_pred[i].flatten()
                all_residuals.append(residual.mean())  # Predict scalar correction
        
        # Train correction model
        X = np.array(all_features)
        y = np.array(all_residuals)
        
        # Scale features
        X_scaled = self.feature_scaler.fit_transform(X)
        
        # Ridge regression for robustness
        self.correction_model = Ridge(alpha=self.config.correction_alpha)
        self.correction_model.fit(X_scaled, y)
        
        # Evaluate on training data
        y_pred = self.correction_model.predict(X_scaled)
        train_r2 = 1 - np.sum((y - y_pred)**2) / (np.sum((y - y.mean())**2) + 1e-8)
        
        print(f"  [Residual] Correction model R²: {train_r2:.4f}")
        print(f"  [Residual] Mean residual: {y.mean():.4f}, Std: {y.std():.4f}")
        print(f"  [Residual] Model coefficients: {self.correction_model.coef_[:5]}")
        
    def predict(self, x: torch.Tensor, num_output: int = 1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Make corrected predictions.
        
        Args:
            x: Input tensor (batch, num_input, token_size)
            num_output: Number of output tokens
            
        Returns:
            corrected_pred: Corrected prediction
            uncertainty: Mask variance
        """
        batch_size = x.shape[0]
        
        # Get mask predictions
        all_preds, mean_pred, std_pred = self._get_mask_predictions(x, num_output)
        x_np = x.cpu().numpy()
        
        corrected_preds = []
        
        for i in range(batch_size):
            # Extract features
            features = self._extract_features(x_np[i], all_preds[:, i], std_pred[i])
            features_scaled = self.feature_scaler.transform(features.reshape(1, -1))
            
            # Predict correction
            if self.correction_model is not None:
                correction = self.correction_model.predict(features_scaled)[0]
                
                # Clip correction to prevent extreme adjustments
                max_corr = abs(mean_pred[i].mean()) * self.config.max_correction
                correction = np.clip(correction, -max_corr, max_corr)
            else:
                correction = 0.0
            
            # Apply correction
            corrected = mean_pred[i] + correction
            corrected_preds.append(corrected)
        
        return np.array(corrected_preds), std_pred


def evaluate_with_residual_correction(model, valid_loader, test_loader,
                                       num_output: int, device: str,
                                       config: ResidualCorrectionConfig = None):
    """
    Evaluate CAPE with residual correction.
    
    Args:
        model: CAPE model
        valid_loader: Validation data for calibration
        test_loader: Test data for evaluation
        num_output: Number of output tokens
        device: Device
        config: Correction config
        
    Returns:
        preds, stds, targets, metrics
    """
    config = config or ResidualCorrectionConfig()
    
    corrector = CAPEResidualCorrector(model, device, config)
    
    # Calibrate on validation
    print("  [Correction] Calibrating on validation data...")
    corrector.calibrate(valid_loader, num_output)
    
    # Evaluate on test
    print("  [Correction] Evaluating on test data...")
    all_preds, all_stds, all_targets = [], [], []
    
    for batch in tqdm(test_loader, desc="Testing"):
        x, y = batch['input'].to(device), batch['label'].numpy()
        
        pred, std = corrector.predict(x, num_output)
        
        all_preds.append(pred)
        all_stds.append(std)
        all_targets.append(y)
    
    preds = np.concatenate(all_preds)
    stds = np.concatenate(all_stds)
    targets = np.concatenate(all_targets)
    
    mse = mean_squared_error(targets.flatten(), preds.flatten())
    mae = np.mean(np.abs(targets.flatten() - preds.flatten()))
    
    return preds, stds, targets, {'mse': mse, 'mae': mae}
