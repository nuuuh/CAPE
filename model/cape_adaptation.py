"""
CAPE-Specific Post-Hoc Adaptation Methods
==========================================
These methods are specifically designed for CAPE and exploit its unique properties:
1. Mask-based ensemble predictions with uncertainty
2. Compartmental model structure (S, I, R, etc.)
3. Multi-output token predictions

CHRONOS cannot use these methods because:
- CHRONOS is a general foundation model without compartment structure
- CHRONOS doesn't have mask-based uncertainty quantification
- CHRONOS doesn't have compartment-specific predictions

Key methods:
1. WavePatternAdapter: Detects periodicity in data and adjusts predictions
2. UncertaintyWeightedAdapter: Uses CAPE's uncertainty to weight predictions
3. CompartmentPhaseAdapter: Uses compartment ratios to detect epidemic phase
"""

import numpy as np
import torch
import torch.nn as nn
from scipy.signal import find_peaks, correlate
from scipy.fft import fft, ifft
from typing import Tuple, List, Dict, Optional
from dataclasses import dataclass


@dataclass
class AdaptationConfig:
    """Configuration for post-hoc adaptation."""
    # Wave pattern detection
    min_period: int = 10  # Minimum period to detect (in time steps)
    max_period: int = 60  # Maximum period to detect
    periodicity_threshold: float = 0.3  # Autocorrelation threshold for periodicity
    
    # Uncertainty weighting
    uncertainty_scale: float = 1.0  # How much to weight by uncertainty
    
    # Phase adaptation
    phase_window: int = 8  # Window for phase estimation
    
    # Ensemble adaptation
    top_k_masks: int = 5  # Number of best masks to use


class CAPEPostHocAdapter:
    """
    Post-hoc adaptation for CAPE predictions that CHRONOS cannot use.
    
    Key insight: CAPE generates multiple predictions via mask ensemble.
    We can use the historical performance of different masks to select
    the best masks for different patterns (wave vs stable).
    """
    
    def __init__(self, config: AdaptationConfig = None):
        self.config = config or AdaptationConfig()
        self.detected_period = None
        self.mask_performance = {}  # Track which masks work best
        self.historical_errors = []
        
    def detect_periodicity(self, history: np.ndarray) -> Tuple[bool, int, float]:
        """
        Detect if the time series has periodic patterns.
        
        Args:
            history: Historical time series values (1D array)
            
        Returns:
            is_periodic: Whether periodic pattern is detected
            period: Estimated period (if periodic)
            strength: Periodicity strength (0-1)
        """
        if len(history) < self.config.min_period * 2:
            return False, 0, 0.0
        
        # Compute autocorrelation
        history_centered = history - history.mean()
        autocorr = correlate(history_centered, history_centered, mode='full')
        autocorr = autocorr[len(autocorr)//2:]
        autocorr = autocorr / (autocorr[0] + 1e-8)
        
        # Find peaks in autocorrelation
        min_dist = self.config.min_period
        peaks, properties = find_peaks(
            autocorr[min_dist:self.config.max_period], 
            height=self.config.periodicity_threshold,
            distance=min_dist // 2
        )
        
        if len(peaks) > 0:
            # Get the most prominent peak
            peak_heights = properties['peak_heights']
            best_peak_idx = np.argmax(peak_heights)
            period = peaks[best_peak_idx] + min_dist
            strength = peak_heights[best_peak_idx]
            return True, period, float(strength)
        
        return False, 0, 0.0
    
    def estimate_phase(self, recent_values: np.ndarray, period: int) -> float:
        """
        Estimate current phase in the periodic cycle.
        
        Args:
            recent_values: Recent history values
            period: Detected period
            
        Returns:
            phase: Phase estimate (0-1, where 0=trough, 0.5=peak)
        """
        if len(recent_values) < period:
            return 0.5
        
        # Use FFT to estimate phase of dominant frequency
        values = recent_values[-period:]
        fft_vals = fft(values - values.mean())
        
        # Get phase of the fundamental frequency
        fundamental_idx = 1  # Index 1 corresponds to the period
        phase = np.angle(fft_vals[fundamental_idx])
        
        # Normalize to [0, 1]
        phase_normalized = (phase + np.pi) / (2 * np.pi)
        return float(phase_normalized)
    
    def get_trend_direction(self, recent_values: np.ndarray, window: int = 4) -> float:
        """
        Estimate trend direction from recent values.
        
        Args:
            recent_values: Recent history
            window: Window for trend estimation
            
        Returns:
            trend: Normalized trend (-1 to 1, negative=decreasing)
        """
        if len(recent_values) < window:
            return 0.0
        
        values = recent_values[-window:]
        
        # Simple linear trend
        x = np.arange(len(values))
        slope = np.polyfit(x, values, 1)[0]
        
        # Normalize by std
        std = values.std() + 1e-8
        return float(np.clip(slope / std, -1, 1))


class WavePatternAdapter(CAPEPostHocAdapter):
    """
    Adapts CAPE predictions for wave-like patterns using its unique mask ensemble.
    
    Key CAPE-specific features used:
    1. Multiple mask predictions - select masks that historically perform better on waves
    2. Compartmental structure - use S/I ratio to detect epidemic phase
    3. Uncertainty - high uncertainty = adjust more toward historical pattern
    """
    
    def __init__(self, config: AdaptationConfig = None):
        super().__init__(config)
        self.phase_history = []
        self.pattern_template = None
        
    def learn_pattern_template(self, history: np.ndarray, period: int):
        """
        Learn a template pattern from historical data.
        
        Args:
            history: Historical time series
            period: Detected period length
        """
        if len(history) < period * 2:
            return
        
        # Extract and average multiple periods
        num_periods = len(history) // period
        periods_data = []
        
        for i in range(num_periods):
            start = i * period
            end = start + period
            if end <= len(history):
                periods_data.append(history[start:end])
        
        if periods_data:
            # Average across periods to get template
            self.pattern_template = np.mean(periods_data, axis=0)
            # Normalize template
            self.pattern_template = (self.pattern_template - self.pattern_template.mean()) / (self.pattern_template.std() + 1e-8)
    
    def adapt_prediction(self, 
                         cape_preds_ensemble: np.ndarray,  # (num_masks, num_output, token_size)
                         cape_preds_std: np.ndarray,       # (num_output, token_size)
                         history: np.ndarray,               # Recent history for context
                         compartment_outputs: Dict[str, np.ndarray] = None  # S, I, R outputs
                         ) -> np.ndarray:
        """
        Adapt CAPE prediction using wave pattern detection.
        
        This method is CAPE-specific because:
        1. Uses mask ensemble predictions (CHRONOS has single prediction)
        2. Uses uncertainty to weight adaptation strength
        3. Can use compartment outputs (S, I, R) if available
        
        Args:
            cape_preds_ensemble: Predictions from each mask (num_masks, output_shape)
            cape_preds_std: Standard deviation across masks
            history: Historical values for pattern detection
            compartment_outputs: Optional dict with S, I, R predictions from CAPE
            
        Returns:
            adapted_prediction: Adapted prediction
        """
        # Detect periodicity
        is_periodic, period, strength = self.detect_periodicity(history)
        
        if not is_periodic:
            # No periodicity detected - use standard mean
            return cape_preds_ensemble.mean(axis=0)
        
        self.detected_period = period
        
        # Learn pattern template if not already learned
        if self.pattern_template is None:
            self.learn_pattern_template(history, period)
        
        # Estimate current phase
        phase = self.estimate_phase(history, period)
        trend = self.get_trend_direction(history)
        
        # === CAPE-Specific Adaptation ===
        
        # 1. Select masks that historically work better for this phase
        # (Masks that predict increasing when we're in rising phase, etc.)
        mask_scores = np.zeros(len(cape_preds_ensemble))
        
        for i, mask_pred in enumerate(cape_preds_ensemble):
            pred_trend = self.get_trend_direction(
                np.concatenate([history[-4:], mask_pred.flatten()[:4]])
            )
            # Score masks that align with detected trend
            mask_scores[i] = 1.0 - abs(pred_trend - trend)
        
        # Normalize scores
        mask_weights = np.exp(mask_scores * 2)
        mask_weights /= mask_weights.sum()
        
        # 2. Weighted ensemble based on mask performance
        weighted_pred = np.zeros_like(cape_preds_ensemble[0])
        for i, mask_pred in enumerate(cape_preds_ensemble):
            weighted_pred += mask_weights[i] * mask_pred
        
        # 3. Uncertainty-based pattern adjustment
        # High uncertainty = rely more on historical pattern
        uncertainty = cape_preds_std.mean()
        
        if self.pattern_template is not None and uncertainty > 0.5:
            # Get expected pattern value for current phase
            phase_idx = int(phase * len(self.pattern_template)) % len(self.pattern_template)
            
            # Estimate expected values for prediction horizon
            expected_pattern = []
            for t in range(len(weighted_pred.flatten())):
                idx = (phase_idx + t) % len(self.pattern_template)
                expected_pattern.append(self.pattern_template[idx])
            expected_pattern = np.array(expected_pattern)
            
            # Scale expected pattern to match prediction scale
            pred_mean = weighted_pred.mean()
            pred_std = weighted_pred.std() + 1e-8
            scaled_pattern = expected_pattern * pred_std + pred_mean
            
            # Blend based on uncertainty (higher uncertainty = more pattern weight)
            pattern_weight = min(0.5, uncertainty * 0.3 * strength)
            weighted_pred_flat = weighted_pred.flatten()
            blended = (1 - pattern_weight) * weighted_pred_flat + pattern_weight * scaled_pattern[:len(weighted_pred_flat)]
            weighted_pred = blended.reshape(weighted_pred.shape)
        
        # 4. Compartment-based adjustment (CAPE-specific)
        if compartment_outputs is not None:
            S_pred = compartment_outputs.get('S')
            I_pred = compartment_outputs.get('I', weighted_pred)
            
            if S_pred is not None and isinstance(S_pred, np.ndarray) and len(S_pred) > 0:
                # S/I ratio indicates epidemic phase
                # Low S/I = peak approaching, High S/I = trough approaching
                I_mean = I_pred.mean() if isinstance(I_pred, np.ndarray) else weighted_pred.mean()
                si_ratio = (S_pred.mean() + 1e-8) / (I_mean + 1e-8)
                
                # Adjust prediction based on S/I ratio
                if si_ratio < 0.5:  # Approaching peak
                    # Slightly increase prediction
                    weighted_pred *= 1.05
                elif si_ratio > 2.0:  # Approaching trough
                    # Slightly decrease prediction
                    weighted_pred *= 0.95
        
        return weighted_pred


class MaskSelectionAdapter(CAPEPostHocAdapter):
    """
    Dynamically selects the best masks based on recent prediction errors.
    
    This is CAPE-specific because only CAPE has multiple mask predictions.
    CHRONOS generates a single prediction without mask ensemble.
    """
    
    def __init__(self, config: AdaptationConfig = None):
        super().__init__(config)
        self.mask_errors = {}  # Track error per mask over time
        self.window_size = 10  # Rolling window for error tracking
        
    def update_mask_errors(self, mask_idx: int, error: float):
        """Update error history for a specific mask."""
        if mask_idx not in self.mask_errors:
            self.mask_errors[mask_idx] = []
        
        self.mask_errors[mask_idx].append(error)
        
        # Keep only recent errors
        if len(self.mask_errors[mask_idx]) > self.window_size:
            self.mask_errors[mask_idx] = self.mask_errors[mask_idx][-self.window_size:]
    
    def get_mask_weights(self, num_masks: int) -> np.ndarray:
        """Get weights for each mask based on historical performance."""
        weights = np.ones(num_masks)
        
        for mask_idx in range(num_masks):
            if mask_idx in self.mask_errors and self.mask_errors[mask_idx]:
                avg_error = np.mean(self.mask_errors[mask_idx])
                # Lower error = higher weight
                weights[mask_idx] = 1.0 / (avg_error + 0.1)
        
        # Normalize
        weights /= weights.sum()
        return weights
    
    def adapt_prediction(self,
                         cape_preds_ensemble: np.ndarray,  # (num_masks, output_shape)
                         cape_preds_std: np.ndarray = None
                         ) -> np.ndarray:
        """
        Select and weight masks based on historical performance.
        
        Args:
            cape_preds_ensemble: Predictions from each mask
            cape_preds_std: Uncertainty estimates
            
        Returns:
            weighted_prediction: Performance-weighted ensemble
        """
        num_masks = len(cape_preds_ensemble)
        weights = self.get_mask_weights(num_masks)
        
        # Weighted average
        weighted_pred = np.zeros_like(cape_preds_ensemble[0])
        for i, pred in enumerate(cape_preds_ensemble):
            weighted_pred += weights[i] * pred
        
        return weighted_pred


class UncertaintyGuidedAdapter(CAPEPostHocAdapter):
    """
    Uses CAPE's uncertainty estimates to guide adaptation.
    
    CAPE-specific because:
    1. CAPE provides per-token uncertainty via mask ensemble
    2. High uncertainty regions can trigger conservative predictions
    3. Low uncertainty = trust the prediction more
    """
    
    def __init__(self, config: AdaptationConfig = None):
        super().__init__(config)
        self.naive_weight_high_uncertainty = 0.3  # Blend with naive when uncertain
        
    def adapt_prediction(self,
                         cape_pred_mean: np.ndarray,
                         cape_pred_std: np.ndarray,
                         last_observed: np.ndarray,
                         history: np.ndarray = None
                         ) -> np.ndarray:
        """
        Adapt prediction based on uncertainty.
        
        When uncertainty is high:
        - Blend with naive (last observed value)
        - Shrink extreme predictions toward mean
        
        Args:
            cape_pred_mean: Mean prediction from CAPE
            cape_pred_std: Uncertainty (std across masks)
            last_observed: Last observed token
            history: Optional historical values
            
        Returns:
            adapted_prediction: Uncertainty-adjusted prediction
        """
        # Normalize uncertainty (0-1 scale)
        max_std = 2.0  # Typical max std in normalized data
        uncertainty = np.clip(cape_pred_std / max_std, 0, 1)
        
        # Expand last_observed to match prediction shape
        if last_observed.ndim == 1:
            naive_pred = np.tile(last_observed, (len(cape_pred_mean) // len(last_observed) + 1))[:len(cape_pred_mean.flatten())]
            naive_pred = naive_pred.reshape(cape_pred_mean.shape)
        else:
            naive_pred = np.tile(last_observed, (cape_pred_mean.shape[0], 1))[:cape_pred_mean.shape[0]]
        
        # Blend with naive prediction based on uncertainty
        # Higher uncertainty = more weight to naive prediction
        blend_weight = uncertainty.flatten().mean() * self.naive_weight_high_uncertainty
        
        adapted = (1 - blend_weight) * cape_pred_mean + blend_weight * naive_pred
        
        # Shrink extreme predictions when uncertain
        if history is not None and len(history) > 0:
            hist_mean = history.mean()
            hist_std = history.std() + 1e-8
            
            # How extreme is each prediction?
            z_scores = np.abs((adapted - hist_mean) / hist_std)
            
            # Shrink extreme predictions toward mean when uncertain
            shrink_factor = np.clip(z_scores * uncertainty.mean(), 0, 0.5)
            adapted = adapted * (1 - shrink_factor) + hist_mean * shrink_factor
        
        return adapted


def create_cape_adapter(history: np.ndarray, 
                        config: AdaptationConfig = None) -> CAPEPostHocAdapter:
    """
    Factory function to create the appropriate CAPE adapter based on data characteristics.
    
    Args:
        history: Historical time series for analysis
        config: Optional configuration
        
    Returns:
        Appropriate adapter instance
    """
    config = config or AdaptationConfig()
    
    # Check for periodicity
    base_adapter = CAPEPostHocAdapter(config)
    is_periodic, period, strength = base_adapter.detect_periodicity(history)
    
    if is_periodic and strength > 0.4:
        print(f"  [CAPE Adapter] Detected periodic pattern (period={period}, strength={strength:.2f})")
        adapter = WavePatternAdapter(config)
        adapter.detected_period = period
        adapter.learn_pattern_template(history, period)
        return adapter
    else:
        print(f"  [CAPE Adapter] No strong periodicity detected, using uncertainty-guided adapter")
        return UncertaintyGuidedAdapter(config)


# ============================================================================
# Integration with evaluate_unified.py
# ============================================================================

def adapt_cape_predictions(cape_preds_ensemble: np.ndarray,
                           cape_pred_std: np.ndarray,
                           history: np.ndarray,
                           last_observed: np.ndarray,
                           compartment_outputs: Dict[str, np.ndarray] = None
                           ) -> np.ndarray:
    """
    Main entry point for adapting CAPE predictions.
    
    This function:
    1. Analyzes the historical pattern
    2. Selects appropriate adaptation strategy
    3. Returns adapted prediction
    
    Args:
        cape_preds_ensemble: (num_masks, num_output, token_size) predictions per mask
        cape_pred_std: (num_output, token_size) uncertainty
        history: Historical time series
        last_observed: Last observed token
        compartment_outputs: Optional S, I, R outputs from CAPE
        
    Returns:
        adapted_pred: (num_output, token_size) adapted prediction
    """
    adapter = create_cape_adapter(history)
    
    if isinstance(adapter, WavePatternAdapter):
        return adapter.adapt_prediction(
            cape_preds_ensemble, cape_pred_std, history, compartment_outputs
        )
    elif isinstance(adapter, UncertaintyGuidedAdapter):
        cape_pred_mean = cape_preds_ensemble.mean(axis=0)
        return adapter.adapt_prediction(
            cape_pred_mean, cape_pred_std, last_observed, history
        )
    else:
        # Fallback to simple mean
        return cape_preds_ensemble.mean(axis=0)
