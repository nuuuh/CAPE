#!/usr/bin/env python
"""
Epidemic-Specific Metrics for Forecasting Evaluation
=====================================================
Metrics designed for public health decision-making, emphasizing outbreak detection
and epidemic trajectory accuracy over general forecasting error.

Key Metrics:
- Outbreak Detection Recall: Sensitivity to high-value periods (critical for early warning)
- Alert Sensitivity: Threshold-crossing detection accuracy
- Peak Underestimate Rate: Tendency to underestimate during peaks
- Rising Phase MAE: Error during epidemic growth phases
"""

import numpy as np
from typing import Dict, Optional, Tuple

EPSILON = 1e-8


def compute_outbreak_recall(pred: np.ndarray, target: np.ndarray, 
                            threshold_percentile: float = 50.0,
                            threshold_std_factor: float = 0.5) -> float:
    """
    Outbreak Detection Recall: When actual values are high, how often does model predict high?
    
    Formula: Recall = TP / (TP + FN) where:
        - High threshold τ = median(target) + threshold_std_factor * std(target)
        - TP = sum(pred > τ AND target > τ)
        - FN = sum(pred ≤ τ AND target > τ)
    
    Args:
        pred: Predictions [N,] or [N, D]
        target: Targets [N,] or [N, D]
        threshold_percentile: Base percentile for threshold (default: 50 = median)
        threshold_std_factor: Std multiplier added to percentile (default: 0.5)
    
    Returns:
        Recall percentage (0-100), or None if insufficient high values
    """
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    # Define "high" threshold
    threshold = np.percentile(target_flat, threshold_percentile) + threshold_std_factor * np.std(target_flat)
    
    actual_high = target_flat > threshold
    pred_high = pred_flat > threshold
    
    n_actual_high = np.sum(actual_high)
    if n_actual_high < 3:
        return None  # Insufficient data
    
    true_positives = np.sum(actual_high & pred_high)
    recall = true_positives / n_actual_high * 100
    
    return float(recall)


def compute_alert_sensitivity(pred: np.ndarray, target: np.ndarray,
                              threshold_std_factor: float = 1.0) -> float:
    """
    Alert Sensitivity: When values exceed alert threshold, does model also predict above?
    
    Formula: Sensitivity = TP / P where:
        - Alert threshold α = mean(target) + threshold_std_factor * std(target)
        - P = count(target > α)
        - TP = count(pred > α AND target > α)
    
    Args:
        pred: Predictions
        target: Targets
        threshold_std_factor: Std multiplier above mean for alert threshold
    
    Returns:
        Sensitivity percentage (0-100)
    """
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    threshold = np.mean(target_flat) + threshold_std_factor * np.std(target_flat)
    
    actual_alert = target_flat > threshold
    pred_alert = pred_flat > threshold
    
    n_actual_alert = np.sum(actual_alert)
    if n_actual_alert < 2:
        return None
    
    true_positives = np.sum(actual_alert & pred_alert)
    sensitivity = true_positives / n_actual_alert * 100
    
    return float(sensitivity)


def compute_peak_underestimate_rate(pred: np.ndarray, target: np.ndarray,
                                    top_percentile: float = 90.0) -> float:
    """
    Peak Underestimate Rate: At peak values, how often does model underestimate?
    
    Formula: Rate = count(pred < target | target ∈ top 10%) / count(top 10%)
    
    Lower is better - underestimating peaks leads to under-prepared resources.
    
    Args:
        pred: Predictions
        target: Targets
        top_percentile: Percentile defining "peak" values (default: 90 = top 10%)
    
    Returns:
        Underestimate rate percentage (0-100)
    """
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    threshold = np.percentile(target_flat, top_percentile)
    peak_indices = target_flat >= threshold
    
    n_peaks = np.sum(peak_indices)
    if n_peaks < 2:
        return None
    
    underestimates = pred_flat[peak_indices] < target_flat[peak_indices]
    rate = np.mean(underestimates) * 100
    
    return float(rate)


def compute_rising_phase_mae(pred: np.ndarray, target: np.ndarray, 
                             last_input: np.ndarray,
                             increase_threshold: float = 0.1) -> float:
    """
    Rising Phase MAE: Error during epidemic growth (increasing) phases.
    
    Formula: MAE = mean(|pred - target|) for samples where target > last_input + threshold
    
    Args:
        pred: Predictions
        target: Targets  
        last_input: Last observed values before prediction
        increase_threshold: Minimum increase to count as "rising"
    
    Returns:
        MAE during rising phases
    """
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    last_flat = last_input.flatten()
    
    actual_change = target_flat - last_flat
    rising = actual_change > increase_threshold
    
    n_rising = np.sum(rising)
    if n_rising < 3:
        return None
    
    mae = np.mean(np.abs(pred_flat[rising] - target_flat[rising]))
    return float(mae)


# =============================================================================
# ADDITIONAL EPIDEMIC-SPECIFIC METRICS
# =============================================================================

def compute_trend_accuracy(pred: np.ndarray, target: np.ndarray,
                           last_input: np.ndarray) -> float:
    """
    Trend Accuracy: Does the model correctly predict direction of change?
    
    Formula: Accuracy = count(sign(pred - last) == sign(target - last)) / N
    
    Higher is better - measures ability to predict epidemic trajectory direction.
    
    Args:
        pred: Predictions
        target: Targets
        last_input: Last observed values before prediction
    
    Returns:
        Accuracy percentage (0-100)
    """
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    last_flat = last_input.flatten()
    
    pred_change = pred_flat - last_flat
    actual_change = target_flat - last_flat
    
    # Compare signs (including zero as neutral)
    pred_direction = np.sign(pred_change)
    actual_direction = np.sign(actual_change)
    
    correct = pred_direction == actual_direction
    accuracy = np.mean(correct) * 100
    
    return float(accuracy)


def compute_relative_peak_error(pred: np.ndarray, target: np.ndarray,
                                 top_percentile: float = 90.0) -> float:
    """
    Relative Peak Error: MAPE specifically for peak values.
    
    Formula: RPE = mean(|pred - target| / |target|) for target in top 10%
    
    Lower is better - captures proportional accuracy during critical peaks.
    
    Args:
        pred: Predictions
        target: Targets
        top_percentile: Percentile defining peaks (default: 90)
    
    Returns:
        Relative error percentage (0-100+)
    """
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    threshold = np.percentile(np.abs(target_flat), top_percentile)
    peak_mask = np.abs(target_flat) >= threshold
    
    n_peaks = np.sum(peak_mask)
    if n_peaks < 2:
        return None
    
    # Avoid division by zero
    target_peaks = target_flat[peak_mask]
    pred_peaks = pred_flat[peak_mask]
    
    # Use absolute values for denominator to handle negative predictions
    rel_error = np.abs(pred_peaks - target_peaks) / (np.abs(target_peaks) + EPSILON)
    mape = np.mean(rel_error) * 100
    
    return float(mape)


def compute_outbreak_precision(pred: np.ndarray, target: np.ndarray,
                               threshold_percentile: float = 50.0,
                               threshold_std_factor: float = 0.5) -> float:
    """
    Outbreak Precision: When model predicts high, how often is it actually high?
    
    Formula: Precision = TP / (TP + FP) where:
        - High threshold τ = median(target) + threshold_std_factor * std(target)
        - TP = sum(pred > τ AND target > τ)
        - FP = sum(pred > τ AND target ≤ τ)
    
    Higher is better - reduces false alarms in outbreak detection.
    
    Args:
        pred: Predictions
        target: Targets
        threshold_percentile: Base percentile for threshold
        threshold_std_factor: Std multiplier added to percentile
    
    Returns:
        Precision percentage (0-100)
    """
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    threshold = np.percentile(target_flat, threshold_percentile) + threshold_std_factor * np.std(target_flat)
    
    actual_high = target_flat > threshold
    pred_high = pred_flat > threshold
    
    n_pred_high = np.sum(pred_high)
    if n_pred_high < 2:
        return None
    
    true_positives = np.sum(actual_high & pred_high)
    precision = true_positives / n_pred_high * 100
    
    return float(precision)


def compute_outbreak_f1(pred: np.ndarray, target: np.ndarray,
                        threshold_percentile: float = 50.0,
                        threshold_std_factor: float = 0.5) -> float:
    """
    Outbreak F1 Score: Harmonic mean of outbreak precision and recall.
    
    Formula: F1 = 2 * (Precision * Recall) / (Precision + Recall)
    
    Higher is better - balances detection sensitivity and precision.
    """
    recall = compute_outbreak_recall(pred, target, threshold_percentile, threshold_std_factor)
    precision = compute_outbreak_precision(pred, target, threshold_percentile, threshold_std_factor)
    
    if recall is None or precision is None:
        return None
    
    if recall + precision < EPSILON:
        return 0.0
    
    f1 = 2 * (precision * recall) / (precision + recall)
    return float(f1)


def compute_critical_success_index(pred: np.ndarray, target: np.ndarray,
                                   threshold_percentile: float = 75.0) -> float:
    """
    Critical Success Index (CSI/Threat Score): TP / (TP + FP + FN)
    
    Used in meteorology for rare event forecasting. Suitable for outbreak detection.
    
    Higher is better - penalizes both false alarms and missed detections.
    
    Args:
        pred: Predictions
        target: Targets
        threshold_percentile: Percentile for "critical" threshold
    
    Returns:
        CSI percentage (0-100)
    """
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    threshold = np.percentile(target_flat, threshold_percentile)
    
    actual_high = target_flat > threshold
    pred_high = pred_flat > threshold
    
    tp = np.sum(actual_high & pred_high)
    fp = np.sum(~actual_high & pred_high)
    fn = np.sum(actual_high & ~pred_high)
    
    denominator = tp + fp + fn
    if denominator < 1:
        return None
    
    csi = tp / denominator * 100
    return float(csi)


def compute_normalized_mse(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Normalized MSE: MSE normalized by target variance.
    
    Formula: NMSE = MSE / Var(target)
    
    Lower is better - scale-independent error measure.
    Values < 1 indicate better than predicting mean.
    """
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    mse = np.mean((pred_flat - target_flat) ** 2)
    var = np.var(target_flat)
    
    if var < EPSILON:
        return None
    
    nmse = mse / var
    return float(nmse)


def compute_correlation(pred: np.ndarray, target: np.ndarray) -> float:
    """
    Pearson Correlation: Linear correlation between predictions and targets.
    
    Higher is better (closer to 100) - measures linear relationship.
    
    Returns:
        Correlation as percentage (-100 to 100)
    """
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    if len(pred_flat) < 3:
        return None
    
    # Handle constant arrays
    if np.std(pred_flat) < EPSILON or np.std(target_flat) < EPSILON:
        return None
    
    corr = np.corrcoef(pred_flat, target_flat)[0, 1]
    
    if np.isnan(corr):
        return None
    
    return float(corr * 100)


def compute_skill_score(pred: np.ndarray, target: np.ndarray,
                        last_input: np.ndarray) -> float:
    """
    Skill Score: Improvement over naive persistence forecast.
    
    Formula: SS = 1 - MSE(pred) / MSE(persistence)
    where persistence = last observed value
    
    Higher is better - values > 0 indicate better than persistence.
    Values as percentage (0-100 scale, can be negative).
    """
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    last_flat = last_input.flatten()
    
    mse_pred = np.mean((pred_flat - target_flat) ** 2)
    mse_persist = np.mean((last_flat - target_flat) ** 2)
    
    if mse_persist < EPSILON:
        return None
    
    skill = (1 - mse_pred / mse_persist) * 100
    return float(skill)


def compute_weighted_mae(pred: np.ndarray, target: np.ndarray,
                         weight_power: float = 1.0) -> float:
    """
    Weighted MAE: Errors on larger values weighted more heavily.
    
    Formula: WMAE = sum(|target|^power * |pred - target|) / sum(|target|^power)
    
    Lower is better - prioritizes accuracy on significant values.
    """
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    weights = np.abs(target_flat) ** weight_power + EPSILON
    weighted_errors = weights * np.abs(pred_flat - target_flat)
    
    wmae = np.sum(weighted_errors) / np.sum(weights)
    return float(wmae)


def compute_epidemic_metrics(pred: np.ndarray, target: np.ndarray,
                             last_input: Optional[np.ndarray] = None) -> Dict[str, float]:
    """
    Compute all epidemic-specific metrics.
    
    Args:
        pred: Predictions [N, D] or [N,]
        target: Targets [N, D] or [N,]
        last_input: Optional last input values for trajectory metrics [N, D] or [N,]
    
    Returns:
        Dictionary of metric name -> value
    """
    metrics = {}
    
    # Standard metrics
    pred_flat = pred.flatten()
    target_flat = target.flatten()
    
    metrics['mse'] = float(np.mean((pred_flat - target_flat) ** 2))
    metrics['mae'] = float(np.mean(np.abs(pred_flat - target_flat)))
    
    # Epidemic metrics
    outbreak_recall = compute_outbreak_recall(pred, target)
    if outbreak_recall is not None:
        metrics['outbreak_recall'] = outbreak_recall
    
    alert_sens = compute_alert_sensitivity(pred, target)
    if alert_sens is not None:
        metrics['alert_sensitivity'] = alert_sens
    
    peak_under = compute_peak_underestimate_rate(pred, target)
    if peak_under is not None:
        metrics['peak_underestimate_rate'] = peak_under
    
    if last_input is not None:
        rising_mae = compute_rising_phase_mae(pred, target, last_input)
        if rising_mae is not None:
            metrics['rising_phase_mae'] = rising_mae
    
    # NEW METRICS
    # Outbreak detection metrics
    outbreak_prec = compute_outbreak_precision(pred, target)
    if outbreak_prec is not None:
        metrics['outbreak_precision'] = outbreak_prec
    
    outbreak_f1 = compute_outbreak_f1(pred, target)
    if outbreak_f1 is not None:
        metrics['outbreak_f1'] = outbreak_f1
    
    csi = compute_critical_success_index(pred, target)
    if csi is not None:
        metrics['critical_success_index'] = csi
    
    # Peak-focused metrics
    rel_peak_err = compute_relative_peak_error(pred, target)
    if rel_peak_err is not None:
        metrics['relative_peak_error'] = rel_peak_err
    
    # Correlation and skill
    corr = compute_correlation(pred, target)
    if corr is not None:
        metrics['correlation'] = corr
    
    nmse = compute_normalized_mse(pred, target)
    if nmse is not None:
        metrics['normalized_mse'] = nmse
    
    # Weighted errors
    wmae = compute_weighted_mae(pred, target)
    if wmae is not None:
        metrics['weighted_mae'] = wmae
    
    # Trajectory metrics (require last_input)
    if last_input is not None:
        trend_acc = compute_trend_accuracy(pred, target, last_input)
        if trend_acc is not None:
            metrics['trend_accuracy'] = trend_acc
        
        skill = compute_skill_score(pred, target, last_input)
        if skill is not None:
            metrics['skill_score'] = skill
    
    return metrics


# =============================================================================
# METRIC COMPARISONS
# =============================================================================

# Define which metrics are "higher is better" vs "lower is better"
HIGHER_BETTER = {
    'outbreak_recall', 'alert_sensitivity', 
    'outbreak_precision', 'outbreak_f1', 'critical_success_index',
    'correlation', 'trend_accuracy', 'skill_score'
}
LOWER_BETTER = {
    'mse', 'mae', 'peak_underestimate_rate', 'rising_phase_mae',
    'relative_peak_error', 'normalized_mse', 'weighted_mae'
}


def compare_metrics(metrics_a: Dict[str, float], metrics_b: Dict[str, float],
                    primary_metric: str = 'outbreak_recall') -> Dict[str, str]:
    """
    Compare two sets of metrics and determine winners.
    
    Args:
        metrics_a: First model's metrics
        metrics_b: Second model's metrics
        primary_metric: Primary metric for overall comparison
    
    Returns:
        Dictionary mapping metric name to winner ('a', 'b', or 'tie')
    """
    results = {}
    
    for metric in set(metrics_a.keys()) | set(metrics_b.keys()):
        if metric not in metrics_a or metric not in metrics_b:
            continue
        
        val_a = metrics_a[metric]
        val_b = metrics_b[metric]
        
        if val_a is None or val_b is None:
            continue
        
        if metric in HIGHER_BETTER:
            if val_a > val_b:
                results[metric] = 'a'
            elif val_b > val_a:
                results[metric] = 'b'
            else:
                results[metric] = 'tie'
        else:  # Lower is better
            if val_a < val_b:
                results[metric] = 'a'
            elif val_b < val_a:
                results[metric] = 'b'
            else:
                results[metric] = 'tie'
    
    return results
