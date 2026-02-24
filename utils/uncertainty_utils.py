import torch
import numpy as np
from typing import Dict, List, Tuple, Optional


def generate_compartment_masks(num_masks: int = 10, 
                               num_compartments: int = 12,
                               min_active: int = 2,
                               max_active: int = 8,
                               always_include: List[int] = [1],  # Always include 'I' (index 1)
                               seed: Optional[int] = None) -> torch.Tensor:
    """
    Generate diverse compartment masks for uncertainty estimation
    
    Args:
        num_masks: Number of different masks to generate
        num_compartments: Total number of compartments
        min_active: Minimum number of active compartments per mask
        max_active: Maximum number of active compartments per mask
        always_include: Compartment indices that must always be active
        seed: Random seed for reproducibility
        
    Returns:
        masks: [num_masks, num_compartments] binary tensor
    """
    if seed is not None:
        torch.manual_seed(seed)
        np.random.seed(seed)
    
    masks = []
    for _ in range(num_masks):
        # Start with always-included compartments
        mask = torch.zeros(num_compartments)
        for idx in always_include:
            mask[idx] = 1
        
        # Randomly activate other compartments
        num_to_add = np.random.randint(min_active, max_active + 1) - len(always_include)
        num_to_add = max(0, num_to_add)
        
        available_indices = [i for i in range(num_compartments) if i not in always_include]
        if num_to_add > 0 and len(available_indices) > 0:
            selected = np.random.choice(available_indices, 
                                       size=min(num_to_add, len(available_indices)), 
                                       replace=False)
            mask[selected] = 1
        
        masks.append(mask)
    
    return torch.stack(masks)


def predict_with_uncertainty(model, 
                            input_seq: torch.Tensor,
                            compartment_masks: torch.Tensor,
                            target_compartment: str = 'I') -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[torch.Tensor]]:
    """
    Make predictions with uncertainty estimation using compartment mask ensembling
    
    Args:
        model: CompartmentalCAPE model
        input_seq: Input sequence [batch, seq_len, input_size]
        compartment_masks: Ensemble of masks [num_masks, num_compartments]
        target_compartment: Which compartment to return predictions for
        
    Returns:
        mean_pred: Mean prediction [batch, seq_len, input_size]
        std_pred: Standard deviation (uncertainty) [batch, seq_len, input_size]
        all_preds: All ensemble predictions [num_masks, batch, seq_len, input_size]
        all_predictions_dict: List of prediction dicts from each mask
    """
    model.eval()
    batch_size = input_seq.size(0)
    num_masks = compartment_masks.size(0)
    
    all_preds = []
    all_predictions_dict = []
    
    with torch.no_grad():
        for i in range(num_masks):
            # Expand mask to batch size [batch, num_compartments]
            mask = compartment_masks[i:i+1].expand(batch_size, -1).to(input_seq.device)
            
            # Forward pass with this mask
            predictions_dict = model(input_seq, compartment_mask=mask, compute_R_t=False)
            all_predictions_dict.append(predictions_dict)
            
            # Extract target compartment prediction
            if target_compartment in predictions_dict:
                pred = predictions_dict[target_compartment]
                all_preds.append(pred)
    
    # Stack predictions [num_masks, batch, seq_len, input_size]
    all_preds = torch.stack(all_preds)
    
    # Compute statistics
    mean_pred = all_preds.mean(dim=0)  # [batch, seq_len, input_size]
    std_pred = all_preds.std(dim=0)    # [batch, seq_len, input_size]
    
    return mean_pred, std_pred, all_preds, all_predictions_dict


def evaluate_uncertainty_quality(predictions: torch.Tensor,
                                 uncertainties: torch.Tensor,
                                 targets: torch.Tensor,
                                 num_bins: int = 10) -> Dict[str, float]:
    """
    Evaluate quality of uncertainty estimates using calibration metrics
    
    Args:
        predictions: Mean predictions [batch, seq_len, input_size]
        uncertainties: Std predictions [batch, seq_len, input_size]
        targets: Ground truth [batch, seq_len, input_size]
        num_bins: Number of bins for calibration
        
    Returns:
        metrics: Dict with uncertainty quality metrics
    """
    # Flatten for easier computation
    pred_flat = predictions.reshape(-1).cpu().numpy()
    std_flat = uncertainties.reshape(-1).cpu().numpy()
    target_flat = targets.reshape(-1).cpu().numpy()
    
    # Compute errors
    errors = np.abs(pred_flat - target_flat)
    
    # 1. Correlation between uncertainty and error
    if std_flat.std() > 1e-8:  # Avoid division by zero
        uncertainty_error_corr = np.corrcoef(std_flat, errors)[0, 1]
    else:
        uncertainty_error_corr = 0.0
    
    # 2. Calibration: Check if predicted std matches actual error distribution
    # Sort by uncertainty
    sorted_indices = np.argsort(std_flat)
    bin_size = len(sorted_indices) // num_bins
    
    calibration_errors = []
    for i in range(num_bins):
        start_idx = i * bin_size
        end_idx = (i + 1) * bin_size if i < num_bins - 1 else len(sorted_indices)
        bin_indices = sorted_indices[start_idx:end_idx]
        
        # Mean predicted uncertainty in this bin
        mean_pred_std = std_flat[bin_indices].mean()
        
        # Actual RMSE in this bin
        actual_rmse = np.sqrt(np.mean(errors[bin_indices] ** 2))
        
        # Calibration error: difference between predicted and actual
        calibration_errors.append(np.abs(mean_pred_std - actual_rmse))
    
    expected_calibration_error = np.mean(calibration_errors)
    
    # 3. Sharpness: Average uncertainty (lower is better, but needs to be balanced with calibration)
    sharpness = std_flat.mean()
    
    # 4. Coverage: Proportion of targets within predicted intervals
    # For 1-sigma interval (~68% coverage expected)
    within_1sigma = np.abs(pred_flat - target_flat) <= std_flat
    coverage_1sigma = within_1sigma.mean()
    
    # For 2-sigma interval (~95% coverage expected)
    within_2sigma = np.abs(pred_flat - target_flat) <= 2 * std_flat
    coverage_2sigma = within_2sigma.mean()
    
    # 5. Negative Log-Likelihood (assuming Gaussian)
    nll = 0.5 * np.log(2 * np.pi * std_flat**2 + 1e-8) + (errors**2) / (2 * std_flat**2 + 1e-8)
    avg_nll = nll.mean()
    
    return {
        'uncertainty_error_correlation': float(uncertainty_error_corr),
        'expected_calibration_error': float(expected_calibration_error),
        'sharpness': float(sharpness),
        'coverage_1sigma': float(coverage_1sigma),
        'coverage_2sigma': float(coverage_2sigma),
        'negative_log_likelihood': float(avg_nll),
    }


def visualize_uncertainty(predictions: np.ndarray,
                         uncertainties: np.ndarray,
                         targets: np.ndarray,
                         time_steps: Optional[np.ndarray] = None,
                         save_path: Optional[str] = None):
    """
    Visualize predictions with uncertainty bands
    
    Args:
        predictions: Mean predictions [seq_len]
        uncertainties: Std predictions [seq_len]
        targets: Ground truth [seq_len]
        time_steps: Optional time axis
        save_path: Path to save figure
    """
    import matplotlib.pyplot as plt
    
    if time_steps is None:
        time_steps = np.arange(len(predictions))
    
    plt.figure(figsize=(12, 6))
    
    # Plot ground truth
    plt.plot(time_steps, targets, 'k-', label='Ground Truth', linewidth=2)
    
    # Plot mean prediction
    plt.plot(time_steps, predictions, 'b-', label='Prediction (Mean)', linewidth=2)
    
    # Plot uncertainty bands (1-sigma and 2-sigma)
    plt.fill_between(time_steps,
                     predictions - uncertainties,
                     predictions + uncertainties,
                     alpha=0.3, color='blue', label='1σ Uncertainty')
    
    plt.fill_between(time_steps,
                     predictions - 2*uncertainties,
                     predictions + 2*uncertainties,
                     alpha=0.15, color='blue', label='2σ Uncertainty')
    
    plt.xlabel('Time Step')
    plt.ylabel('Value')
    plt.title('Predictions with Uncertainty Estimates (Compartment Mask Ensembling)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved uncertainty plot to {save_path}")
    else:
        plt.show()
    
    plt.close()
