"""
Uncertainty Estimation Wrappers
================================

Contains uncertainty estimation methods:
- MCDropoutWrapper: MC Dropout for deep learning baselines
- CAPEUncertaintyEstimator: Compartment mask ensembling for CAPE
"""

import torch
import torch.nn as nn
from typing import List, Tuple


class MCDropoutWrapper(nn.Module):
    """
    Wrapper that enables MC Dropout for uncertainty estimation.
    
    During inference, keeps dropout enabled and runs multiple forward passes
    to estimate prediction uncertainty.
    """
    
    def __init__(self, model: nn.Module, num_samples: int = 20, dropout_rate: float = 0.1):
        """
        Args:
            model: PyTorch model to wrap
            num_samples: Number of MC forward passes
            dropout_rate: Dropout rate (for reference, actual rate from model)
        """
        super().__init__()
        self.model = model
        self.num_samples = num_samples
        self.dropout_rate = dropout_rate
        self._enable_dropout()
    
    def _enable_dropout(self):
        """Enable dropout layers even in eval mode"""
        for module in self.model.modules():
            if isinstance(module, nn.Dropout):
                module.train()
    
    def forward(self, x, *args, **kwargs):
        """Single forward pass"""
        return self.model(x, *args, **kwargs)
    
    def predict_with_uncertainty(self, x, *args, **kwargs) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Multiple forward passes with dropout for uncertainty estimation.
        
        Args:
            x: Input tensor
            *args, **kwargs: Additional arguments for model
            
        Returns:
            mean_pred: Mean prediction across MC samples
            std_pred: Standard deviation (uncertainty)
            all_preds: All predictions from MC samples [num_samples, ...]
        """
        self._enable_dropout()
        predictions = []
        
        with torch.no_grad():
            for _ in range(self.num_samples):
                pred = self.model(x, *args, **kwargs)
                if isinstance(pred, tuple):
                    pred = pred[0]
                if isinstance(pred, dict):
                    pred = pred.get('output', pred.get('I', next(iter(pred.values()))))
                predictions.append(pred)
        
        all_preds = torch.stack(predictions)
        mean_pred = all_preds.mean(dim=0)
        std_pred = all_preds.std(dim=0)
        
        return mean_pred, std_pred, all_preds


class CAPEUncertaintyEstimator:
    """
    Uncertainty estimation for CAPE models using compartment mask ensembling.
    
    For CompartmentalCAPE: Generates multiple predictions with different
    random compartment masks and computes ensemble statistics.
    
    Always includes S (Susceptible) and I (Infected) compartments,
    randomly activates 3-9 compartments total.
    """
    
    def __init__(self, model, num_masks: int = 20, min_active: int = 3, 
                 max_active: int = 9, device: str = 'cpu'):
        """
        Args:
            model: CAPE or CompartmentalCAPE model
            num_masks: Number of different masks for ensembling
            min_active: Minimum number of active compartments
            max_active: Maximum number of active compartments
            device: Device for tensor operations
        """
        self.model = model
        self.num_masks = num_masks
        self.min_active = min_active
        self.max_active = max_active
        self.device = device
        
        # Check if model is CompartmentalCAPE
        self._check_model_type()
    
    def _check_model_type(self):
        """Determine if model is CompartmentalCAPE"""
        try:
            from model.CAPE_Compartmental import CompartmentalCAPE
            self.is_compartmental = isinstance(self.model, CompartmentalCAPE)
        except ImportError:
            self.is_compartmental = False
        
        if self.is_compartmental:
            self.num_compartments = len(self.model.compartments)
            self.I_idx = self.model.compartments.index('I')
            self.S_idx = self.model.compartments.index('S') if 'S' in self.model.compartments else 0
        else:
            self.num_compartments = 0
    
    def generate_random_mask(self, batch_size: int) -> torch.Tensor:
        """
        Generate a random compartment mask.
        
        Always includes I and S compartments, randomly activates others.
        
        Args:
            batch_size: Batch size for mask
            
        Returns:
            Boolean mask tensor [batch_size, num_compartments]
        """
        mask = torch.zeros(batch_size, self.num_compartments, dtype=torch.bool, device=self.device)
        
        # Always include I and S compartments
        mask[:, self.I_idx] = True
        mask[:, self.S_idx] = True
        
        # Randomly activate other compartments
        num_active = torch.randint(self.min_active, self.max_active + 1, (1,)).item()
        other_indices = [i for i in range(self.num_compartments) 
                        if i != self.I_idx and i != self.S_idx]
        
        num_to_add = max(0, num_active - 2)
        if num_to_add > 0 and len(other_indices) > 0:
            perm = torch.randperm(len(other_indices))[:num_to_add]
            for idx in perm.tolist():
                mask[:, other_indices[idx]] = True
        
        return mask
    
    def predict_with_uncertainty(self, input_seq: torch.Tensor
                                 ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """
        Generate predictions with uncertainty using compartment mask ensembling.
        
        Args:
            input_seq: Input sequence [batch, seq_len, features]
            
        Returns:
            mean_pred: Mean prediction across masks
            std_pred: Standard deviation (uncertainty)
            all_preds: List of predictions from different masks
        """
        self.model.eval()
        batch_size = input_seq.size(0)
        all_predictions = []
        
        with torch.no_grad():
            for _ in range(self.num_masks):
                if self.is_compartmental:
                    # Generate random mask
                    mask = self.generate_random_mask(batch_size)
                    
                    try:
                        predictions = self.model(input_seq, compartment_mask=mask, compute_R_t=False)
                        pred_I = predictions['I']
                        
                        if not (torch.isnan(pred_I).any() or torch.isinf(pred_I).any()):
                            all_predictions.append(pred_I.cpu())
                    except Exception as e:
                        continue
                else:
                    # Regular CAPE - single forward pass (no ensembling)
                    predictions = self.model(input_seq)
                    if isinstance(predictions, dict):
                        pred = predictions.get('I', predictions.get('output', next(iter(predictions.values()))))
                    else:
                        pred = predictions
                    all_predictions.append(pred.cpu())
                    break  # No ensembling for non-compartmental CAPE
        
        if len(all_predictions) == 0:
            # Fallback to single prediction without mask
            predictions = self.model(input_seq)
            if isinstance(predictions, dict):
                pred = predictions['I']
            else:
                pred = predictions
            return pred.cpu(), torch.zeros_like(pred.cpu()), [pred.cpu()]
        
        if len(all_predictions) == 1:
            return all_predictions[0], torch.zeros_like(all_predictions[0]), all_predictions
        
        all_predictions_stacked = torch.stack(all_predictions)
        mean_pred = all_predictions_stacked.mean(dim=0)
        std_pred = all_predictions_stacked.std(dim=0)
        
        return mean_pred, std_pred, all_predictions
