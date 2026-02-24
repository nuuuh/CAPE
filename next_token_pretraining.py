"""
Next-Token-Prediction Pretraining Script
Trains CAPE model in an autoregressive manner similar to language model pretraining
"""

import torch
from dataset.dataset_wrapper import DataSetWrapper
import matplotlib.pyplot as plt
import sys
import pandas as pd
import os 
import time
import random
import numpy as np
from tqdm import tqdm

from config import Config, setup_seed
from model import CAPE, CAPEForPretraining, create_cape_model
from utils.model_factory import get_model

import warnings
warnings.filterwarnings('ignore')


class NextTokenPretrainer:
    """Trainer for next-token-prediction pretraining"""
    
    def __init__(self, model, train_loader, valid_loader, test_loader, 
                 lr=1e-3, weight_decay=1e-3, device='cpu', token_size=4, config=None,
                 use_multi_gpu=True):
        self.device = device
        self.token_size = token_size
        self.config = config
        self.weight_decay = weight_decay
        self.use_multi_gpu = use_multi_gpu
        
        # Check if using compartmental model
        from model.CAPE_Compartmental import CompartmentalCAPE
        self.is_compartmental = isinstance(model, CompartmentalCAPE)
        
        # Multi-GPU support with DataParallel
        if use_multi_gpu and torch.cuda.device_count() > 1 and device != 'cpu':
            print(f"Using DataParallel with {torch.cuda.device_count()} GPUs")
            self.model = torch.nn.DataParallel(model).to(device)
            self.model_unwrapped = model  # Keep reference to unwrapped model
        else:
            self.model = model.to(device)
            self.model_unwrapped = model
        
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader
        
        if self.is_compartmental:
            print("Using Compartmental CAPE model")
            self.loss_weights = getattr(config, 'loss_weights_dict', None)
            
            # R(t) computation settings
            self.compute_R_t = getattr(config, 'compute_R_t', False)
            self.R_t_loss_weight = getattr(config, 'R_t_loss_weight', 0.1)
            
            # MAE loss weight (combined MSE + MAE training)
            self.mae_loss_weight = getattr(config, 'mae_loss_weight', 0.0)
            
            # Finetuning strategy
            print("Finetuning strategy: Full mask with I compartment supervision")
            print("  → Training uses all compartments but only I has supervision")
            print("  → Test-time uncertainty via multiple random masks")
            
            if self.compute_R_t:
                print(f"R(t) time series supervision enabled (weight={self.R_t_loss_weight})")
                print(f"  Note: R(t) supervision only works with synthetic data (--use_synthetic_data True)")
                print(f"  R(t) supervision will be automatically disabled for real data batches")
            
            if self.mae_loss_weight > 0:
                print(f"Combined MSE + MAE loss enabled (MAE weight={self.mae_loss_weight})")
        
        # Optimizer
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=10, T_mult=2, eta_min=lr * 0.01
        )
    
    def train(self, epoch):
        """Train for one epoch"""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        for batch in self.train_loader:
            # Get input tokens
            input_tokens = batch['input'].to(self.device)
            
            # Get compartment mask if available
            compartment_mask = batch.get('compartment_mask', None)
            if compartment_mask is not None:
                compartment_mask = compartment_mask.to(self.device)
            
            if self.is_compartmental:
                # Compartmental training
                has_R_t_data = 'target_R_t' in batch
                
                # Check if batch contains GP samples - disable R_t supervision for GP samples
                # GP samples have fake R_t values that shouldn't be used for supervision
                is_gp_batch = batch.get('is_gp_sample', None)
                if is_gp_batch is not None and is_gp_batch.any():
                    # If any sample in batch is a GP sample, disable R_t for the whole batch
                    # (simpler than per-sample masking, and GP ratio is typically low)
                    can_compute_R_t = False
                else:
                    can_compute_R_t = self.compute_R_t and has_R_t_data
                
                # Get targets - check if we have compartment data or just I compartment
                if 'target_compartments' in batch:
                    # Full compartmental data available (from synthetic data)
                    target_compartments = batch['target_compartments']
                    targets = {comp: target_compartments[comp].to(self.device) 
                              for comp in target_compartments.keys()}
                    
                    predictions = self.model(input_tokens, compartment_mask=compartment_mask, compute_R_t=can_compute_R_t)
                    
                    if can_compute_R_t:
                        target_R_t = batch['target_R_t'].to(self.device)
                        targets['R_t'] = target_R_t
                    
                    loss_weights = self.loss_weights.copy() if self.loss_weights else {}
                    if can_compute_R_t:
                        loss_weights['R_t'] = self.R_t_loss_weight
                    
                    # Get sequence mask if available (to ignore padded positions)
                    sequence_mask = batch.get('sequence_mask', None)
                    if sequence_mask is not None:
                        sequence_mask = sequence_mask.to(self.device)
                    
                    loss, losses_dict = self.model_unwrapped.compute_loss(predictions, targets, loss_weights=loss_weights, compartment_mask=compartment_mask, sequence_mask=sequence_mask, mae_loss_weight=self.mae_loss_weight)
                else:
                    # Only I compartment available (from real data) - use shifted sequence
                    label_tokens = batch.get('target', batch.get('label')).to(self.device)
                    
                    # Concatenate input and label to create full sequence
                    full_sequence = torch.cat([input_tokens, label_tokens], dim=1)
                    
                    # Create shifted pairs: input = [t0...tn-1], target = [t1...tn]
                    input_seq = full_sequence[:, :-1, :]
                    target_seq = full_sequence[:, 1:, :]
                    
                    # FINETUNING STRATEGY: Create full mask when None
                    # Since we only have "I" compartment as supervision, we use a full mask
                    # to allow all compartments to be active. At test time, we can use
                    # different masks for uncertainty estimation.
                    if compartment_mask is None:
                        batch_size = input_seq.size(0)
                        num_compartments = len(self.model_unwrapped.compartments)
                        # Use full mask (all compartments active)
                        compartment_mask = torch.ones(batch_size, num_compartments, dtype=torch.bool, device=self.device)
                    
                    # Forward pass - predict only I compartment (but use diverse masks)
                    predictions = self.model(input_seq, compartment_mask=compartment_mask, compute_R_t=False)
                    
                    # Only use I compartment for loss (we only have I as supervision)
                    targets = {'I': target_seq}
                    loss_weights = {'I': 1.0}  # Only I compartment
                    
                    loss, losses_dict = self.model_unwrapped.compute_loss(predictions, targets, loss_weights=loss_weights, compartment_mask=compartment_mask, mae_loss_weight=self.mae_loss_weight)
            else:
                # Regular CAPE
                # For next-token prediction, concatenate input and label to create shifted training pairs
                label_tokens = batch.get('target', batch.get('label')).to(self.device)
                
                # Concatenate input and label to create full sequence
                full_sequence = torch.cat([input_tokens, label_tokens], dim=1)  # [batch, total_tokens, token_size]
                
                # Create shifted pairs: input = [t0...tn-1], target = [t1...tn]
                input_seq = full_sequence[:, :-1, :]  # All but last token
                target_seq = full_sequence[:, 1:, :]  # All but first token
                
                # Forward pass
                predictions = self.model(input_seq)
                
                # MSE loss for next-token prediction
                loss = torch.nn.functional.mse_loss(predictions, target_seq)
            
            # Check for NaN/Inf before backward
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN/Inf loss detected ({loss.item():.2e}), skipping batch")
                continue
            
            # Debug: Check if loss is unreasonably large (first batch only)
            if num_batches == 0 and loss.item() > 1000:
                print(f"\nWARNING: First batch loss is very large: {loss.item():.2e}")
                print(f"Investigating data ranges...")
                if self.is_compartmental and 'target_compartments' in batch:
                    for comp, targ in batch['target_compartments'].items():
                        targ_cpu = targ.cpu()
                        print(f"  Target '{comp}': [{targ_cpu.min():.3f}, {targ_cpu.max():.3f}], mean={targ_cpu.mean():.3f}, std={targ_cpu.std():.3f}")
                    input_cpu = input_tokens.cpu()
                    print(f"  Input: [{input_cpu.min():.3f}, {input_cpu.max():.3f}], mean={input_cpu.mean():.3f}, std={input_cpu.std():.3f}")
                    for comp, pred in predictions.items():
                        if comp != 'R_t' and comp in targets:
                            pred_cpu = pred.detach().cpu()
                            print(f"  Prediction '{comp}': [{pred_cpu.min():.3f}, {pred_cpu.max():.3f}], mean={pred_cpu.mean():.3f}, std={pred_cpu.std():.3f}")
                    if self.is_compartmental and hasattr(self, 'compute_R_t'):
                        print(f"  Loss breakdown:")
                        for comp, comp_loss in losses_dict.items():
                            print(f"    {comp}: {comp_loss.item():.2e}")
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            
            # Clip gradients
            grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            # Check for NaN gradients
            if torch.isnan(grad_norm) or torch.isinf(grad_norm):
                print(f"Warning: NaN/Inf gradients detected, skipping batch")
                continue
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
        
        # Step scheduler
        self.scheduler.step()
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        return avg_loss
    
    def evaluate(self, data_loader, save_path=None, epoch=None, phase='valid'):
        """Evaluate on validation/test set
        
        Args:
            data_loader: DataLoader to evaluate on
            save_path: If provided, save visualization of sample predictions
            epoch: Current epoch number for labeling visualizations
            phase: 'valid' or 'test' for labeling visualizations
        """
        self.model.eval()
        total_loss = 0
        total_mae = 0
        num_batches = 0
        compartment_losses = {}
        compartment_maes = {}
        
        # Store samples for visualization
        samples_for_viz = []
        max_viz_samples = 25
        
        with torch.no_grad():
            progress_bar = tqdm(data_loader, desc="Evaluating", unit="batch", leave=False)
            for batch in progress_bar:
                input_tokens = batch['input'].to(self.device)
                
                # Store samples for visualization (only collect first 5 samples)
                if len(samples_for_viz) < max_viz_samples:
                    batch_size = input_tokens.size(0)
                    for i in range(min(batch_size, max_viz_samples - len(samples_for_viz))):
                        samples_for_viz.append({
                            'input': input_tokens[i:i+1].clone(),
                            'batch': batch,
                            'batch_idx': i
                        })
                
                # Get compartment mask if available
                compartment_mask = batch.get('compartment_mask', None)
                if compartment_mask is not None:
                    compartment_mask = compartment_mask.to(self.device)
                
                if self.is_compartmental:
                    # Compartmental model
                    # Check for R(t) data in validation
                    has_R_t_data = 'target_R_t' in batch
                    can_compute_R_t = self.compute_R_t and has_R_t_data
                    
                    # Get all compartment targets
                    if 'target_compartments' in batch:
                        # Full compartmental data
                        target_compartments = batch['target_compartments']
                        targets = {comp: target_compartments[comp].to(self.device) 
                                  for comp in target_compartments.keys()}
                        
                        predictions = self.model(input_tokens, compartment_mask=compartment_mask, compute_R_t=can_compute_R_t)
                        
                        if can_compute_R_t:
                            target_R_t = batch['target_R_t']
                            if target_R_t is not None:
                                targets['R_t'] = target_R_t.to(self.device)
                        
                        # Get sequence mask if available (to ignore padded positions)
                        sequence_mask = batch.get('sequence_mask', None)
                        if sequence_mask is not None:
                            sequence_mask = sequence_mask.to(self.device)
                        
                        # Compute loss
                        loss, losses_dict = self.model_unwrapped.compute_loss(
                            predictions, targets, loss_weights=self.loss_weights, compartment_mask=compartment_mask, sequence_mask=sequence_mask, mae_loss_weight=self.mae_loss_weight
                        )
                    else:
                        # Only I compartment (real data) - use shifted sequence
                        label_tokens = batch.get('target', batch.get('label'))
                        if label_tokens is None:
                            continue
                        
                        label_tokens = label_tokens.to(self.device)
                        
                        # Concatenate input and label to create full sequence
                        full_sequence = torch.cat([input_tokens, label_tokens], dim=1)
                        
                        # Create shifted pairs
                        input_seq = full_sequence[:, :-1, :]
                        target_seq = full_sequence[:, 1:, :]
                        
                        # Forward pass
                        predictions = self.model(input_seq, compartment_mask=compartment_mask, compute_R_t=False)
                        
                        # Only I compartment
                        targets = {'I': target_seq}
                        loss_weights = {'I': 1.0}
                        
                        # Compute loss
                        losses_dict = {}
                        if 'I' in predictions and 'I' in targets:
                            loss = torch.nn.functional.mse_loss(predictions['I'], targets['I'])
                            losses_dict['I'] = loss
                        else:
                            continue
                    
                    # Compute MAE for each compartment
                    for comp, pred in predictions.items():
                        if comp in targets:
                            mae_val = torch.nn.functional.l1_loss(pred, targets[comp])
                            if comp not in compartment_maes:
                                compartment_maes[comp] = 0
                            compartment_maes[comp] += mae_val.item()
                    
                    # Accumulate compartment losses
                    for comp, comp_loss in losses_dict.items():
                        if comp not in compartment_losses:
                            compartment_losses[comp] = 0
                        compartment_losses[comp] += comp_loss.item()
                    
                    # Overall MAE (for I compartment)
                    mae = compartment_maes.get('I', 0) if compartment_maes else 0
                else:
                    # Regular CAPE model
                    label_tokens = batch.get('target', batch.get('label'))
                    if label_tokens is None:
                        continue  # Skip batch if no targets
                    
                    label_tokens = label_tokens.to(self.device)
                    
                    # Concatenate input and label to create full sequence
                    full_sequence = torch.cat([input_tokens, label_tokens], dim=1)
                    
                    # Create shifted pairs: input = [t0...tn-1], target = [t1...tn]
                    input_seq = full_sequence[:, :-1, :]
                    target_seq = full_sequence[:, 1:, :]
                    
                    # Forward pass
                    predictions = self.model(input_seq)
                    
                    # MSE loss
                    loss = torch.nn.functional.mse_loss(predictions, target_seq)
                    
                    # MAE for interpretability
                    mae = torch.nn.functional.l1_loss(predictions, target_seq)
                
                total_loss += loss.item()
                total_mae += mae if isinstance(mae, float) else mae.item()
                num_batches += 1
        
        avg_loss = total_loss / num_batches if num_batches > 0 else 0
        avg_mae = total_mae / num_batches if num_batches > 0 else 0
        
        # Average compartment metrics
        if compartment_losses:
            for comp in compartment_losses:
                compartment_losses[comp] /= num_batches
        if compartment_maes:
            for comp in compartment_maes:
                compartment_maes[comp] /= num_batches
        
        # Visualize sample predictions if save_path provided
        if save_path is not None and samples_for_viz:
            self._visualize_predictions(samples_for_viz, save_path, epoch, phase)
        
        if self.is_compartmental:
            return avg_loss, avg_mae, compartment_losses, compartment_maes
        else:
            return avg_loss, avg_mae
    
    def _visualize_predictions(self, samples, save_path, epoch, phase, num_uncertainty_samples=20):
        """Visualize sample predictions vs ground truth with uncertainty estimation
        
        Args:
            samples: List of sample dictionaries with input and batch info
            save_path: Directory to save plots
            epoch: Current epoch number
            phase: 'valid' or 'test'
            num_uncertainty_samples: Number of forward passes with different masks for uncertainty
        """
        import matplotlib.pyplot as plt
        
        self.model.eval()
        num_samples = len(samples)
        
        fig, axes = plt.subplots(num_samples, 1, figsize=(12, 3*num_samples))
        if num_samples == 1:
            axes = [axes]
        
        # Collect all sample results for saving to numpy file
        all_sample_results = {
            'input_series': [],
            'pred_series': [],
            'target_series': [],
            'std_series': [],
            'mae': [],
            'mse': [],
            'calibration': [],
            'avg_uncertainty': [],
            'valid_length': [],
        }
        
        with torch.no_grad():
            for idx, sample_info in enumerate(samples):
                input_tokens = sample_info['input'].to(self.device)
                batch = sample_info['batch']
                batch_idx = sample_info['batch_idx']
                
                # Get compartment mask if available
                compartment_mask = batch.get('compartment_mask', None)
                if compartment_mask is not None:
                    compartment_mask = compartment_mask[batch_idx:batch_idx+1].to(self.device)
                
                # Prepare input and target based on model type
                if self.is_compartmental:
                    # Get predictions and targets for compartmental model
                    if 'target_compartments' in batch:
                        # Synthetic data with full compartments
                        target_compartments = batch['target_compartments']
                        target_I = target_compartments['I'][batch_idx:batch_idx+1].to(self.device)
                        use_shifted = False
                    else:
                        # Real data - use shifted sequence
                        label_tokens = batch.get('target', batch.get('label'))
                        if label_tokens is None:
                            continue
                        
                        label_tokens = label_tokens[batch_idx:batch_idx+1].to(self.device)
                        full_sequence = torch.cat([input_tokens, label_tokens], dim=1)
                        input_tokens = full_sequence[:, :-1, :]
                        target_I = full_sequence[:, 1:, :]
                        use_shifted = True
                    
                    # Generate predictions with uncertainty estimation using random compartment masks
                    # Different mask configurations act as an ensemble of compartment models
                    all_predictions = []
                    
                    if self.is_compartmental and num_uncertainty_samples > 1:
                        if idx == 0:
                            print(f"\nGenerating {num_uncertainty_samples} predictions with random compartment masks (ensemble)...")
                        
                        batch_size = input_tokens.size(0)
                        num_compartments = len(self.model_unwrapped.compartments)
                        I_idx = self.model_unwrapped.compartments.index('I')
                        S_idx = self.model_unwrapped.compartments.index('S') if 'S' in self.model_unwrapped.compartments else 0
                        
                        for sample_idx in range(num_uncertainty_samples):
                            # Generate random mask with 3-10 active compartments
                            num_active = torch.randint(3, min(11, num_compartments + 1), (1,)).item()
                            mask = torch.zeros(batch_size, num_compartments, dtype=torch.bool, device=self.device)
                            
                            # Always include I and S compartments
                            mask[:, I_idx] = True
                            mask[:, S_idx] = True
                            
                            # Randomly activate other compartments
                            other_indices = [i for i in range(num_compartments) if i != I_idx and i != S_idx]
                            perm = torch.randperm(len(other_indices))[:num_active-2]
                            for oidx in perm.tolist():
                                mask[:, other_indices[oidx]] = True
                            
                            try:
                                predictions = self.model(input_tokens, compartment_mask=mask, compute_R_t=False)
                                pred_I = predictions['I']
                                
                                if not (torch.isnan(pred_I).any() or torch.isinf(pred_I).any()):
                                    all_predictions.append(pred_I.cpu())
                            except Exception as e:
                                if idx == 0 and sample_idx < 5:
                                    print(f"  Sample {sample_idx} error: {str(e)[:50]}")
                        
                        if idx == 0:
                            print(f"  Collected {len(all_predictions)} valid predictions")
                    else:
                        # Single forward pass (no uncertainty)
                        sample_compartment_mask = batch.get('compartment_mask', None)
                        if sample_compartment_mask is not None:
                            sample_compartment_mask = sample_compartment_mask[batch_idx:batch_idx+1].to(self.device)
                        predictions = self.model(input_tokens, compartment_mask=sample_compartment_mask, compute_R_t=False)
                        pred_I = predictions['I']
                        all_predictions.append(pred_I.cpu())
                    
                    # Stack predictions
                    if all_predictions and len(all_predictions) > 1:
                        all_predictions = torch.stack(all_predictions)  # [num_samples, 1, seq_len, features]
                        pred_I_mean = all_predictions.mean(dim=0).squeeze(0)
                        pred_I_std = all_predictions.std(dim=0).squeeze(0)
                        
                        # Debug: Print statistics about uncertainty
                        if idx == 0:
                            print(f"Uncertainty stats - Min: {pred_I_std.min():.6f}, Max: {pred_I_std.max():.6f}, Mean: {pred_I_std.mean():.6f}")
                    elif all_predictions:
                        # Only one prediction, no uncertainty
                        pred_I_mean = all_predictions[0].squeeze(0)
                        pred_I_std = None
                    else:
                        continue
                    
                else:
                    # Regular CAPE model (no compartment masks, no uncertainty)
                    label_tokens = batch.get('target', batch.get('label'))
                    if label_tokens is None:
                        continue
                    
                    label_tokens = label_tokens[batch_idx:batch_idx+1].to(self.device)
                    full_sequence = torch.cat([input_tokens, label_tokens], dim=1)
                    input_seq = full_sequence[:, :-1, :]
                    target_I = full_sequence[:, 1:, :]
                    
                    predictions = self.model(input_seq)
                    pred_I_mean = predictions.cpu().squeeze(0)
                    pred_I_std = None
                
                # Convert to numpy and flatten time series
                pred_I_mean_np = pred_I_mean.numpy()
                if pred_I_std is not None:
                    pred_I_std_np = pred_I_std.numpy()
                else:
                    pred_I_std_np = None
                
                target_I_np = target_I.cpu().numpy().squeeze()
                
                # Get input for context (original input before any shifting)
                if use_shifted if 'use_shifted' in locals() else False:
                    # For shifted sequences, extract original input
                    orig_input = sample_info['input'].cpu().numpy().squeeze()
                else:
                    orig_input = input_tokens.cpu().numpy().squeeze()
                
                # Get sequence mask to determine valid (non-padded) length
                sequence_mask = batch.get('sequence_mask', None)
                if sequence_mask is not None:
                    # Get the valid length for this sample
                    sample_seq_mask = sequence_mask[batch_idx].cpu().numpy()
                    valid_length = int(sample_seq_mask.sum())  # Number of True values
                else:
                    valid_length = None  # No mask, use full length
                
                # Flatten tokens to time series (tokens x token_size -> full time series)
                if len(pred_I_mean_np.shape) == 2:  # [num_tokens, token_size]
                    pred_series = pred_I_mean_np.flatten()
                    if pred_I_std_np is not None:
                        std_series = pred_I_std_np.flatten()
                    else:
                        std_series = None
                elif len(pred_I_mean_np.shape) == 1:
                    pred_series = pred_I_mean_np
                    std_series = pred_I_std_np if pred_I_std_np is not None else None
                else:
                    pred_series = pred_I_mean_np.flatten()
                    std_series = pred_I_std_np.flatten() if pred_I_std_np is not None else None
                
                if len(target_I_np.shape) == 2:
                    target_series = target_I_np.flatten()
                else:
                    target_series = target_I_np
                
                if len(orig_input.shape) == 2:
                    input_series = orig_input.flatten()
                else:
                    input_series = orig_input
                
                # Truncate to valid length if sequence mask is available
                # valid_length is in tokens, need to convert to flattened time steps
                if valid_length is not None:
                    token_size = batch['input'].shape[-1] if len(batch['input'].shape) > 2 else 1
                    valid_timesteps = valid_length * token_size
                    
                    # Truncate all series to valid length
                    if len(pred_series) > valid_timesteps:
                        pred_series = pred_series[:valid_timesteps]
                    if std_series is not None and len(std_series) > valid_timesteps:
                        std_series = std_series[:valid_timesteps]
                    if len(target_series) > valid_timesteps:
                        target_series = target_series[:valid_timesteps]
                
                # Create time axis
                input_len = len(input_series)
                pred_len = len(pred_series)
                time_axis = np.arange(input_len + pred_len)
                pred_time_axis = np.arange(input_len, input_len + pred_len)
                
                # Plot
                ax = axes[idx]
                
                # Plot input (context)
                ax.plot(np.arange(input_len), input_series, 
                       label='Input (Context)', color='gray', linewidth=2, alpha=0.7)
                
                # Plot ground truth
                ax.plot(pred_time_axis, target_series,
                       label='Ground Truth', color='blue', linewidth=2, marker='o', markersize=3)
                
                # Plot prediction with uncertainty
                ax.plot(pred_time_axis, pred_series,
                       label='Prediction (Mean)', color='red', linewidth=2, linestyle='--', marker='x', markersize=3)
                
                # Plot uncertainty band (1 std deviation)
                if std_series is not None and len(pred_series) == len(std_series):
                    # Debug info for first sample
                    if idx == 0:
                        print(f"Plotting uncertainty - Pred len: {len(pred_series)}, Std len: {len(std_series)}")
                        print(f"Std range: [{np.min(std_series):.6f}, {np.max(std_series):.6f}], mean: {np.mean(std_series):.6f}")
                    
                    # Only plot if there's actual uncertainty (not all zeros)
                    if np.max(std_series) > 1e-6:
                        ax.fill_between(pred_time_axis, 
                                        pred_series - std_series, 
                                        pred_series + std_series,
                                        alpha=0.3, color='red', label='Uncertainty (±1σ)')
                    else:
                        if idx == 0:
                            print("Warning: Uncertainty is zero or too small to display")
                elif std_series is None:
                    if idx == 0:
                        print("No uncertainty available (std_series is None)")
                
                # Add vertical line to separate input from prediction
                ax.axvline(x=input_len, color='black', linestyle=':', linewidth=1, alpha=0.5)
                
                # Calculate metrics for this sample (only on valid, non-padded data)
                # Ensure both series have the same length for metrics
                min_metric_len = min(len(pred_series), len(target_series))
                pred_for_metrics = pred_series[:min_metric_len]
                target_for_metrics = target_series[:min_metric_len]
                
                mae = np.mean(np.abs(pred_for_metrics - target_for_metrics))
                mse = np.mean((pred_for_metrics - target_for_metrics)**2)
                
                # Calculate calibration metric if uncertainty available
                if std_series is not None:
                    # Ensure same length for calibration calculation
                    std_for_metrics = std_series[:min_metric_len] if len(std_series) > min_metric_len else std_series
                    # Percentage of ground truth points within 1 std
                    within_1std = np.mean((target_for_metrics >= pred_for_metrics - std_for_metrics) & 
                                         (target_for_metrics <= pred_for_metrics + std_for_metrics))
                    calibration_str = f' | Calibration: {within_1std*100:.1f}%'
                    avg_uncertainty = np.mean(std_for_metrics)
                else:
                    within_1std = np.nan
                    calibration_str = ''
                    avg_uncertainty = 0
                
                # Store sample results for numpy saving
                all_sample_results['input_series'].append(input_series)
                all_sample_results['pred_series'].append(pred_for_metrics)
                all_sample_results['target_series'].append(target_for_metrics)
                all_sample_results['std_series'].append(std_for_metrics if std_series is not None else np.array([]))
                all_sample_results['mae'].append(mae)
                all_sample_results['mse'].append(mse)
                all_sample_results['calibration'].append(within_1std)
                all_sample_results['avg_uncertainty'].append(avg_uncertainty)
                all_sample_results['valid_length'].append(valid_length if valid_length is not None else len(pred_series))
                
                # Get time resolution if available
                time_res = 'weekly'
                if 'time_resolution' in batch:
                    if isinstance(batch['time_resolution'], str):
                        time_res = batch['time_resolution']
                    elif hasattr(batch['time_resolution'], '__getitem__'):
                        time_res = batch['time_resolution'][batch_idx] if len(batch['time_resolution']) > batch_idx else 'weekly'
                
                ax.set_xlabel(f'Time Steps ({time_res})', fontsize=10)
                ax.set_ylabel('Infected (I) - Normalized', fontsize=10)
                
                # Title with metrics
                title = f'Sample {idx+1}/{num_samples} | MAE: {mae:.4f} | MSE: {mse:.4f}'
                if std_series is not None:
                    title += f' | Avg σ: {avg_uncertainty:.4f}{calibration_str}'
                title += f' | {time_res}'
                ax.set_title(title, fontsize=10)
                
                ax.legend(loc='best', fontsize=9)
                ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save figure
        epoch_str = f'_epoch{epoch}' if epoch is not None else ''
        filename = f'predictions_{phase}{epoch_str}.png'
        save_file = os.path.join(save_path, filename)
        plt.savefig(save_file, dpi=150, bbox_inches='tight')
        plt.close()
        
        # Save sample results to numpy file
        # Pad variable-length arrays to create regular arrays
        max_input_len = max(len(s) for s in all_sample_results['input_series']) if all_sample_results['input_series'] else 0
        max_pred_len = max(len(s) for s in all_sample_results['pred_series']) if all_sample_results['pred_series'] else 0
        max_std_len = max(len(s) for s in all_sample_results['std_series'] if len(s) > 0) if any(len(s) > 0 for s in all_sample_results['std_series']) else 0
        
        n_samples = len(all_sample_results['mae'])
        
        # Create padded arrays (pad with NaN for easy masking later)
        input_array = np.full((n_samples, max_input_len), np.nan)
        pred_array = np.full((n_samples, max_pred_len), np.nan)
        target_array = np.full((n_samples, max_pred_len), np.nan)
        std_array = np.full((n_samples, max_std_len if max_std_len > 0 else 1), np.nan)
        
        for i, (inp, pred, targ, std) in enumerate(zip(
            all_sample_results['input_series'],
            all_sample_results['pred_series'],
            all_sample_results['target_series'],
            all_sample_results['std_series']
        )):
            input_array[i, :len(inp)] = inp
            pred_array[i, :len(pred)] = pred
            target_array[i, :len(targ)] = targ
            if len(std) > 0:
                std_array[i, :len(std)] = std
        
        np_filename = f'predictions_{phase}{epoch_str}.npz'
        np_save_file = os.path.join(save_path, np_filename)
        np.savez(
            np_save_file,
            input_series=input_array,
            pred_series=pred_array,
            target_series=target_array,
            std_series=std_array,
            mae=np.array(all_sample_results['mae']),
            mse=np.array(all_sample_results['mse']),
            calibration=np.array(all_sample_results['calibration']),
            avg_uncertainty=np.array(all_sample_results['avg_uncertainty']),
            valid_length=np.array(all_sample_results['valid_length']),
            num_samples=np.array([n_samples]),
            epoch=np.array([epoch if epoch is not None else -1]),
            phase=np.array([phase])
        )
        
        print(f"Visualization saved to {save_file}")
        print(f"Sample results saved to {np_save_file} ({n_samples} samples)")
    
    def evaluate_with_uncertainty(self, data_loader, num_masks=20):
        """
        Evaluate with uncertainty estimation using multiple compartment masks
        Only applicable for CompartmentalCAPE models
        """
        if not self.is_compartmental:
            print("Uncertainty estimation only available for CompartmentalCAPE models")
            return self.evaluate(data_loader)
        
        self.model.eval()
        
        # Store predictions with different masks
        all_predictions = []  # List of [num_masks] predictions for each batch
        all_targets = []
        all_mae_results = []
        
        with torch.no_grad():
            progress_bar = tqdm(data_loader, desc="Evaluating with uncertainty", unit="batch", leave=False)
            for batch in progress_bar:
                input_tokens = batch['input'].to(self.device)
                
                # Get targets
                if 'target_compartments' in batch:
                    targets = {comp: batch['target_compartments'][comp].to(self.device) 
                              for comp in batch['target_compartments'].keys()}
                else:
                    # Real data - use shifted sequence
                    label_tokens = batch.get('target', batch.get('label'))
                    if label_tokens is None:
                        continue
                    
                    label_tokens = label_tokens.to(self.device)
                    full_sequence = torch.cat([input_tokens, label_tokens], dim=1)
                    input_tokens = full_sequence[:, :-1, :]
                    targets = {'I': full_sequence[:, 1:, :]}
                
                # Generate multiple random compartment masks
                batch_size = input_tokens.size(0)
                num_compartments = len(self.model_unwrapped.compartments)
                
                mask_predictions = []
                for mask_idx in range(num_masks):
                    # Random mask: randomly select 3-9 active compartments
                    num_active = torch.randint(3, min(10, num_compartments + 1), (1,)).item()
                    mask = torch.zeros(batch_size, num_compartments, dtype=torch.bool, device=self.device)
                    
                    # Always include I compartment
                    I_idx = self.model_unwrapped.compartments.index('I')
                    mask[:, I_idx] = True
                    
                    # Randomly activate other compartments
                    other_indices = [i for i in range(num_compartments) if i != I_idx]
                    perm = torch.randperm(len(other_indices))[:num_active-1]
                    for idx in perm.tolist():
                        mask[:, other_indices[idx]] = True
                    
                    # Forward pass with this mask
                    try:
                        predictions = self.model(input_tokens, compartment_mask=mask, compute_R_t=False)
                        
                        # Extract I compartment prediction
                        if 'I' in predictions:
                            pred_I = predictions['I']
                            # Check for NaN/Inf before adding
                            if torch.isnan(pred_I).any() or torch.isinf(pred_I).any():
                                # Skip this mask, it produced NaN
                                continue
                            mask_predictions.append(pred_I.cpu())
                    except Exception as e:
                        # Skip this mask if it causes an error
                        continue
                
                if mask_predictions:
                    # Stack predictions: [num_masks, batch, seq_len, features]
                    mask_predictions = torch.stack(mask_predictions)
                    all_predictions.append(mask_predictions)
                    all_targets.append(targets['I'].cpu())
        
        if not all_predictions:
            print("Warning: No predictions collected for uncertainty estimation")
            return self.evaluate(data_loader)
        
        # Concatenate all batches
        all_predictions = torch.cat(all_predictions, dim=1)  # [num_masks, total_samples, seq_len, features]
        all_targets = torch.cat(all_targets, dim=0)  # [total_samples, seq_len, features]
        
        # Debug: Check for NaN/Inf in predictions
        if torch.isnan(all_predictions).any() or torch.isinf(all_predictions).any():
            print(f"Warning: Found NaN or Inf in predictions")
            print(f"  NaN count: {torch.isnan(all_predictions).sum().item()}")
            print(f"  Inf count: {torch.isinf(all_predictions).sum().item()}")
        
        # Compute statistics
        mean_pred = all_predictions.mean(dim=0)  # [total_samples, seq_len, features]
        std_pred = all_predictions.std(dim=0)    # [total_samples, seq_len, features]
        
        # Compute metrics
        mse = torch.nn.functional.mse_loss(mean_pred, all_targets)
        mae = torch.nn.functional.l1_loss(mean_pred, all_targets)
        
        # Uncertainty metrics
        avg_uncertainty = std_pred.mean().item()
        max_uncertainty = std_pred.max().item()
        
        # Calibration: correlation between uncertainty and error
        errors = torch.abs(mean_pred - all_targets)
        flat_errors = errors.flatten()
        flat_uncertainty = std_pred.flatten()
        
        # Compute correlation
        correlation = torch.corrcoef(torch.stack([flat_errors, flat_uncertainty]))[0, 1].item()
        
        uncertainty_metrics = {
            'mean_uncertainty': avg_uncertainty,
            'max_uncertainty': max_uncertainty,
            'uncertainty_error_correlation': correlation
        }
        
        return mse.item(), mae.item(), uncertainty_metrics
    
    def save(self, path, epoch=None, save_config=False):
        """Save model checkpoint"""
        os.makedirs(path, exist_ok=True)
        
        checkpoint_data = {
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }
        
        if epoch is not None:
            checkpoint_data['epoch'] = epoch
        
        # Save with epoch number if provided
        if epoch is not None:
            checkpoint_file = os.path.join(path, f'checkpoint_epoch_{epoch}.pth')
        else:
            checkpoint_file = os.path.join(path, 'checkpoint.pth')
        
        torch.save(checkpoint_data, checkpoint_file)
        
        # Save model config if requested
        if save_config and self.config is not None:
            self.save_model_config(path)
    
    def save_model_config(self, path):
        """Save model configuration to JSON file"""
        import json
        
        config_data = {
            'model': 'CAPE',
            'token_size': self.token_size,
            'hidden_size': self.model.hidden_size if hasattr(self.model, 'hidden_size') else getattr(self.config, 'hidden_size', 256),
            'num_layers': self.model.num_layers if hasattr(self.model, 'num_layers') else getattr(self.config, 'layers', 4),
            'dropout': getattr(self.model, 'dropout', getattr(self.config, 'dropout', 0.1)),
            'max_tokens': getattr(self.config, 'max_tokens', 512),
        }
        
        # Add compartmental model specific config
        if self.is_compartmental:
            config_data['is_compartmental'] = True
            config_data['patch_encoder_type'] = getattr(self.model, 'patch_encoder_type', getattr(self.config, 'patch_encoder_type', 'transformer'))
            config_data['num_heads'] = getattr(self.model, 'num_heads', getattr(self.config, 'num_heads', 8))
            config_data['num_embeddings'] = getattr(self.model, 'num_embeddings', getattr(self.config, 'num_embeddings', 6))
        else:
            config_data['is_compartmental'] = False
            config_data['architecture'] = getattr(self.model, 'architecture', getattr(self.config, 'architecture', 'hybrid'))
        
        config_file = os.path.join(path, 'model_config.json')
        with open(config_file, 'w') as f:
            json.dump(config_data, f, indent=2)
        print(f"Model config saved to {config_file}")
    
    def load(self, path, epoch=None):
        """Load model checkpoint"""
        if epoch is not None:
            checkpoint_file = os.path.join(path, f'checkpoint_epoch_{epoch}.pth')
        else:
            checkpoint_file = os.path.join(path, 'checkpoint.pth')
        
        checkpoint = torch.load(checkpoint_file, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state_dict'], strict=False)
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    
    def test(self, data_loader):
        """Wrapper for compatibility with evaluation script - evaluates on data loader"""
        result = self.evaluate(data_loader)
        
        # evaluate() returns (avg_loss, avg_mae) or (avg_loss, avg_mae, compartment_losses, compartment_maes)
        if self.is_compartmental:
            avg_loss, avg_mae, compartment_losses, compartment_maes = result
            # Use MAE^2 as MSE approximation for compatibility
            mse = avg_mae ** 2
            return avg_loss, mse, avg_mae
        else:
            avg_loss, avg_mae = result
            # Use MAE^2 as MSE approximation for compatibility
            mse = avg_mae ** 2
            return avg_loss, mse, avg_mae
        


if __name__ == "__main__":
    setup_seed(15)
    config = Config()
    
    # Force next-token-prediction mode
    config.next_token_prediction = True

    # Set default exp name if not provided
    if config.exp is None:
        config.exp = 'next_token_pretrain'
    
    # Ensure CAPE model
    if 'CAPE' not in config.model:
        print("Warning: Model should be CAPE for next-token-prediction. Forcing CAPE model.")
        config.model = 'CAPE'
    
    # Set default pretrain path if not provided
    if config.pretrain_path is None:
        config.pretrain_path = f'checkpoints/pretraining/{config.exp}/'
    
    # Set default file path if not provided (use tycho_US.pt for diverse multi-disease pretraining)
    if config.file_path is None:
        config.file_path = 'data/tycho_US.pt'
        print(f"Using default dataset: {config.file_path}")
    
    # Force disease='All' for pretraining (use all diseases like LM pretraining)
    if config.disease != 'All':
        print(f"Note: Changing disease from '{config.disease}' to 'All' for pretraining")
        print("      (Pretraining uses all diseases, similar to language model pretraining)")
    config.disease = 'All'
    
    if not os.path.exists(config.pretrain_path):
        os.makedirs(config.pretrain_path)
    
    exp_path = f"results/{config.exp}/"
    if not os.path.exists(exp_path):
        os.makedirs(exp_path)
    save_path = f"results/{config.exp}/{config.model}_next_token/"
    if not os.path.exists(save_path):
        os.makedirs(save_path)
    
    print("="*80)
    print("NEXT-TOKEN-PREDICTION PRETRAINING")
    print("="*80)
    print(f"Model: {config.model}")
    print(f"Token size: {config.token_size}")
    print(f"Hidden size: {config.hidden_size}")
    print(f"Layers: {config.layers}")
    print(f"Batch size: {config.batch_size}")
    print(f"Learning rate: {config.learning_rate}")
    print(f"Epochs: {config.epochs}")
    print("="*80)
    
    # Load dataset with next-token-prediction mode
    # Note: lookback/horizon not used in next-token prediction - sequence length comes from data
    dataset = DataSetWrapper(
        lookback=104,  # Dummy value, not used in next-token mode
        horizon=52,    # Dummy value, not used in next-token mode
        batch_size=config.batch_size,
        cut_year=config.cut_year,
        valid_rate=config.valid_rate,
        data_path=config.file_path,
        disease=config.disease,
        num_features=config.num_features,
        max_length=512,  # Maximum sequence length to consider
        shuffle=config.shuffle,
        ahead=config.ahead
    )
    
    # Get next-token-prediction loaders
    if config.use_synthetic_data:
        print("\n" + "="*80)
        print("USING SYNTHETIC DATA MODE")
        if config.time_resolution == 'mixed':
            print(f"Time Resolution: MIXED (Weekly + Daily)")
            print(f"  - {(1-config.daily_ratio)*100:.0f}% weekly samples")
            print(f"  - {config.daily_ratio*100:.0f}% daily samples")
        else:
            print(f"Time Resolution: {config.time_resolution.upper()}")
        # NEW: Display GP augmentation settings
        if getattr(config, 'use_gp_augmentation', False):
            print(f"GP Augmentation: ENABLED ({getattr(config, 'gp_ratio', 0.2)*100:.0f}% GP samples)")
        else:
            print("GP Augmentation: DISABLED")
        print("="*80)
        train_loader, valid_loader, test_loader = dataset.get_synthetic_next_token_loaders(
            token_size=config.token_size,
            num_train=config.synthetic_num_train,
            num_valid=config.synthetic_num_valid,
            num_test=config.synthetic_num_test,
            univariate=config.synthetic_univariate,
            use_groups=getattr(config, 'synthetic_use_groups', True),
            group_ratio=getattr(config, 'synthetic_group_ratio', 0.5),
            min_compartments=config.synthetic_min_compartments,
            max_compartments=config.synthetic_max_compartments,
            min_transitions=config.synthetic_min_transitions,
            max_transitions=config.synthetic_max_transitions,
            min_weeks=config.synthetic_min_weeks,
            max_weeks=config.synthetic_max_weeks,
            time_resolution=config.time_resolution,
            daily_ratio=config.daily_ratio,
            rng_seed=config.synthetic_seed,
            streaming=config.synthetic_streaming,
            num_workers=config.num_workers,
            use_gp_augmentation=getattr(config, 'use_gp_augmentation', False),
            gp_ratio=getattr(config, 'gp_ratio', 0.2)
        )
    else:
        print("\n" + "="*80)
        print("USING REAL DATA MODE")
        print("="*80)
        train_loader, valid_loader, test_loader = dataset.get_next_token_pretrain_loaders(
            strategy=config.data_strategy,
            token_size=config.token_size
        )
    
    print("\nInitializing CAPE model...")
    model = create_cape_model(config)
    
    print(f"\nModel parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    print("\nCreating Next-Token-Prediction Trainer...")
    trainer = NextTokenPretrainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        test_loader=test_loader,
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
        device=config.device,
        token_size=config.token_size,
        config=config,
        use_multi_gpu=getattr(config, 'multi_gpu', True)
    )
    
    print("\nStarting Pretraining...")
    
    # Save model config at the start
    trainer.save_model_config(config.pretrain_path)
    
    best_loss = sys.maxsize
    
    # Track metrics at batch level
    batch_train_losses = []  # Loss for each batch
    batch_eval_losses = []   # Validation loss at evaluation points
    batch_eval_maes = []     # Validation MAE at evaluation points
    batch_eval_I_losses = [] # "I" compartment loss at evaluation points
    batch_numbers = []       # Batch numbers where evaluation happened
    eval_interval = config.eval_interval
    
    # Legacy epoch-level tracking (for compatibility)
    epoch_train_losses = []
    epoch_valid_losses = []
    epoch_valid_maes = []
    epoch_valid_I_losses = []  # Track I compartment loss per epoch
    compartment_loss_history = {}
    
    total_batches = 0
    epoch_progress = tqdm(range(config.epochs), desc="Overall Progress", unit="epoch")
    
    for epoch in epoch_progress:
        start_time = time.time()
        
        # Train with batch-level evaluation
        trainer.model.train()
        epoch_loss = 0
        epoch_batches = 0
        
        progress_bar = tqdm(trainer.train_loader, desc=f"Epoch {epoch+1}/{config.epochs}", unit="batch", leave=False)
        for batch_idx, batch in enumerate(progress_bar):
            # Get input tokens
            input_tokens = batch['input'].to(trainer.device)
            
            # Get compartment mask if available
            compartment_mask = batch.get('compartment_mask', None)
            if compartment_mask is not None:
                compartment_mask = compartment_mask.to(trainer.device)
            
            if trainer.is_compartmental:
                # Compartmental training
                has_R_t_data = 'target_R_t' in batch
                can_compute_R_t = trainer.compute_R_t and has_R_t_data
                
                predictions = trainer.model(input_tokens, compartment_mask=compartment_mask, compute_R_t=can_compute_R_t)
                
                # Get targets
                if 'target_compartments' in batch:
                    target_compartments = batch['target_compartments']
                    targets = {comp: target_compartments[comp].to(trainer.device) 
                              for comp in target_compartments.keys()}
                else:
                    target_tokens = batch.get('target', batch.get('label')).to(trainer.device)
                    targets = {'I': target_tokens}
                
                if can_compute_R_t:
                    target_R_t = batch['target_R_t'].to(trainer.device)
                    targets['R_t'] = target_R_t
                
                loss_weights = trainer.loss_weights.copy() if trainer.loss_weights else {}
                if can_compute_R_t:
                    loss_weights['R_t'] = trainer.R_t_loss_weight
                
                # Get sequence mask if available (to ignore padded positions)
                sequence_mask = batch.get('sequence_mask', None)
                if sequence_mask is not None:
                    sequence_mask = sequence_mask.to(trainer.device)
                
                mae_weight = getattr(trainer, 'mae_loss_weight', 0.0)
                loss, losses_dict = trainer.model_unwrapped.compute_loss(predictions, targets, loss_weights=loss_weights, compartment_mask=compartment_mask, sequence_mask=sequence_mask, mae_loss_weight=mae_weight)
            else:
                # Regular CAPE
                target_tokens = batch.get('target', batch.get('label')).to(trainer.device)
                predictions = trainer.model(input_tokens)
                loss = torch.nn.functional.mse_loss(predictions, target_tokens)
            
            # Backward pass
            trainer.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(trainer.model.parameters(), max_norm=1.0)
            trainer.optimizer.step()
            
            # Track batch loss
            batch_loss = loss.item()
            batch_train_losses.append(batch_loss)
            epoch_loss += batch_loss
            epoch_batches += 1
            total_batches += 1
            
            # Update progress bar
            progress_bar.set_postfix({'loss': f'{batch_loss:.6f}'})
            
            # Evaluate every N batches
            if total_batches % eval_interval == 0:
                trainer.model.eval()
                eval_result = trainer.evaluate(valid_loader, save_path=save_path, epoch=epoch+1, phase='valid')
                
                if trainer.is_compartmental:
                    eval_loss, eval_mae, eval_comp_losses, _ = eval_result
                    eval_I_loss = eval_comp_losses.get('I', eval_loss)
                else:
                    eval_loss, eval_mae = eval_result
                    eval_I_loss = eval_loss
                
                batch_eval_losses.append(eval_loss)
                batch_eval_maes.append(eval_mae)
                batch_eval_I_losses.append(eval_I_loss)
                batch_numbers.append(total_batches)
                
                tqdm.write(f"[Batch {total_batches}] Train Loss: {batch_loss:.6f} | Valid Loss: {eval_loss:.6f} | Valid MAE: {eval_mae:.6f} | I Loss: {eval_I_loss:.6f}")
                
                trainer.model.train()
        
        # End of epoch - compute epoch-level metrics
        train_loss = epoch_loss / epoch_batches if epoch_batches > 0 else 0
        epoch_train_losses.append(train_loss)
        
        # Final validation for this epoch (no visualization here to avoid duplicates)
        valid_result = trainer.evaluate(valid_loader)
        if trainer.is_compartmental:
            valid_loss, valid_mae, valid_comp_losses, valid_comp_maes = valid_result
        else:
            valid_loss, valid_mae = valid_result
            valid_comp_losses = {}
            valid_comp_maes = {}
        
        epoch_valid_losses.append(valid_loss)
        epoch_valid_maes.append(valid_mae)
        
        # Track I compartment loss
        if trainer.is_compartmental and valid_comp_losses:
            epoch_valid_I_losses.append(valid_comp_losses.get('I', valid_loss))
        else:
            epoch_valid_I_losses.append(valid_loss)
        
        # Update learning rate
        trainer.scheduler.step()
        current_lr = trainer.optimizer.param_groups[0]['lr']
        
        # Save model for this epoch
        trainer.save(config.pretrain_path, epoch=epoch+1, save_config=False)
        
        # Save best model
        if valid_loss < best_loss:
            best_loss = valid_loss
            trainer.save(config.pretrain_path, save_config=True)  # Save as best (checkpoint.pth) with config
            tqdm.write(f"✓ Best model saved at epoch {epoch+1}")
        
        # Log progress
        elapsed = time.time() - start_time
        log_msg = (f"Epoch [{epoch+1}/{config.epochs}] "
                   f"Train Loss: {train_loss:.6f} | "
                   f"Valid Loss: {valid_loss:.6f} | "
                   f"Valid MAE: {valid_mae:.6f} | "
                   f"LR: {current_lr:.2e} | "
                   f"Time: {elapsed:.2f}s")
        
        # Add compartment losses if available
        if trainer.is_compartmental and valid_comp_losses:
            comp_strs = [f"{comp}: {loss:.6f}" for comp, loss in valid_comp_losses.items()]
            log_msg += f"\n  Compartment losses: {', '.join(comp_strs)}"
        
        tqdm.write(log_msg)
        
        # Update epoch progress bar
        epoch_progress.set_postfix({
            'train_loss': f'{train_loss:.6f}',
            'valid_loss': f'{valid_loss:.6f}',
            'best': f'{best_loss:.6f}'
        })
        
        # Save training curves periodically (batch-level)
        if (epoch + 1) % 1 == 0 and len(batch_numbers) > 0:
            fig = plt.figure(figsize=(20, 10))
            
            # Batch-level training curve (overall)
            plt.subplot(2, 3, 1)
            plt.plot(range(1, len(batch_train_losses)+1), batch_train_losses, 
                    label='Train Loss (per batch)', alpha=0.3, linewidth=0.5)
            plt.plot(batch_numbers, batch_eval_losses, label='Valid Loss (Total)', 
                    marker='o', markersize=3, linewidth=2)
            plt.xlabel('Batch Number')
            plt.ylabel('MSE Loss')
            plt.title('Total Loss (Batch-level)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Batch-level I compartment loss
            plt.subplot(2, 3, 2)
            plt.plot(batch_numbers, batch_eval_I_losses, label='Valid I Loss', 
                    marker='s', markersize=3, linewidth=2, color='green')
            plt.xlabel('Batch Number')
            plt.ylabel('MSE Loss')
            plt.title('I Compartment Loss (Batch-level)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Batch-level MAE
            plt.subplot(2, 3, 3)
            plt.plot(batch_numbers, batch_eval_maes, label='Valid MAE', 
                    marker='o', markersize=3, linewidth=2, color='orange')
            plt.xlabel('Batch Number')
            plt.ylabel('MAE')
            plt.title('Validation MAE (Batch-level)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Epoch-level total loss
            plt.subplot(2, 3, 4)
            plt.plot(range(1, len(epoch_train_losses)+1), epoch_train_losses, 
                    label='Train Loss', marker='o')
            plt.plot(range(1, len(epoch_valid_losses)+1), epoch_valid_losses, 
                    label='Valid Loss (Total)', marker='s')
            plt.xlabel('Epoch')
            plt.ylabel('MSE Loss')
            plt.title('Total Loss (Epoch-level)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            # Epoch-level I compartment loss
            if len(epoch_valid_I_losses) > 0:
                plt.subplot(2, 3, 5)
                plt.plot(range(1, len(epoch_valid_I_losses)+1), epoch_valid_I_losses, 
                        label='Valid I Loss', marker='s', color='green')
                plt.xlabel('Epoch')
                plt.ylabel('MSE Loss')
                plt.title('I Compartment Loss (Epoch-level)')
                plt.legend()
                plt.grid(True, alpha=0.3)
            
            # Epoch-level MAE
            plt.subplot(2, 3, 6)
            plt.plot(range(1, len(epoch_valid_maes)+1), epoch_valid_maes, 
                    label='Valid MAE', marker='s', color='orange')
            plt.xlabel('Epoch')
            plt.ylabel('MAE')
            plt.title('Validation MAE (Epoch-level)')
            plt.legend()
            plt.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.savefig(os.path.join(save_path, 'training_curves.png'), dpi=150, bbox_inches='tight')
            plt.close()
            
            # Save metrics to numpy file for easy loading and analysis
            np.savez(
                os.path.join(save_path, 'training_metrics.npz'),
                # Batch-level metrics
                batch_train_losses=np.array(batch_train_losses),
                batch_eval_losses=np.array(batch_eval_losses),
                batch_eval_maes=np.array(batch_eval_maes),
                batch_eval_I_losses=np.array(batch_eval_I_losses),
                batch_numbers=np.array(batch_numbers),
                eval_interval=np.array([eval_interval]),
                # Epoch-level metrics
                epoch_train_losses=np.array(epoch_train_losses),
                epoch_valid_losses=np.array(epoch_valid_losses),
                epoch_valid_maes=np.array(epoch_valid_maes),
                epoch_valid_I_losses=np.array(epoch_valid_I_losses),
                # Current best
                best_valid_loss=np.array([best_loss]),
                current_epoch=np.array([epoch + 1])
            )
    
    print("\n" + "="*80)
    print("Pretraining Complete!")
    print(f"Best validation loss: {best_loss:.6f}")
    
    # Final evaluation on test set
    print("\nEvaluating on test set...")
    trainer.load(config.pretrain_path)
    test_result = trainer.evaluate(test_loader, save_path=save_path, epoch='final', phase='test')
    if trainer.is_compartmental:
        test_loss, test_mae, test_comp_losses, test_comp_maes = test_result
        print(f"Test Loss (MSE): {test_loss:.6f}")
        print(f"Test MAE: {test_mae:.6f}")
        if test_comp_losses:
            print("\nCompartment-wise test metrics:")
            for comp in sorted(test_comp_losses.keys()):
                print(f"  {comp}: Loss={test_comp_losses[comp]:.6f}, MAE={test_comp_maes.get(comp, 0):.6f}")
    else:
        test_loss, test_mae = test_result
        print(f"Test Loss (MSE): {test_loss:.6f}")
        print(f"Test MAE: {test_mae:.6f}")
    
    # Save metrics (batch-level and epoch-level)
    metrics = {
        # Batch-level metrics (for detailed training curves)
        'batch_train_losses': batch_train_losses,
        'batch_eval_losses': batch_eval_losses,
        'batch_eval_maes': batch_eval_maes,
        'batch_eval_I_losses': batch_eval_I_losses,
        'batch_numbers': batch_numbers,
        'eval_interval': eval_interval,
        
        # Epoch-level metrics (for summary)
        'epoch_train_losses': epoch_train_losses,
        'epoch_valid_losses': epoch_valid_losses,
        'epoch_valid_maes': epoch_valid_maes,
        'epoch_valid_I_losses': epoch_valid_I_losses,
        
        # Test metrics
        'test_loss': test_loss,
        'test_mae': test_mae,
        'best_valid_loss': best_loss
    }
    
    # Add compartmental metrics if available
    if trainer.is_compartmental:
        if test_comp_losses:
            metrics['test_compartment_losses'] = test_comp_losses
        if test_comp_maes:
            metrics['test_compartment_maes'] = test_comp_maes
    
    import json
    with open(os.path.join(save_path, 'metrics.json'), 'w') as f:
        json.dump(metrics, f, indent=2)
    
    # Save final metrics to numpy file
    final_metrics_dict = {
        # Batch-level metrics
        'batch_train_losses': np.array(batch_train_losses),
        'batch_eval_losses': np.array(batch_eval_losses),
        'batch_eval_maes': np.array(batch_eval_maes),
        'batch_eval_I_losses': np.array(batch_eval_I_losses),
        'batch_numbers': np.array(batch_numbers),
        'eval_interval': np.array([eval_interval]),
        # Epoch-level metrics
        'epoch_train_losses': np.array(epoch_train_losses),
        'epoch_valid_losses': np.array(epoch_valid_losses),
        'epoch_valid_maes': np.array(epoch_valid_maes),
        'epoch_valid_I_losses': np.array(epoch_valid_I_losses),
        # Test metrics
        'test_loss': np.array([test_loss]),
        'test_mae': np.array([test_mae]),
        'best_valid_loss': np.array([best_loss])
    }
    
    # Add compartmental metrics if available
    if trainer.is_compartmental:
        if test_comp_losses:
            for comp, loss in test_comp_losses.items():
                final_metrics_dict[f'test_loss_{comp}'] = np.array([loss])
        if test_comp_maes:
            for comp, mae in test_comp_maes.items():
                final_metrics_dict[f'test_mae_{comp}'] = np.array([mae])
    
    np.savez(os.path.join(save_path, 'final_metrics.npz'), **final_metrics_dict)
    
    print(f"\nResults saved to {save_path}")
    print(f"  - training_curves.png (visualization)")
    print(f"  - training_metrics.npz (numpy, updated during training)")
    print(f"  - final_metrics.npz (numpy, complete results)")
    print(f"  - metrics.json (JSON format)")
    print("="*80)
