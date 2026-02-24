#!/usr/bin/env python
"""
Second-Stage Pretraining on Real-World Disease Data
====================================================
Continues pretraining CAPE model on the first 20% of time series from all diseases
in the Tycho dataset. This bridges the gap between synthetic pretraining and
disease-specific finetuning.

Usage:
    python second_stage_pretrain.py --backbone_path checkpoints/pretraining/next_token_pretrain_v5/checkpoint.pth
    python second_stage_pretrain.py --backbone_path ... --epochs 50 --lr 1e-4
    python second_stage_pretrain.py --backbone_path ... --pretrain_ratio 0.3
"""

import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import matplotlib.pyplot as plt

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from model.CAPE_Compartmental import CompartmentalCAPE


def setup_seed(seed=15):
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# =============================================================================
# DATA LOADING
# =============================================================================

class MultiDiseasePretrainDataset(Dataset):
    """Dataset for second-stage pretraining using first N% of all diseases."""
    
    def __init__(self, tokens_list: list, num_input: int, num_output: int):
        """
        Args:
            tokens_list: List of (disease_name, tokens_array) tuples
            num_input: Number of input tokens
            num_output: Number of output tokens
        """
        self.samples = []
        total = num_input + num_output
        
        for disease_name, tokens in tokens_list:
            # Create sliding window samples from each disease
            for i in range(len(tokens) - total + 1):
                self.samples.append({
                    'disease': disease_name,
                    'input': tokens[i:i+num_input],
                    'target': tokens[i+num_input:i+total]
                })
        
        print(f"Created {len(self.samples)} training samples from {len(tokens_list)} diseases")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        return {
            'input': torch.tensor(sample['input'], dtype=torch.float32),
            'label': torch.tensor(sample['target'], dtype=torch.float32)
        }


## ============================================================================
# IN-DOMAIN / OUT-OF-DOMAIN DISEASE SPLIT
# ============================================================================
# IN-DOMAIN (for pretraining): 8 diseases with long time spans (50+ years)
# spanning both historical and modern eras - rich for learning diverse patterns
INDOMAIN_DISEASES = [
    "Pertussis",        # 1888→2017, 129 years, 109,761 records
    "Varicella",        # 1889→2017, 128 years, 33,298 records
    "Tuberculosis",     # 1890→2014, 124 years, 95,564 records
    "measle",           # 1888→2001, 113 years, 151,867 records
    "TyphoidFever",     # 1888→1983, 95 years, 89,868 records
    "Mumps",            # 1924→2017, 93 years, 50,215 records
    "Diphtheria",       # 1888→1981, 93 years, 112,037 records
    "ScarletFever",     # 1888→1969, 81 years, 129,460 records
]

# OUT-OF-DOMAIN (held out for evaluation): 9 diseases
# Group A: Historical-only (ends before 1972)
#   - Smallpox, Influenza, Pneumonia, AcutePoliomyelitis, MeningococcalMeningitis
# Group B: Modern-only (starts after 1950)
#   - Gonorrhea, HepatitisA, HepatitisB, Rubella
OUTDOMAIN_DISEASES = [
    # Historical-only
    "Smallpox",                 # 1888→1952, 64 years
    "Influenza",                # 1919→1951, 32 years
    "Pneumonia",                # 1912→1951, 39 years
    "AcutePoliomyelitis",       # 1912→1971, 59 years
    "MeningococcalMeningitis",  # 1926→1964, 38 years
    # Modern-only
    "Gonorrhea",                # 1972→2017, 45 years
    "HepatitisA",               # 1966→2007, 41 years
    "HepatitisB",               # 1952→2007, 55 years
    "Rubella",                  # 1966→2017, 51 years
]


def load_all_diseases_pretrain_data(data_path, token_size=4, pretrain_ratio=0.2, valid_ratio=0.05, 
                                     use_indomain_only=True):
    """Load the first pretrain_ratio of each disease's time series for second-stage pretraining.
    
    Args:
        data_path: Path to tycho_US.pt
        token_size: Size of each token
        pretrain_ratio: Fraction of each disease's data to use (default 20%)
        valid_ratio: Fraction for validation (from pretrain portion)
        use_indomain_only: If True, only use INDOMAIN_DISEASES for pretraining (default True)
    
    Returns:
        train_tokens_list: List of (disease_name, train_tokens) tuples
        valid_tokens_list: List of (disease_name, valid_tokens) tuples
        scalers: Dict of {disease_name: scaler}
    """
    data = torch.load(data_path, weights_only=False)
    
    train_tokens_list = []
    valid_tokens_list = []
    scalers = {}
    
    # Get disease names - filter to in-domain only if requested
    all_diseases = list(data.keys())
    if use_indomain_only:
        diseases = [d for d in all_diseases if d in INDOMAIN_DISEASES]
        print(f"[IN-DOMAIN ONLY] Using {len(diseases)} in-domain diseases for pretraining:")
        print(f"  {diseases}")
        print(f"  Held out (out-of-domain): {[d for d in all_diseases if d in OUTDOMAIN_DISEASES]}")
    else:
        diseases = all_diseases
        print(f"[ALL DISEASES] Using all {len(diseases)} diseases: {diseases}")
    
    for disease in diseases:
        disease_data = data[disease]
        
        # Aggregate data across all locations (same as strategy_improved.py)
        total = {}
        for values in disease_data.values():
            for w, t in enumerate(values[0][1]):
                total[t.item()] = total.get(t.item(), 0) + int(values[0][0][w].item())
        
        values = np.array([total[k] for k in sorted(total)], dtype=np.float32)
        
        if len(values) < token_size * 10:  # Skip diseases with too little data
            print(f"  Skipping {disease}: only {len(values)} time points")
            continue
        
        # Normalize
        scaler = StandardScaler()
        values_norm = scaler.fit_transform(values.reshape(-1, 1)).flatten()
        scalers[disease] = scaler
        
        # Tokenize
        num_tokens = len(values_norm) // token_size
        tokens = values_norm[:num_tokens * token_size].reshape(num_tokens, token_size)
        
        # Take first pretrain_ratio of tokens
        pretrain_end = int(num_tokens * pretrain_ratio)
        pretrain_tokens = tokens[:pretrain_end]
        
        if len(pretrain_tokens) < 10:  # Need at least 10 tokens
            print(f"  Skipping {disease}: only {len(pretrain_tokens)} pretrain tokens")
            continue
        
        # Split pretrain portion into train/valid
        train_end = int(len(pretrain_tokens) * (1 - valid_ratio))
        train_tokens = pretrain_tokens[:train_end]
        valid_tokens = pretrain_tokens[train_end:]
        
        train_tokens_list.append((disease, train_tokens))
        valid_tokens_list.append((disease, valid_tokens))
        
        print(f"  {disease}: {num_tokens} total tokens, using {len(train_tokens)} train / {len(valid_tokens)} valid")
    
    return train_tokens_list, valid_tokens_list, scalers


def load_pretrained_cape(pretrain_path, device):
    """Load pretrained CAPE model from checkpoint."""
    checkpoint = torch.load(pretrain_path, map_location=device, weights_only=False)
    config_path = os.path.join(os.path.dirname(pretrain_path), 'model_config.json')
    
    if os.path.exists(config_path):
        with open(config_path) as f:
            cfg = json.load(f)
    else:
        cfg = {}
    
    model = CompartmentalCAPE(
        input_size=cfg.get('token_size', 4),
        hidden_size=cfg.get('hidden_dim', cfg.get('hidden_size', 512)),
        num_layers=cfg.get('num_layers', 4),
        num_heads=cfg.get('num_heads', 4),
        num_embeddings=cfg.get('num_embeddings', 6),
        patch_encoder_type=cfg.get('patch_encoder_type', 'transformer')
    )
    
    # Load state dict
    state = {k.replace('module.', ''): v for k, v in checkpoint['model_state_dict'].items()}
    model.load_state_dict(state, strict=False)
    
    print(f"Loaded pretrained model from {pretrain_path}")
    print(f"  Config: {cfg}")
    
    return model.to(device), cfg


# =============================================================================
# TRAINING
# =============================================================================

class SecondStagePretrainer:
    """Trainer for second-stage pretraining on real-world data."""
    
    def __init__(self, model, train_loader, valid_loader, lr, weight_decay, device, token_size=4):
        self.device = device
        self.token_size = token_size
        self.model = model.to(device)
        self.train_loader = train_loader
        self.valid_loader = valid_loader
        
        # Optimizer with lower learning rate for continued pretraining
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
        
        # Cosine annealing scheduler
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=5, T_mult=2, eta_min=lr * 0.01
        )
        
        # Loss function
        self.criterion = nn.HuberLoss(delta=1.0)
    
    def train_epoch(self):
        """Train for one epoch."""
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        progress_bar = tqdm(self.train_loader, desc="Training", leave=False)
        for batch in progress_bar:
            input_tokens = batch['input'].to(self.device)
            label_tokens = batch['label'].to(self.device)
            
            # Concatenate for shifted sequence training (next-token prediction)
            full_sequence = torch.cat([input_tokens, label_tokens], dim=1)
            input_seq = full_sequence[:, :-1, :]
            target_seq = full_sequence[:, 1:, :]
            
            # Create compartment mask with only I and S active (real data only has I compartment)
            batch_size = input_seq.size(0)
            num_compartments = len(self.model.compartments)
            compartment_mask = torch.zeros(batch_size, num_compartments, dtype=torch.bool, device=self.device)
            # Only activate I and S compartments
            I_idx = self.model.compartments.index('I')
            S_idx = self.model.compartments.index('S') if 'S' in self.model.compartments else 0
            compartment_mask[:, I_idx] = True
            compartment_mask[:, S_idx] = True
            
            # Forward pass
            predictions = self.model(input_seq, compartment_mask=compartment_mask, compute_R_t=False)
            
            # Use I compartment prediction as output (real data only has I)
            pred_I = predictions['I']
            
            # Compute loss
            loss = self.criterion(pred_I, target_seq)
            
            if torch.isnan(loss) or torch.isinf(loss):
                print(f"Warning: NaN/Inf loss, skipping batch")
                continue
            
            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            progress_bar.set_postfix({'loss': f'{loss.item():.6f}'})
        
        self.scheduler.step()
        
        return total_loss / num_batches if num_batches > 0 else 0
    
    def evaluate(self):
        """Evaluate on validation set."""
        self.model.eval()
        total_loss = 0
        total_mae = 0
        num_batches = 0
        
        with torch.no_grad():
            for batch in self.valid_loader:
                input_tokens = batch['input'].to(self.device)
                label_tokens = batch['label'].to(self.device)
                
                full_sequence = torch.cat([input_tokens, label_tokens], dim=1)
                input_seq = full_sequence[:, :-1, :]
                target_seq = full_sequence[:, 1:, :]
                
                # Create compartment mask with only I and S active
                batch_size = input_seq.size(0)
                num_compartments = len(self.model.compartments)
                compartment_mask = torch.zeros(batch_size, num_compartments, dtype=torch.bool, device=self.device)
                I_idx = self.model.compartments.index('I')
                S_idx = self.model.compartments.index('S') if 'S' in self.model.compartments else 0
                compartment_mask[:, I_idx] = True
                compartment_mask[:, S_idx] = True
                
                predictions = self.model(input_seq, compartment_mask=compartment_mask, compute_R_t=False)
                pred_I = predictions['I']
                
                loss = self.criterion(pred_I, target_seq)
                mae = torch.mean(torch.abs(pred_I - target_seq))
                
                total_loss += loss.item()
                total_mae += mae.item()
                num_batches += 1
        
        return total_loss / num_batches, total_mae / num_batches
    
    def save(self, save_dir, epoch=None, save_config=True, original_config=None):
        """Save model checkpoint."""
        os.makedirs(save_dir, exist_ok=True)
        
        if epoch is not None:
            checkpoint_path = os.path.join(save_dir, f'checkpoint_epoch_{epoch}.pth')
        else:
            checkpoint_path = os.path.join(save_dir, 'checkpoint.pth')
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'epoch': epoch
        }, checkpoint_path)
        
        if save_config and original_config:
            config_path = os.path.join(save_dir, 'model_config.json')
            with open(config_path, 'w') as f:
                json.dump(original_config, f, indent=2)


def main():
    parser = argparse.ArgumentParser(
        description='Second-Stage Pretraining on Real-World Disease Data',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Basic usage with default settings
  python second_stage_pretrain.py --backbone_path checkpoints/pretraining/next_token_pretrain_v5/checkpoint.pth
  
  # Custom training settings
  python second_stage_pretrain.py --backbone_path ... --epochs 100 --lr 5e-5 --pretrain_ratio 0.25
  
  # Quick test
  python second_stage_pretrain.py --backbone_path ... --epochs 5 --quick
"""
    )
    
    parser.add_argument('--backbone_path', type=str, required=True,
                        help='Path to first-stage pretrained checkpoint')
    parser.add_argument('--data_path', type=str, default='data/tycho_US.pt',
                        help='Path to Tycho dataset')
    parser.add_argument('--output_dir', type=str, default='checkpoints/pretraining/second_stage_pretrain',
                        help='Output directory for checkpoints')
    parser.add_argument('--pretrain_ratio', type=float, default=0.2,
                        help='Fraction of each disease time series to use (default: 0.2)')
    parser.add_argument('--valid_ratio', type=float, default=0.1,
                        help='Validation split from pretrain portion (default: 0.1)')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Number of training epochs (default: 50)')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Learning rate (default: 1e-4)')
    parser.add_argument('--weight_decay', type=float, default=1e-4,
                        help='Weight decay (default: 1e-4)')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size (default: 32)')
    parser.add_argument('--num_input_tokens', type=int, default=8,
                        help='Number of input tokens (default: 8)')
    parser.add_argument('--num_output_tokens', type=int, default=1,
                        help='Number of output tokens (default: 1)')
    parser.add_argument('--token_size', type=int, default=4,
                        help='Token size (default: 4)')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (default: cuda)')
    parser.add_argument('--seed', type=int, default=15,
                        help='Random seed (default: 15)')
    parser.add_argument('--save_every', type=int, default=10,
                        help='Save checkpoint every N epochs (default: 10)')
    parser.add_argument('--quick', action='store_true',
                        help='Quick mode with fewer epochs')
    parser.add_argument('--use_indomain_only', action='store_true', default=True,
                        help='Use only in-domain diseases for pretraining (default: True)')
    parser.add_argument('--use_all_diseases', action='store_true',
                        help='Use all diseases for pretraining (overrides --use_indomain_only)')
    
    args = parser.parse_args()
    
    if args.quick:
        args.epochs = 5
        args.save_every = 1
    
    setup_seed(args.seed)
    
    print("=" * 80)
    print("SECOND-STAGE PRETRAINING ON REAL-WORLD DATA")
    print("=" * 80)
    print(f"Backbone: {args.backbone_path}")
    print(f"Data path: {args.data_path}")
    print(f"Output directory: {args.output_dir}")
    print(f"Pretrain ratio: {args.pretrain_ratio}")
    print(f"Learning rate: {args.lr}")
    print(f"Epochs: {args.epochs}")
    print(f"Batch size: {args.batch_size}")
    print("=" * 80)
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Load data
    use_indomain = args.use_indomain_only and not args.use_all_diseases
    print(f"\nLoading data (in-domain only: {use_indomain})...")
    train_tokens_list, valid_tokens_list, scalers = load_all_diseases_pretrain_data(
        args.data_path,
        token_size=args.token_size,
        pretrain_ratio=args.pretrain_ratio,
        valid_ratio=args.valid_ratio,
        use_indomain_only=use_indomain
    )
    
    # Create datasets
    train_dataset = MultiDiseasePretrainDataset(
        train_tokens_list, args.num_input_tokens, args.num_output_tokens
    )
    valid_dataset = MultiDiseasePretrainDataset(
        valid_tokens_list, args.num_input_tokens, args.num_output_tokens
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"\nTrain samples: {len(train_dataset)}")
    print(f"Valid samples: {len(valid_dataset)}")
    
    # Load pretrained model
    print("\nLoading pretrained model...")
    model, original_config = load_pretrained_cape(args.backbone_path, args.device)
    print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
    
    # Create trainer
    trainer = SecondStagePretrainer(
        model=model,
        train_loader=train_loader,
        valid_loader=valid_loader,
        lr=args.lr,
        weight_decay=args.weight_decay,
        device=args.device,
        token_size=args.token_size
    )
    
    # Training loop
    print("\nStarting second-stage pretraining...")
    best_loss = float('inf')
    train_losses = []
    valid_losses = []
    valid_maes = []
    
    for epoch in range(args.epochs):
        start_time = time.time()
        
        # Train
        train_loss = trainer.train_epoch()
        train_losses.append(train_loss)
        
        # Evaluate
        valid_loss, valid_mae = trainer.evaluate()
        valid_losses.append(valid_loss)
        valid_maes.append(valid_mae)
        
        elapsed = time.time() - start_time
        
        # Log progress
        print(f"Epoch [{epoch+1}/{args.epochs}] "
              f"Train Loss: {train_loss:.6f} | "
              f"Valid Loss: {valid_loss:.6f} | "
              f"Valid MAE: {valid_mae:.6f} | "
              f"LR: {trainer.optimizer.param_groups[0]['lr']:.2e} | "
              f"Time: {elapsed:.2f}s")
        
        # Save best model
        if valid_loss < best_loss:
            best_loss = valid_loss
            trainer.save(args.output_dir, save_config=True, original_config=original_config)
            print(f"  ✓ Best model saved (loss: {best_loss:.6f})")
        
        # Save periodic checkpoint
        if (epoch + 1) % args.save_every == 0:
            trainer.save(args.output_dir, epoch=epoch+1, save_config=False, original_config=original_config)
    
    # Save final checkpoint
    trainer.save(args.output_dir, epoch=args.epochs, save_config=False, original_config=original_config)
    
    # Plot training curves
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    axes[0].plot(range(1, len(train_losses)+1), train_losses, label='Train Loss')
    axes[0].plot(range(1, len(valid_losses)+1), valid_losses, label='Valid Loss')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Training and Validation Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(range(1, len(valid_maes)+1), valid_maes, label='Valid MAE', color='orange')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('MAE')
    axes[1].set_title('Validation MAE')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, 'training_curves.png'), dpi=150)
    plt.close()
    
    print("\n" + "=" * 80)
    print("SECOND-STAGE PRETRAINING COMPLETE")
    print("=" * 80)
    print(f"Best validation loss: {best_loss:.6f}")
    print(f"Checkpoint saved to: {args.output_dir}/checkpoint.pth")
    print(f"\nTo use this model in run_improved_comparison_varied_trainrate.sh:")
    print(f'  PRETRAIN_PATH="{args.output_dir}/checkpoint.pth"')
    print("=" * 80)


if __name__ == "__main__":
    main()
