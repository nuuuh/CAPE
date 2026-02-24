import argparse
import torch
import numpy as np
import random

def str2bool(v):
    """Proper bool parser for argparse."""
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')

def Config():
    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument(
        "--file_path",
        default=None,
        type=str,
        required=False,
        help="The input data path.",
    )
    parser.add_argument(
        "--disease",
        default='Mumps',
        type=str,
        required=False,
        help="",
    )
    parser.add_argument(
        "--model",
        default='Dlinear',
        type=str,
        required=False,
        help="",
    )
    parser.add_argument(
        "--pretrain_path",
        default=None,
        type=str,
        required=False,
        help="The storage path of the pre-trained model.",
    )
    parser.add_argument(
        "--finetune_path",
        default=None,
        type=str,
        required=False,
        help="The output directory where the fine-tuning checkpoints will be written.",
    )
    parser.add_argument(
        "--lookback",
        default=64,
        type=int,
        help="")
    parser.add_argument(
        "--horizon",
        default=14,
        type=int,
        help="")
    parser.add_argument(
        "--norm",
        default=True,
        type=str2bool,
        help="")
    parser.add_argument(
        "--patching",
        default=0,
        type=int,
        help="")
    parser.add_argument(
        "--patch_len",
        default=12,
        type=int,
        help="")
    parser.add_argument(
        "--stride",
        default=1,
        type=int,
        help="")
    parser.add_argument(
        "--masking",
        default=True,
        type=str2bool,
        help="")
    parser.add_argument(
        "--train_rate",
        default=0.3,
        type=float,
        help="Proportion of data for training (default: 0.7)")
    parser.add_argument(
        "--valid_rate",
        default=0.1,
        type=float,
        help="Proportion of data for validation (default: 0.1, test gets the rest)")
    parser.add_argument(
        "--cut_year",
        default=None,
        type=float,
        help="Optional: Year cutoff for train/test split (if None, uses train_rate)")
    parser.add_argument(
        "--max_length",
        default=128,
        type=int,
        help="The maximum length of input time series. Sequences longer "
        "than this will be truncated, sequences shorter will be padded.",
    )
    parser.add_argument(
        "--num_features",
        default=1,
        type=int,
        help="",
    )

    parser.add_argument(
        "--epochs",
        default=100,
        type=int,
        help="",
    )
    parser.add_argument(
        "--batch_size",
        default=64,
        type=int,
        help="",
    )
    parser.add_argument(
        "--hidden_size",
        default=256,
        type=int,
        help="",
    )
    parser.add_argument(
        "--layers",
        default=3,
        type=int,
        help="",
    )
    parser.add_argument(
        "--attn_heads",
        default=8,
        type=int,
        help="",
    )
    parser.add_argument(
        "--learning_rate",
        default=2e-5,
        type=float,
        help="",
    )
    parser.add_argument(
        "--weight_decay",
        default=1e-3,
        type=float,
        help="",
    )
    parser.add_argument(
        "--dropout",
        default=0.1,
        type=float,
        help="",
    )
    parser.add_argument(
        "--use_revin",
        default=False,
        type=lambda x: x.lower() in ['true', '1', 'yes'],
        help="Enable RevIN (Reversible Instance Normalization) for CAPE",
    )
    parser.add_argument(
        "--revin_affine",
        default=True,
        type=lambda x: x.lower() in ['true', '1', 'yes'],
        help="Whether RevIN has learnable affine parameters",
    )
    parser.add_argument(
        "--device",
        default='cuda' if torch.cuda.is_available() else 'cpu',
        type=str,
        help="Device to use (cuda or cpu)",
    )
    parser.add_argument(
        "--multi_gpu",
        default=True,
        type=lambda x: x.lower() in ['true', '1', 'yes'],
        help="Enable multi-GPU training with DataParallel",
    )
    parser.add_argument(
        "--mask_prob",
        default=0.2,
        type=float,
        help="",
    )
    parser.add_argument(
        "--exp",
        default=None,
        type=str,
        help="",
    )
    parser.add_argument(
        "--gamma",
        default=0,
        type=float,
        help="",
    )
    parser.add_argument(
        "--data_strategy",
        default='SUM',
        type=str,
        help="",
    )
    parser.add_argument(
        "--random_mask",
        default=0.0,
        type=float,
        help="",
    )
    parser.add_argument(
        "--peak_mask",
        default=0.0,
        type=float,
        help="Probability of applying peak masking (masking around the peak value)",
    )
    parser.add_argument(
        "--shuffle",
        default=0,
        type=int,
        help="",
    )
    parser.add_argument(
        "--ahead",
        default=0,
        type=int,
        help="",
    )
    parser.add_argument(
        "--num_envs",
        default=5,
        type=int,
        help="",
    )
    parser.add_argument(
        "--d_env",
        default=128,
        type=int,
        help="",
    )
    parser.add_argument(
        "--loss",
        default='mse',
        type=str,
        help="",
    )
    parser.add_argument(
        "--label_len",
        default=16,
        type=int,
        help="",
    )
    parser.add_argument(
        "--contrast_prob",
        default=1,
        type=float,
        help="",
    )
    parser.add_argument(
        "--moment_mode",
        default='full_finetuning',
        type=str,
        help="",
    )
    parser.add_argument(
        "--orthogonality_penalty",
        default=0,
        type=int,
        help="",
    )
    parser.add_argument(
        "--alpha",
        default=0,
        type=float,
        help="",
    )

    parser.add_argument(
        "--EM",
        default=0,
        type=int,
        help="",
    )
    parser.add_argument(
        "--inner_loops",
        default=2,
        type=int,
        help="",
    )
    parser.add_argument(
        "--Trep_epochs",
        default=8,
        type=int,
        help="",
    )

    parser.add_argument(
        "--decay",
        default=1e-5,
        type=float,
        help="",
    )

    parser.add_argument(
        "--R0_min",
        default=1.0,
        type=float,
        help="",
    )

    parser.add_argument(
        "--R0_max",
        default=1.0,
        type=float,
        help="",
    )
    parser.add_argument(
        "--R0_alpha",
        default=1e-6,
        type=float,
        help="",
    )

    parser.add_argument(
        "--reg_alpha",
        default=1e-6,
        type=float,
        help="",
    )

    parser.add_argument(
        "--aux_weight",
        default=1e-6,
        type=float,
        help="",
    )
    parser.add_argument(
        "--ablation",
        default=0,
        type=int,
        help="",
    )
    parser.add_argument(
        "--evaluate_mode",
        default=0,
        type=int,
        help="",
    )
    # Online forecasting parameters
    parser.add_argument(
        "--initial_train_size",
        default=52,
        type=int,
        help="Initial training window size for online forecasting (in time steps)",
    )
    parser.add_argument(
        "--retrain_frequency",
        default=4,
        type=int,
        help="Retrain frequency for online forecasting (1=every step, 4=monthly)",
    )
    parser.add_argument(
        "--min_train_size",
        default=26,
        type=int,
        help="Minimum training size for online forecasting",
    )
    # Next-token-prediction parameters
    parser.add_argument(
        "--next_token_prediction",
        default=False,
        type=str2bool,
        help="Enable next-token-prediction mode (autoregressive)",
    )
    parser.add_argument(
        "--token_size",
        default=4,
        type=int,
        help="Number of time steps per token for next-token-prediction",
    )
    
    # Synthetic data generation parameters
    parser.add_argument(
        "--use_synthetic_data",
        default=False,
        type=str2bool,
        help="Use synthetic epidemic data generated on-the-fly (instead of saved data)",
    )
    parser.add_argument(
        "--synthetic_num_train",
        default=1000,
        type=int,
        help="Number of synthetic epidemics for training",
    )
    parser.add_argument(
        "--synthetic_num_valid",
        default=200,
        type=int,
        help="Number of synthetic epidemics for validation",
    )
    parser.add_argument(
        "--synthetic_num_test",
        default=200,
        type=int,
        help="Number of synthetic epidemics for testing",
    )
    parser.add_argument(
        "--synthetic_univariate",
        default=False,
        type=str2bool,
        help="Use only infected (I) compartment (True) or all compartments (False) for synthetic data",
    )
    parser.add_argument(
        "--synthetic_min_compartments",
        default=3,
        type=int,
        help="Minimum number of compartments in synthetic epidemic models",
    )
    parser.add_argument(
        "--synthetic_max_compartments",
        default=7,
        type=int,
        help="Maximum number of compartments in synthetic epidemic models",
    )
    parser.add_argument(
        "--synthetic_min_transitions",
        default=3,
        type=int,
        help="Minimum number of transition rules in synthetic epidemic models",
    )
    parser.add_argument(
        "--synthetic_max_transitions",
        default=8,
        type=int,
        help="Maximum number of transition rules in synthetic epidemic models",
    )
    parser.add_argument(
        "--synthetic_min_weeks",
        default=52,
        type=int,
        help="Minimum simulation duration in weeks for synthetic data",
    )
    parser.add_argument(
        "--synthetic_max_weeks",
        default=260,
        type=int,
        help="Maximum simulation duration in weeks for synthetic data",
    )
    parser.add_argument(
        "--synthetic_seed",
        default=None,
        type=int,
        help="Random seed for synthetic data generation (None for random)",
    )
    parser.add_argument(
        "--synthetic_streaming",
        default=True,
        type=str2bool,
        help="Generate synthetic data on-the-fly during training (True) or pre-generate all (False)",
    )
    parser.add_argument(
        "--synthetic_use_groups",
        default=True,
        type=str2bool,
        help="Use group-stratified epidemic models (age groups, regions, etc.) for more realistic simulations",
    )
    parser.add_argument(
        "--synthetic_group_ratio",
        default=0.5,
        type=float,
        help="Fraction of synthetic samples that use group-stratified models (0.0-1.0)",
    )
    parser.add_argument(
        "--use_gp_augmentation",
        default=False,
        type=lambda x: x.lower() in ['true', '1', 'yes'] if isinstance(x, str) else bool(x),
        help="Mix in GP-generated samples for learning periodic patterns (KernelSynth-inspired)",
    )
    parser.add_argument(
        "--gp_ratio",
        default=0.2,
        type=float,
        help="Fraction of training samples from GP generator (0.0-1.0), default 0.2 (20%)",
    )
    parser.add_argument(
        "--use_seasonal_forcing",
        default=True,
        type=lambda x: x.lower() in ['true', '1', 'yes'] if isinstance(x, str) else bool(x),
        help="Enable seasonal forcing in synthetic epidemic models (time-varying beta)",
    )
    parser.add_argument(
        "--seasonal_forcing_ratio",
        default=0.5,
        type=float,
        help="Fraction of synthetic samples with seasonal forcing (0.0-1.0), only used if use_seasonal_forcing=True",
    )
    parser.add_argument(
        "--num_workers",
        default=4,
        type=int,
        help="Number of DataLoader workers for parallel data generation (0=single-threaded, 4-8 recommended)",
    )
    parser.add_argument(
        "--time_resolution",
        default='weekly',
        type=str,
        help="Time resolution for synthetic data: 'weekly', 'daily', or 'mixed' for mixed training",
    )
    parser.add_argument(
        "--daily_ratio",
        default=0.5,
        type=float,
        help="When time_resolution='mixed', fraction of samples that are daily resolution (0.0-1.0)",
    )
    parser.add_argument(
        "--architecture",
        default='transformer',
        type=str,
        help="Model architecture to use: 'transformer' or 'tcn'",
    )
    
    # Compartmental modeling parameters
    parser.add_argument(
        "--use_compartmental",
        default=True,
        type=str2bool,
        help="Use compartmental CAPE model with learnable epidemic compartments",
    )
    parser.add_argument(
        "--num_embeddings",
        default=6,
        type=int,
        help="Number of embeddings per compartment (multi-view projections)",
    )
    parser.add_argument(
        "--compartments",
        default=None,
        type=str,
        help="Comma-separated list of compartments to use (e.g., 'S,I,R' or 'S,E,I,R'). Default: all",
    )
    parser.add_argument(
        "--loss_weights",
        default='S:0.5,I:1.0,E:0.5,R:0.5,H:0.5,V:0.5,Q:0.5,D:0.5,P:0.5,W:0.5,A:0.5,C:0.5',
        type=str,
        help="Comma-separated loss weights for compartments. Default: I=1.0 (observed), others=0.5",
    )
    parser.add_argument(
        "--num_heads",
        default=4,
        type=int,
        help="Number of attention heads for transformer",
    )
    parser.add_argument(
        "--ff_ratio",
        default=4,
        type=int,
        help="Feedforward expansion ratio",
    )
    parser.add_argument(
        "--patch_encoder_type",
        default='transformer',
        type=str,
        choices=['transformer', 'tcn', 'lstm', 'hybrid'],
        help="Type of patch encoder architecture: 'transformer' (GPT-style, causal), 'tcn' (causal convolution), 'lstm' (recurrent), or 'hybrid' (TCN+LSTM). All are autoregressive/decoder-only.",
    )
    parser.add_argument(
        "--max_tokens",
        default=512,
        type=int,
        help="Maximum number of tokens for positional encoding",
    )
    parser.add_argument(
        "--compute_R_t",
        default=False,
        type=str2bool,
        help="Compute R(t) time series from attention patterns during training",
    )
    parser.add_argument(
        "--R_t_loss_weight",
        default=0.01,
        type=float,
        help="Weight for R(t) supervision loss (only used if compute_R_t=True). R(t) time series come from synthetic data.",
    )
    parser.add_argument(
        "--mae_loss_weight",
        default=0.0,
        type=float,
        help="Weight for MAE loss component (combined with MSE). Default 0.0 means MSE only. Set to 1.0 for equal MSE+MAE.",
    )
    parser.add_argument(
        "--eval_interval",
        default=100,
        type=int,
        help="Evaluate every N batches (for batch-level training curves)",
    )
    parser.add_argument(
        "--eval_uncertainty",
        default=False,
        type=str2bool,
        help="Evaluate prediction uncertainty using compartment mask ensembling (for CompartmentalCAPE only)",
    )
    parser.add_argument(
        "--num_uncertainty_masks",
        default=20,
        type=int,
        help="Number of different compartment masks for uncertainty estimation",
    )
    parser.add_argument(
        "--finetune_random_masks",
        default=True,
        type=str2bool,
        help="Use random compartment masks during finetuning on real data (maintains diversity)",
    )
    parser.add_argument(
        "--finetune_mask_min_active",
        default=3,
        type=int,
        help="Minimum number of active compartments in random masks during finetuning",
    )
    parser.add_argument(
        "--finetune_mask_max_active",
        default=9,
        type=int,
        help="Maximum number of active compartments in random masks during finetuning",
    )
    parser.add_argument(
        "--tune_hyperparameters",
        default=False,
        type=lambda x: x.lower() in ['true', '1', 'yes'],
        help="Enable hyperparameter tuning for baselines (grid search over hidden_sizes, layers, lr, weight_decay)",
    )
    parser.add_argument(
        "--mc_dropout_samples",
        default=20,
        type=int,
        help="Number of MC dropout samples for uncertainty estimation in deep learning baselines",
    )
    
    args = parser.parse_args()
    
    # Parse compartments list if provided
    if args.compartments is not None:
        args.compartments = [c.strip() for c in args.compartments.split(',')]
    
    # Parse loss weights if provided
    if args.loss_weights is not None:
        weights_dict = {}
        for item in args.loss_weights.split(','):
            comp, weight = item.split(':')
            weights_dict[comp.strip()] = float(weight.strip())
        args.loss_weights_dict = weights_dict
    else:
        args.loss_weights_dict = None
    
    return args

def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True