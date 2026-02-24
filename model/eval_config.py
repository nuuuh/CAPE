"""
Configuration Dataclasses for Evaluation
=========================================

Contains:
- HyperparameterGrid: Search space for hyperparameter tuning
- ModelConfig: Configuration for a specific model instance
"""

from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class HyperparameterGrid:
    """
    Hyperparameter search space for model tuning.
    
    For CAPE (pretrained): Only tune learning_rate and weight_decay
    For baselines: Also tune hidden_sizes and num_layers
    For statistical models: Tune model-specific parameters
    """
    # Training hyperparameters (tunable for all models)
    learning_rates: List[float] = field(default_factory=lambda: [1e-3, 5e-4, 1e-4])
    weight_decays: List[float] = field(default_factory=lambda: [1e-4, 1e-3, 1e-2])
    
    # Architecture hyperparameters (for baselines only, NOT for pretrained CAPE)
    hidden_sizes: List[int] = field(default_factory=lambda: [64, 128, 256])
    num_layers: List[int] = field(default_factory=lambda: [2, 3, 4])
    
    # ARIMA parameters
    arima_orders: List[Tuple[int, int, int]] = field(default_factory=lambda: [
        (1, 0, 1), (1, 1, 1), (2, 0, 2), (2, 1, 2), (3, 0, 3)
    ])
    
    # SIR parameters
    sir_beta_range: Tuple[float, float] = (0.1, 0.5)
    sir_gamma_range: Tuple[float, float] = (0.05, 0.2)


@dataclass
class ModelConfig:
    """
    Configuration for a specific model instance.
    
    model_type values:
    - 'cape': Pretrained CAPE (architecture is FIXED, only tune lr/wd)
    - 'deep_learning': Baseline DL models (can tune architecture)
    - 'arima', 'sir', 'naive': Statistical models
    """
    name: str
    model_type: str  # 'cape', 'deep_learning', 'arima', 'sir', 'naive'
    
    # Architecture (fixed for pretrained CAPE, tunable for baselines)
    hidden_size: int = 128
    num_layers: int = 3
    dropout: float = 0.1
    
    # Training hyperparameters (tunable for all)
    learning_rate: float = 1e-4
    weight_decay: float = 1e-3
    
    # ARIMA-specific
    arima_order: Tuple[int, int, int] = (2, 0, 2)
    
    # SIR-specific
    sir_beta: float = 0.3
    sir_gamma: float = 0.1
    
    # CAPE-specific
    token_size: int = 4
    use_compartmental: bool = True
    num_uncertainty_masks: int = 20
    pretrain_path: str = None  # Path to pretrained weights
    
    # Flag indicating if this is a pretrained model (architecture is fixed)
    is_pretrained: bool = False
    
    def __post_init__(self):
        """Set is_pretrained based on model type"""
        if self.model_type == 'cape':
            self.is_pretrained = True
