import numpy as np
from typing import Dict, List, Tuple, Optional
import functools


# =============================================================================
# EPIDEMIC-RELEVANT KERNEL DEFINITIONS
# =============================================================================

class Kernel:
    """Base kernel class."""
    def __call__(self, X1: np.ndarray, X2: np.ndarray = None) -> np.ndarray:
        raise NotImplementedError
    
    def __add__(self, other):
        return SumKernel(self, other)
    
    def __mul__(self, other):
        return ProductKernel(self, other)


class SumKernel(Kernel):
    def __init__(self, k1: Kernel, k2: Kernel):
        self.k1, self.k2 = k1, k2
    
    def __call__(self, X1, X2=None):
        return self.k1(X1, X2) + self.k2(X1, X2)


class ProductKernel(Kernel):
    def __init__(self, k1: Kernel, k2: Kernel):
        self.k1, self.k2 = k1, k2
    
    def __call__(self, X1, X2=None):
        return self.k1(X1, X2) * self.k2(X1, X2)


class RBFKernel(Kernel):
    """Radial Basis Function (squared exponential) kernel."""
    def __init__(self, length_scale: float = 1.0, variance: float = 1.0):
        self.length_scale = length_scale
        self.variance = variance
    
    def __call__(self, X1, X2=None):
        if X2 is None:
            X2 = X1
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        dist_sq = np.sum(X1**2, axis=1, keepdims=True) + np.sum(X2**2, axis=1) - 2 * X1 @ X2.T
        return self.variance * np.exp(-0.5 * dist_sq / (self.length_scale**2))


class PeriodicKernel(Kernel):
    """Periodic (ExpSineSquared) kernel - key for seasonal patterns."""
    def __init__(self, periodicity: float, length_scale: float = 1.0, variance: float = 1.0):
        self.periodicity = periodicity
        self.length_scale = length_scale
        self.variance = variance
    
    def __call__(self, X1, X2=None):
        if X2 is None:
            X2 = X1
        X1 = np.atleast_2d(X1).flatten()
        X2 = np.atleast_2d(X2).flatten()
        diff = X1[:, None] - X2[None, :]
        sin_term = np.sin(np.pi * diff / self.periodicity)
        return self.variance * np.exp(-2 * (sin_term / self.length_scale)**2)


class LinearKernel(Kernel):
    """Linear (DotProduct) kernel for trends."""
    def __init__(self, sigma_0: float = 0.0, variance: float = 1.0):
        self.sigma_0 = sigma_0
        self.variance = variance
    
    def __call__(self, X1, X2=None):
        if X2 is None:
            X2 = X1
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        return self.variance * (self.sigma_0**2 + X1 @ X2.T)


class RationalQuadraticKernel(Kernel):
    """Rational Quadratic kernel - mixture of RBFs at different scales."""
    def __init__(self, alpha: float = 1.0, length_scale: float = 1.0, variance: float = 1.0):
        self.alpha = alpha
        self.length_scale = length_scale
        self.variance = variance
    
    def __call__(self, X1, X2=None):
        if X2 is None:
            X2 = X1
        X1 = np.atleast_2d(X1)
        X2 = np.atleast_2d(X2)
        dist_sq = np.sum(X1**2, axis=1, keepdims=True) + np.sum(X2**2, axis=1) - 2 * X1 @ X2.T
        return self.variance * (1 + dist_sq / (2 * self.alpha * self.length_scale**2))**(-self.alpha)


class WhiteNoiseKernel(Kernel):
    """White noise kernel."""
    def __init__(self, noise_level: float = 0.1):
        self.noise_level = noise_level
    
    def __call__(self, X1, X2=None):
        if X2 is None:
            n = len(np.atleast_1d(X1))
            return self.noise_level * np.eye(n)
        return np.zeros((len(np.atleast_1d(X1)), len(np.atleast_1d(X2))))


class ConstantKernel(Kernel):
    """Constant kernel."""
    def __init__(self, constant: float = 1.0):
        self.constant = constant
    
    def __call__(self, X1, X2=None):
        n1 = len(np.atleast_1d(X1))
        n2 = len(np.atleast_1d(X2)) if X2 is not None else n1
        return self.constant * np.ones((n1, n2))


# =============================================================================
# EPIDEMIC-SPECIFIC KERNEL BANK
# =============================================================================

def create_epidemic_kernel_bank(length: int = 256):
    """
    Create a bank of kernels relevant for epidemic time series.
    
    The kernels are designed to capture:
    1. Seasonal patterns (annual, biannual, multi-year cycles)
    2. Outbreak shapes (sharp peaks, broad waves)
    3. Trends (endemic growth, decline)
    4. Multi-scale variations
    """
    # Normalize periodicity by length for consistent behavior
    kernels = [
        # === SEASONAL PATTERNS (crucial for epidemic data!) ===
        # Weekly patterns (human behavior)
        PeriodicKernel(periodicity=7/length, length_scale=0.3),
        
        # Annual seasonality (flu, respiratory)
        PeriodicKernel(periodicity=52/length, length_scale=0.5),   # 52 weeks
        PeriodicKernel(periodicity=52/length, length_scale=1.0),
        
        # Biannual (some diseases have 2 peaks/year)
        PeriodicKernel(periodicity=26/length, length_scale=0.5),   # 26 weeks
        
        # Multi-year cycles (measles 2-3 years, pertussis 3-5 years)
        PeriodicKernel(periodicity=104/length, length_scale=1.0),  # 2-year
        PeriodicKernel(periodicity=156/length, length_scale=1.0),  # 3-year
        PeriodicKernel(periodicity=208/length, length_scale=1.0),  # 4-year
        
        # === OUTBREAK SHAPES (RBF at different scales) ===
        RBFKernel(length_scale=0.02),   # Very sharp peaks (single outbreak)
        RBFKernel(length_scale=0.05),   # Sharp peaks
        RBFKernel(length_scale=0.1),    # Medium peaks
        RBFKernel(length_scale=0.2),    # Broad waves
        RBFKernel(length_scale=0.5),    # Very smooth endemic
        
        # === TRENDS ===
        LinearKernel(sigma_0=0.0),      # Linear trend
        LinearKernel(sigma_0=0.5),      # Polynomial-ish trend
        LinearKernel(sigma_0=1.0),      # Stronger polynomial
        
        # === MULTI-SCALE PATTERNS ===
        RationalQuadraticKernel(alpha=0.1, length_scale=0.1),
        RationalQuadraticKernel(alpha=1.0, length_scale=0.2),
        RationalQuadraticKernel(alpha=5.0, length_scale=0.3),
        
        # === NOISE ===
        WhiteNoiseKernel(noise_level=0.01),
        WhiteNoiseKernel(noise_level=0.05),
        WhiteNoiseKernel(noise_level=0.1),
        
        # === BASE LEVEL ===
        ConstantKernel(constant=0.5),
        ConstantKernel(constant=1.0),
    ]
    return kernels


# =============================================================================
# GP SAMPLE GENERATOR
# =============================================================================

class GPSyntheticGenerator:
    """
    Generates synthetic time series using Gaussian Process priors.
    Inspired by Chronos KernelSynth but tailored for epidemic patterns.
    """
    
    def __init__(
        self,
        min_length: int = 52,
        max_length: int = 260,
        max_kernels: int = 4,
        rng_seed: Optional[int] = None
    ):
        """
        Args:
            min_length: Minimum sequence length (weeks)
            max_length: Maximum sequence length (weeks)
            max_kernels: Maximum number of kernels to combine
            rng_seed: Random seed for reproducibility
        """
        self.min_length = min_length
        self.max_length = max_length
        self.max_kernels = max_kernels
        self.rng = np.random.RandomState(rng_seed)
    
    def _random_binary_op(self, k1: Kernel, k2: Kernel) -> Kernel:
        """Combine two kernels with random + or * operator."""
        if self.rng.random() < 0.5:
            return k1 + k2
        else:
            return k1 * k2
    
    def generate_one(
        self, 
        length: Optional[int] = None,
        ensure_positive: bool = True,
        normalize: bool = True,
        kernel_type: Optional[str] = None
    ) -> Dict:
        """
        Generate one GP sample.
        
        Args:
            length: Sequence length (random if None)
            ensure_positive: Shift to ensure non-negative values
            normalize: Standardize the output
            kernel_type: Specific kernel type to use ('periodic', 'rbf', 'linear', 
                        'rational_quadratic', 'composite') or None for random
        
        Returns:
            Dict with 'values', 'time', 'kernel', 'kernel_params'
        """
        # Random length
        if length is None:
            length = self.rng.randint(self.min_length, self.max_length + 1)
        
        # Create kernel based on type
        X = np.linspace(0, 1, length)[:, None]
        kernel_name = kernel_type or self.rng.choice(['periodic', 'rbf', 'linear', 'rational_quadratic', 'composite'])
        kernel_params = {}
        
        if kernel_name == 'periodic':
            # Epidemic-relevant periodicity (seasonal)
            periodicity = self.rng.choice([0.25, 0.5, 1.0])  # Normalized periods
            length_scale = self.rng.uniform(0.1, 0.3)
            combined_kernel = PeriodicKernel(periodicity=periodicity, length_scale=length_scale)
            kernel_params = {'periodicity': periodicity, 'length_scale': length_scale}
        elif kernel_name == 'rbf':
            length_scale = self.rng.uniform(0.05, 0.2)
            combined_kernel = RBFKernel(length_scale=length_scale)
            kernel_params = {'length_scale': length_scale}
        elif kernel_name == 'linear':
            sigma_0 = self.rng.uniform(0, 0.5)
            combined_kernel = LinearKernel(sigma_0=sigma_0)
            kernel_params = {'sigma_0': sigma_0}
        elif kernel_name == 'rational_quadratic':
            alpha = self.rng.uniform(0.5, 2.0)
            length_scale = self.rng.uniform(0.1, 0.3)
            combined_kernel = RationalQuadraticKernel(alpha=alpha, length_scale=length_scale)
            kernel_params = {'alpha': alpha, 'length_scale': length_scale}
        else:  # composite
            # Random composite kernel
            kernel_bank = create_epidemic_kernel_bank(length)
            n_kernels = self.rng.randint(1, self.max_kernels + 1)
            selected_indices = self.rng.choice(len(kernel_bank), n_kernels, replace=True)
            selected_kernels = [kernel_bank[i] for i in selected_indices]
            combined_kernel = selected_kernels[0]
            for k in selected_kernels[1:]:
                combined_kernel = self._random_binary_op(combined_kernel, k)
            kernel_name = 'composite'
            kernel_params = {'n_kernels': n_kernels}
        
        # Sample from GP prior
        max_attempts = 5
        for attempt in range(max_attempts):
            try:
                cov = combined_kernel(X)
                # Add jitter for numerical stability
                cov += 1e-6 * np.eye(length)
                
                # Sample
                ts = self.rng.multivariate_normal(np.zeros(length), cov)
                break
            except np.linalg.LinAlgError:
                if attempt == max_attempts - 1:
                    # Fallback: use simple RBF
                    simple_kernel = RBFKernel(length_scale=0.1)
                    cov = simple_kernel(X) + 1e-6 * np.eye(length)
                    ts = self.rng.multivariate_normal(np.zeros(length), cov)
        
        # Post-process
        if ensure_positive:
            # Shift to make positive (epidemic counts are non-negative)
            ts = ts - ts.min() + 0.1
        
        if normalize:
            # Standardize
            ts = (ts - ts.mean()) / (ts.std() + 1e-8)
        
        # Return as dict for web API compatibility
        return {
            'values': ts,
            'time': np.arange(length),
            'kernel': kernel_name,
            'kernel_params': kernel_params,
            'metadata': {
                'source': 'gp_synthetic',
                'length': length,
                'time_resolution': 'weekly'
            }
        }
    
    def generate_one_legacy(
        self, 
        length: Optional[int] = None,
        ensure_positive: bool = True,
        normalize: bool = True
    ) -> Tuple[np.ndarray, np.ndarray, Dict]:
        """
        Generate one GP sample (legacy format for backward compatibility).
        
        Returns:
            data: Array of shape [length, 1] (univariate)
            time: Array of shape [length]
            metadata: Dict with generation info
        """
        result = self.generate_one(length, ensure_positive, normalize)
        data = result['values'].reshape(-1, 1)
        time = result['time']
        metadata = result['metadata']
        metadata['kernel'] = result['kernel']
        metadata['kernel_params'] = result['kernel_params']
        return data, time, metadata
    
    def generate_batch(
        self, 
        num_samples: int,
        **kwargs
    ) -> List[Dict]:
        """Generate a batch of GP samples."""
        samples = []
        for _ in range(num_samples):
            try:
                sample = self.generate_one(**kwargs)
                samples.append(sample)
            except Exception as e:
                # Skip failed samples
                continue
        return samples


# =============================================================================
# UTILITY FUNCTIONS
# =============================================================================

def gp_sample_to_cape_format(
    data: np.ndarray,
    time: np.ndarray,
    metadata: Dict,
    token_size: int = 4,
    all_compartments: List[str] = None
) -> Optional[Dict]:
    """
    Convert GP sample to CAPE training format (Option A: I-only mode).
    
    Args:
        data: GP sample [length, 1]
        time: Time array [length]
        metadata: Sample metadata
        token_size: Size of each token
        all_compartments: List of all compartment names
    
    Returns:
        Dict with CAPE training format, or None if too short
    """
    import torch
    
    if all_compartments is None:
        all_compartments = ['S', 'I', 'E', 'R', 'H', 'V', 'Q', 'D', 'P', 'W', 'A', 'C']
    
    length = len(data)
    if length < token_size * 2:
        return None
    
    # Tokenize
    num_tokens = length // token_size
    truncated_len = num_tokens * token_size
    
    truncated_data = data[:truncated_len]
    truncated_time = time[:truncated_len]
    
    # Reshape to tokens
    input_tokens = truncated_data.reshape(num_tokens, token_size)
    time_tokens = truncated_time.reshape(num_tokens, token_size)
    
    if num_tokens <= 1:
        return None
    
    # Create autoregressive pairs
    input_seq = input_tokens[:-1]   # [num_tokens-1, token_size]
    target_seq = input_tokens[1:]   # [num_tokens-1, token_size]
    input_time = time_tokens[:-1]
    
    # Create compartment mask: ONLY I is active (Option A)
    compartment_mask = torch.zeros(len(all_compartments), dtype=torch.bool)
    I_idx = all_compartments.index('I')
    compartment_mask[I_idx] = True
    
    # Target compartments: only I
    target_compartments = {'I': torch.FloatTensor(target_seq)}
    
    # Fake R_t (not applicable for GP samples, use constant)
    fake_R_t = torch.ones(num_tokens - 1, token_size) * 1.5
    
    sample = {
        'input': torch.FloatTensor(input_seq),
        'target_compartments': target_compartments,
        'target_R_t': fake_R_t,
        'compartment_mask': compartment_mask,
        'compartment_names': ['I'],  # Only I active
        'input_time': torch.FloatTensor(input_time) % 100,
        'R0_range': torch.FloatTensor([1.0, 2.0]),  # Placeholder
        'R0_scalar': torch.FloatTensor([1.5]),
        'is_grouped': False,
        'num_groups': 1,
        'time_resolution': metadata.get('time_resolution', 'weekly'),
        'is_gp_sample': True  # Flag for debugging
    }
    
    return sample


# =============================================================================
# DEMO
# =============================================================================

if __name__ == "__main__":
    import matplotlib.pyplot as plt
    
    print("=" * 60)
    print("GP Synthetic Generator Demo")
    print("=" * 60)
    
    generator = GPSyntheticGenerator(
        min_length=52,
        max_length=156,
        max_kernels=4,
        rng_seed=42
    )
    
    # Generate samples
    fig, axes = plt.subplots(3, 3, figsize=(15, 10))
    
    for i, ax in enumerate(axes.flat):
        sample = generator.generate_one()
        ax.plot(sample['time'], sample['values'], 'b-', alpha=0.8)
        ax.set_title(f"Sample {i+1}: {sample['kernel']}, len={sample['metadata']['length']}")
        ax.set_xlabel("Time (weeks)")
        ax.set_ylabel("Value")
    
    plt.tight_layout()
    plt.savefig("gp_samples_demo.png", dpi=150)
    print("Saved: gp_samples_demo.png")
    
    # Test conversion to CAPE format
    print("\nTesting CAPE format conversion:")
    data, time, meta = generator.generate_one_legacy(length=104)
    cape_sample = gp_sample_to_cape_format(data, time, meta, token_size=4)
    if cape_sample:
        print(f"  Input shape: {cape_sample['input'].shape}")
        print(f"  Target I shape: {cape_sample['target_compartments']['I'].shape}")
        print(f"  Compartment mask: {cape_sample['compartment_mask']}")
        print(f"  Active compartments: {cape_sample['compartment_names']}")
    
    print("\nDone!")
