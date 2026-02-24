import torch
from torch.utils.data import IterableDataset, Dataset
import numpy as np
from sklearn.preprocessing import StandardScaler
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from dataset.synthetic_data_generator import SyntheticEpidemicGenerator
from dataset.gp_synthetic_generator import GPSyntheticGenerator, gp_sample_to_cape_format


class StreamingSyntheticDataset(IterableDataset):
    """
    Infinite streaming dataset that generates synthetic epidemics on-the-fly
    Each worker generates data independently during training
    """
    
    def __init__(
        self,
        num_samples_per_epoch: int,
        token_size: int = 4,
        univariate: bool = False,
        use_groups: bool = False,
        group_ratio: float = 0.3,
        min_compartments: int = 3,
        max_compartments: int = 7,
        min_transitions: int = 3,
        max_transitions: int = 8,
        min_weeks: int = 104,
        max_weeks: int = 260,
        time_resolution: str = 'weekly',
        daily_ratio: float = 0.0,
        rng_seed: int = None,
        # NEW: GP augmentation parameters
        use_gp_augmentation: bool = False,
        gp_ratio: float = 0.2,  # Fraction of samples from GP (default 20%)
    ):
        """
        Args:
            num_samples_per_epoch: Number of epidemics to generate per epoch
            token_size: Time steps per token
            univariate: Use only I (True) or all compartments (False)
            use_groups: Whether to generate group-stratified models
            group_ratio: Fraction of samples that are group-stratified
            time_resolution: 'weekly', 'daily', or 'mixed' for mixed training
            daily_ratio: When time_resolution='mixed', fraction that are daily
            use_gp_augmentation: Whether to mix in GP-generated samples
            gp_ratio: Fraction of samples that come from GP generator (0.0-1.0)
            Other args: Same as SyntheticEpidemicGenerator
        """
        super().__init__()
        self.num_samples_per_epoch = num_samples_per_epoch
        self.token_size = token_size
        self.univariate = univariate
        self.use_groups = use_groups
        self.group_ratio = group_ratio
        
        # NEW: GP augmentation settings
        self.use_gp_augmentation = use_gp_augmentation
        self.gp_ratio = gp_ratio
        
        # Generator parameters
        self.gen_params = {
            'min_compartments': min_compartments,
            'max_compartments': max_compartments,
            'min_transitions': min_transitions,
            'max_transitions': max_transitions,
            'min_weeks': min_weeks,
            'max_weeks': max_weeks,
            'time_resolution': time_resolution,
            'daily_ratio': daily_ratio,
            'rng_seed': rng_seed
        }
        
    def __iter__(self):
        """Generate samples on-the-fly using parallel batch generation"""
        # Create generator for this worker
        worker_info = torch.utils.data.get_worker_info()
        if worker_info is not None:
            # Multi-worker: each worker gets different seed
            worker_seed = self.gen_params['rng_seed']
            if worker_seed is not None:
                worker_seed = worker_seed + worker_info.id
            generator = SyntheticEpidemicGenerator(
                **{**self.gen_params, 'rng_seed': worker_seed}
            )
            # Each worker generates a portion of samples
            samples_per_worker = self.num_samples_per_epoch // worker_info.num_workers
            if worker_info.id < self.num_samples_per_epoch % worker_info.num_workers:
                samples_per_worker += 1
            
            # NEW: Create GP generator if augmentation enabled
            if self.use_gp_augmentation:
                gp_generator = GPSyntheticGenerator(
                    min_length=self.gen_params['min_weeks'],
                    max_length=self.gen_params['max_weeks'],
                    max_kernels=4,
                    rng_seed=worker_seed + 1000 if worker_seed else None
                )
        else:
            # Single worker
            generator = SyntheticEpidemicGenerator(**self.gen_params)
            samples_per_worker = self.num_samples_per_epoch
            
            # NEW: Create GP generator if augmentation enabled
            if self.use_gp_augmentation:
                gp_seed = self.gen_params['rng_seed']
                gp_generator = GPSyntheticGenerator(
                    min_length=self.gen_params['min_weeks'],
                    max_length=self.gen_params['max_weeks'],
                    max_kernels=4,
                    rng_seed=gp_seed + 1000 if gp_seed else None
                )
        
        # NEW: Local RNG for deciding GP vs epidemic samples
        import random
        local_rng = random.Random(self.gen_params.get('rng_seed'))

        # Generate samples in batches for efficiency
        # IMPORTANT: Disable parallel generation when inside DataLoader workers
        # to avoid nested multiprocessing issues
        use_parallel = (worker_info is None)  # Only parallelize in single-worker mode
        batch_size = 32  # Generate 32 epidemics at a time
        num_batches = (samples_per_worker + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            # Determine how many samples in this batch
            remaining = samples_per_worker - (batch_idx * batch_size)
            current_batch_size = min(batch_size, remaining)
            
            # NEW: Decide how many GP samples vs epidemic samples in this batch
            if self.use_gp_augmentation and self.gp_ratio > 0:
                num_gp_samples = sum(1 for _ in range(current_batch_size) if local_rng.random() < self.gp_ratio)
                num_epidemic_samples = current_batch_size - num_gp_samples
            else:
                num_gp_samples = 0
                num_epidemic_samples = current_batch_size
            
            # Generate epidemic samples
            if num_epidemic_samples > 0:
                raw_samples = generator.generate_batch(
                    num_samples=num_epidemic_samples,
                    univariate=self.univariate,
                    use_groups=self.use_groups,
                    group_ratio=self.group_ratio,
                    verbose=False,
                    parallel=use_parallel
                )
                
                # Process each generated epidemic into training samples
                for data, time, R_t, compartments, R0_or_matrix, metadata in raw_samples:
                    # Create compartment mask based on which compartments are in this model
                    # Standard compartments: ['S', 'I', 'E', 'R', 'H', 'V', 'Q', 'D', 'P', 'W', 'A', 'C']
                    all_compartments = ['S', 'I', 'E', 'R', 'H', 'V', 'Q', 'D', 'P', 'W', 'A', 'C']
                    compartment_mask = torch.zeros(len(all_compartments), dtype=torch.bool)
                    for i, comp in enumerate(all_compartments):
                        if comp in compartments:
                            compartment_mask[i] = True
                    
                    # Handle R0: scalar or matrix
                    if isinstance(R0_or_matrix, (int, float)):
                        # Scalar R0: single population model
                        R0_scalar = R0_or_matrix
                        R0_lower = R0_scalar * 0.8
                        R0_upper = R0_scalar * 1.2
                        is_grouped = False
                        num_groups = 1
                    else:
                        # R0 matrix: group-stratified model
                        # Use dominant eigenvalue as overall R0
                        R0_scalar = metadata['R0_dominant_eigenvalue']
                        R0_lower = R0_scalar * 0.8
                        R0_upper = R0_scalar * 1.2
                        is_grouped = True
                        num_groups = metadata['num_groups']
                
                # Keep all compartments together for multi-compartment supervision
                # data shape: [num_weeks, num_compartments]
                # Normalize each compartment independently
                normalized_data = np.zeros_like(data)
                for comp_idx in range(data.shape[1]):
                    comp_data = data[:, comp_idx].reshape(-1, 1)
                    scaler = StandardScaler()
                    scaler.fit(comp_data)
                    normalized_data[:, comp_idx] = scaler.transform(comp_data).reshape(-1)
                
                # Check minimum length
                if len(normalized_data) >= self.token_size * 2:
                    # Tokenize (each token contains values from all compartments)
                    seq_len = len(normalized_data)
                    num_tokens = seq_len // self.token_size
                    truncated_len = num_tokens * self.token_size
                    
                    truncated_data = normalized_data[:truncated_len]  # [truncated_len, num_compartments]
                    truncated_time = time[:truncated_len]
                    truncated_R_t = R_t[:truncated_len]
                    
                    # For input: we only use I compartment (assumed to be observed)
                    I_idx = compartments.index('I')
                    I_data = truncated_data[:, I_idx].reshape(-1, 1)  # [truncated_len, 1]
                    
                    # Reshape to tokens
                    input_tokens = I_data.reshape(num_tokens, self.token_size)  # I compartment only
                    time_tokens = truncated_time.reshape(num_tokens, self.token_size)
                    R_t_tokens = truncated_R_t.reshape(num_tokens, self.token_size)
                    
                    # Target tokens: create dict with all compartments
                    # Each compartment gets its own target
                    compartment_target_tokens = {}
                    for comp_idx, comp_name in enumerate(compartments):
                        comp_data = truncated_data[:, comp_idx].reshape(-1, 1)
                        comp_tokens = comp_data.reshape(num_tokens, self.token_size)
                        compartment_target_tokens[comp_name] = comp_tokens
                    
                    # Create autoregressive pairs
                    if num_tokens > 1:
                        input_seq = input_tokens[:-1]  # [num_tokens-1, token_size]
                        input_time = time_tokens[:-1]
                        target_R_t = R_t_tokens[1:]  # R_t for next token
                        
                        # Create targets for all compartments (next token)
                        target_compartments = {}
                        for comp_name, comp_tokens in compartment_target_tokens.items():
                            target_compartments[comp_name] = comp_tokens[1:]  # Next token
                        
                        sample = {
                            'input': torch.FloatTensor(input_seq),
                            'target_compartments': {k: torch.FloatTensor(v) for k, v in target_compartments.items()},
                            'target_R_t': torch.FloatTensor(target_R_t),
                            'compartment_mask': compartment_mask,
                            'compartment_names': compartments,  # List of active compartments in this sample
                            'input_time': torch.FloatTensor(input_time) % 100,
                            'R0_range': torch.FloatTensor([R0_lower, R0_upper]),
                            'R0_scalar': torch.FloatTensor([R0_scalar]),
                            'is_grouped': is_grouped,
                            'num_groups': num_groups,
                            'time_resolution': metadata.get('time_resolution', 'weekly') if metadata else 'weekly'
                        }
                        
                        # Add R0 matrix if group-stratified
                        if is_grouped and metadata is not None:
                            sample['R0_matrix'] = torch.FloatTensor(metadata['R0_matrix'])
                        yield sample
            
            # NEW: Generate GP samples for this batch (I-only mode)
            if num_gp_samples > 0 and self.use_gp_augmentation:
                all_compartments = ['S', 'I', 'E', 'R', 'H', 'V', 'Q', 'D', 'P', 'W', 'A', 'C']
                for _ in range(num_gp_samples):
                    try:
                        gp_data, gp_time, gp_meta = gp_generator.generate_one_legacy()
                        gp_sample = gp_sample_to_cape_format(
                            gp_data, gp_time, gp_meta,
                            token_size=self.token_size,
                            all_compartments=all_compartments
                        )
                        if gp_sample is not None:
                            yield gp_sample
                    except Exception:
                        # Skip failed GP samples
                        continue

    
    def __len__(self):
        """Approximate length (actual length depends on compartments per epidemic)"""
        # Estimate: num_samples * average_compartments
        avg_compartments = (self.gen_params['min_compartments'] + self.gen_params['max_compartments']) / 2
        if self.univariate:
            avg_compartments = 1
        return int(self.num_samples_per_epoch * avg_compartments)


class FixedSyntheticDataset(Dataset):
    """
    Fixed-size synthetic dataset (pre-generated but still created on initialization)
    Better for validation/test where we want consistent samples
    """
    
    def __init__(
        self,
        num_samples: int,
        token_size: int = 4,
        univariate: bool = False,
        use_groups: bool = False,
        group_ratio: float = 0.3,
        min_compartments: int = 3,
        max_compartments: int = 7,
        min_transitions: int = 3,
        max_transitions: int = 8,
        min_weeks: int = 104,
        max_weeks: int = 260,
        time_resolution: str = 'weekly',
        daily_ratio: float = 0.0,
        rng_seed: int = None
    ):
        """
        Args:
            num_samples: Number of epidemics to generate
            Other args: Same as StreamingSyntheticDataset
        """
        super().__init__()
        self.token_size = token_size
        self.samples = []
        
        print(f"  Generating {num_samples} fixed synthetic samples...")
        
        # Create generator
        generator = SyntheticEpidemicGenerator(
            min_compartments=min_compartments,
            max_compartments=max_compartments,
            min_transitions=min_transitions,
            max_transitions=max_transitions,
            min_weeks=min_weeks,
            max_weeks=max_weeks,
            time_resolution=time_resolution,
            daily_ratio=daily_ratio,
            rng_seed=rng_seed
        )
        
        # Generate all samples using parallel batch generation
        raw_samples = generator.generate_batch(
            num_samples=num_samples,
            univariate=univariate,
            use_groups=use_groups,
            group_ratio=group_ratio,
            verbose=True,
            parallel=True
        )
        
        # Process each generated sample
        for data, time, R_t, compartments, R0_or_matrix, metadata in raw_samples:
            # Create compartment mask based on which compartments are in this model
            all_compartments = ['S', 'I', 'E', 'R', 'H', 'V', 'Q', 'D', 'P', 'W', 'A', 'C']
            compartment_mask = torch.zeros(len(all_compartments), dtype=torch.bool)
            for idx, comp in enumerate(all_compartments):
                if comp in compartments:
                    compartment_mask[idx] = True
            
            # Handle R0: scalar or matrix
            if isinstance(R0_or_matrix, (int, float)):
                R0_scalar = R0_or_matrix
                R0_lower = R0_scalar * 0.8
                R0_upper = R0_scalar * 1.2
                is_grouped = False
                num_groups = 1
            else:
                R0_scalar = metadata['R0_dominant_eigenvalue']
                R0_lower = R0_scalar * 0.8
                R0_upper = R0_scalar * 1.2
                is_grouped = True
                num_groups = metadata['num_groups']
            
            # Keep all compartments together
            # Normalize each compartment independently
            normalized_data = np.zeros_like(data)
            for comp_idx in range(data.shape[1]):
                comp_data = data[:, comp_idx].reshape(-1, 1)
                scaler = StandardScaler()
                scaler.fit(comp_data)
                normalized_data[:, comp_idx] = scaler.transform(comp_data).reshape(-1)
            
            # Check minimum length
            if len(normalized_data) >= token_size * 2:
                # Tokenize
                seq_len = len(normalized_data)
                num_tokens = seq_len // token_size
                truncated_len = num_tokens * token_size
                
                normalized_data = normalized_data[:truncated_len]
                time_data = time[:truncated_len]
                R_t_data = R_t[:truncated_len]
                
                # Create autoregressive pairs
                if num_tokens > 1:
                    # For input: only use I compartment (index 1)
                    I_idx = compartments.index('I')
                    input_data = normalized_data[:, I_idx:I_idx+1]  # Keep 2D shape
                    input_data = input_data.reshape(num_tokens, token_size)
                    
                    # For targets: create dict with all compartments
                    target_compartments = {}
                    for comp_idx, comp_name in enumerate(compartments):
                        comp_data = normalized_data[:, comp_idx:comp_idx+1]
                        comp_tokens = comp_data.reshape(num_tokens, token_size)
                        target_compartments[comp_name] = comp_tokens[1:]  # Next token
                    
                    # Reshape time and R_t
                    time_tokens = time_data.reshape(num_tokens, token_size)
                    R_t_tokens = R_t_data.reshape(num_tokens, token_size)
                    
                    sample = {
                        'input': torch.FloatTensor(input_data[:-1]),  # Only I compartment
                        'target_compartments': {k: torch.FloatTensor(v) for k, v in target_compartments.items()},
                        'target_R_t': torch.FloatTensor(R_t_tokens[1:]),
                        'compartment_mask': compartment_mask,
                        'compartment_names': compartments,
                        'input_time': torch.FloatTensor(time_tokens[:-1]) % 100,
                        'R0_range': torch.FloatTensor([R0_lower, R0_upper]),
                        'R0_scalar': torch.FloatTensor([R0_scalar]),
                        'is_grouped': is_grouped,
                        'num_groups': num_groups,
                        'time_resolution': metadata.get('time_resolution', 'weekly') if metadata else 'weekly'
                    }
                    
                    # Add R0 matrix if group-stratified
                    if is_grouped and metadata is not None:
                        sample['R0_matrix'] = torch.FloatTensor(metadata['R0_matrix'])
                    
                    self.samples.append(sample)
        
        print(f"  Generated {len(self.samples)} tokenized samples from {num_samples} epidemics")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        return self.samples[idx]


def collate_variable_length(batch):
    """Pad sequences to the same length within a batch, with a validity mask.
    
    Uses zero-padding for shorter sequences but provides a 'sequence_mask' tensor
    that indicates which positions contain real data (True) vs padding (False).
    This mask should be used during loss calculation and visualization to ignore
    padded positions.
    """
    # Find max sequence length in this batch
    max_len = max(item['input'].shape[0] for item in batch)
    batch_size = len(batch)
    token_size = batch[0]['input'].shape[1]
    
    # Pad each sequence and create validity mask
    padded_inputs = []
    padded_times = []
    padded_target_R_t = []
    R0_ranges = []
    compartment_masks = []
    sequence_masks = []  # NEW: mask for valid (non-padded) positions
    sequence_lengths = []  # NEW: actual length of each sequence
    is_gp_sample_flags = []  # NEW: track which samples are GP samples (R_t supervision should be disabled)
    
    # For compartment targets: we'll store all 12 standard compartments
    # but only fill in data for active ones (others stay as zeros)
    all_standard_compartments = ['S', 'I', 'E', 'R', 'H', 'V', 'Q', 'D', 'P', 'W', 'A', 'C']
    
    # Initialize dict to store targets for all standard compartments
    # Each will be [batch_size, max_len, token_size]
    compartment_target_dict = {comp: [] for comp in all_standard_compartments}
    
    for item in batch:
        seq_len = item['input'].shape[0]
        sequence_lengths.append(seq_len)
        
        # Create sequence mask: True for valid positions, False for padding
        seq_mask = torch.ones(max_len, dtype=torch.bool)
        if seq_len < max_len:
            seq_mask[seq_len:] = False
        sequence_masks.append(seq_mask)
        
        if seq_len < max_len:
            # Pad with zeros
            pad_len = max_len - seq_len
            input_pad = torch.zeros(pad_len, token_size)
            time_pad = torch.zeros(pad_len, token_size)
            R_t_pad = torch.zeros(pad_len, token_size)
            
            padded_input = torch.cat([item['input'], input_pad], dim=0)
            padded_time = torch.cat([item['input_time'], time_pad], dim=0)
            padded_R_t = torch.cat([item['target_R_t'], R_t_pad], dim=0)
        else:
            padded_input = item['input']
            padded_time = item['input_time']
            padded_R_t = item['target_R_t']
        
        padded_inputs.append(padded_input)
        padded_times.append(padded_time)
        padded_target_R_t.append(padded_R_t)
        R0_ranges.append(item['R0_range'])
        compartment_masks.append(item['compartment_mask'])
        is_gp_sample_flags.append(item.get('is_gp_sample', False))
        
        # For each standard compartment, add either real data or zeros
        if 'target_compartments' in item:
            for comp in all_standard_compartments:
                if comp in item['target_compartments']:
                    # This sample has this compartment - use actual data
                    comp_target = item['target_compartments'][comp]
                    if seq_len < max_len:
                        pad_len = max_len - seq_len
                        comp_pad = torch.zeros(pad_len, token_size)
                        padded_comp = torch.cat([comp_target, comp_pad], dim=0)
                    else:
                        padded_comp = comp_target
                    compartment_target_dict[comp].append(padded_comp)
                else:
                    # This sample doesn't have this compartment - use zeros
                    compartment_target_dict[comp].append(torch.zeros(max_len, token_size))
    
    result = {
        'input': torch.stack(padded_inputs),
        'target_R_t': torch.stack(padded_target_R_t),
        'compartment_mask': torch.stack(compartment_masks),
        'input_time': torch.stack(padded_times),
        'R0_range': torch.stack(R0_ranges),  # Shape: [batch_size, 2]
        'sequence_mask': torch.stack(sequence_masks),  # NEW: [batch_size, max_len] - True for valid, False for padding
        'sequence_lengths': torch.tensor(sequence_lengths, dtype=torch.long),  # NEW: [batch_size] - actual lengths
        'is_gp_sample': torch.tensor(is_gp_sample_flags, dtype=torch.bool),  # NEW: [batch_size] - True for GP samples (disable R_t)
    }
    
    # Stack all compartment targets
    # Only include compartments that have at least one non-zero sample
    result['target_compartments'] = {}
    for comp, target_list in compartment_target_dict.items():
        stacked = torch.stack(target_list)  # [batch_size, max_len, token_size]
        # Only include if at least one sample in batch has this compartment
        if stacked.abs().sum() > 0:
            result['target_compartments'][comp] = stacked
    
    return result


if __name__ == "__main__":
    # Demo
    print("="*80)
    print("Streaming Synthetic Dataset Demo")
    print("="*80)
    
    from torch.utils.data import DataLoader
    
    # Create streaming dataset
    print("\n1. Streaming Dataset (infinite data):")
    stream_dataset = StreamingSyntheticDataset(
        num_samples_per_epoch=10,
        token_size=4,
        univariate=False,
        min_compartments=3,
        max_compartments=5,
        min_weeks=52,
        max_weeks=104,
        rng_seed=42
    )
    
    stream_loader = DataLoader(
        stream_dataset,
        batch_size=4,
        collate_fn=collate_variable_length
    )
    
    print(f"   Approximate samples per epoch: {len(stream_dataset)}")
    
    # Iterate through one epoch
    for i, batch in enumerate(stream_loader):
        if i >= 3:  # Show first 3 batches
            break
        print(f"   Batch {i+1}: input shape={batch['input'].shape}, target shape={batch['target'].shape}")
    
    # Create fixed dataset
    print("\n2. Fixed Dataset (for validation/test):")
    fixed_dataset = FixedSyntheticDataset(
        num_samples=5,
        token_size=4,
        univariate=False,
        min_compartments=3,
        max_compartments=5,
        min_weeks=52,
        max_weeks=104,
        rng_seed=42
    )
    
    fixed_loader = DataLoader(
        fixed_dataset,
        batch_size=4,
        shuffle=False,
        collate_fn=collate_variable_length
    )
    
    print(f"   Total samples: {len(fixed_dataset)}")
    for i, batch in enumerate(fixed_loader):
        print(f"   Batch {i+1}: input shape={batch['input'].shape}, target shape={batch['target'].shape}")
    
    print("\n" + "="*80)
    print("Demo completed!")
    print("="*80)
