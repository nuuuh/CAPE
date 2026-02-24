"""Synthetic epidemic data generator for next-token pretraining
Uses epirecipe pipeline to generate multivariate time series on-the-fly
Supports both non-stratified and group-stratified models
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import torch
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union

from epirecipe.pipeline import EnhancedEpiRecipePipeline, GroupConfig, generate_mixing_matrix
import random
from multiprocessing import Pool, cpu_count
from functools import lru_cache


@lru_cache(maxsize=128)
def _get_cached_mixing_matrix(num_groups: int, mixing_type: str, assortativity: float, seed: int) -> np.ndarray:
    """Cache mixing matrices to avoid recomputation"""
    rng = np.random.default_rng(seed)
    return generate_mixing_matrix(num_groups, mixing_type, assortativity, rng)


class SyntheticEpidemicGenerator:
    """
    Generates synthetic multivariate epidemic time series using epirecipe pipeline
    Each call produces a new random epidemic model with different compartments and transitions
    
    Optimizations:
    - Cached mixing matrices for group-stratified models
    - Support for parallel batch generation
    - Reuses pipeline instances
    """
    
    def __init__(
        self,
        min_compartments: int = 3,
        max_compartments: int = 7,
        min_transitions: int = 3,
        max_transitions: int = 8,
        min_weeks: int = 52,
        max_weeks: int = 260,
        population_range: Tuple[float, float] = (1e4, 1e6),
        R0_range: Tuple[float, float] = (1.0, 12.0),
        time_resolution: str = 'weekly',
        daily_ratio: float = 0.0,
        rng_seed: Optional[int] = None
    ):
        """
        Args:
            min_compartments: Minimum number of compartments (includes S, I)
            max_compartments: Maximum number of compartments
            min_transitions: Minimum number of transition rules
            max_transitions: Maximum number of transition rules
            min_weeks: Minimum simulation duration in weeks
            max_weeks: Maximum simulation duration in weeks
            population_range: (min, max) population size
            R0_range: (min, max) basic reproduction number
            time_resolution: 'weekly', 'daily', or 'mixed' for mixed training
            daily_ratio: When time_resolution='mixed', fraction of samples that are daily (0.0-1.0)
            rng_seed: Random seed for reproducibility
        """
        self.min_compartments = min_compartments
        self.max_compartments = max_compartments
        self.min_transitions = min_transitions
        self.max_transitions = max_transitions
        self.min_weeks = min_weeks
        self.max_weeks = max_weeks
        self.population_range = population_range
        self.R0_range = R0_range
        self.time_resolution = time_resolution
        self.daily_ratio = daily_ratio
        
        # Initialize pipeline with seed (use enhanced pipeline from epirecipe)
        self.pipeline = EnhancedEpiRecipePipeline(rng_seed=rng_seed)
        self.rng = random.Random(rng_seed)
        
    def generate_one(
        self, 
        univariate: bool = False,
        use_groups: bool = False,
        num_groups: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], Union[float, np.ndarray], Optional[Dict]]:
        """
        Generate one synthetic epidemic time series
        
        Args:
            univariate: If True, return only infected (I) compartment. 
                       If False, return all compartments (multivariate)
            use_groups: If True, generate group-stratified model
            num_groups: Number of groups (2-5), randomly chosen if None
        
        Returns:
            data: Array of shape [num_timesteps, num_features] where num_features is 1 (univariate) 
                  or num_compartments (multivariate) or num_groups*num_compartments (group-stratified)
            time: Array of shape [num_timesteps] with time indices (weeks or days)
            R_t: Array of shape [num_timesteps] with dynamic reproduction number over time
            compartments: List of compartment names in the data
            R0: Basic reproduction number (scalar) or R0 matrix (if group-stratified)
            metadata: Optional dict with additional info (groups, mixing, time_resolution, etc.)
        """
        # Determine time resolution for this sample
        if self.time_resolution == 'mixed':
            use_daily = self.rng.random() < self.daily_ratio
            resolution = 'daily' if use_daily else 'weekly'
        else:
            resolution = self.time_resolution
        
        if not use_groups:
            return self._generate_single_population(univariate, resolution)
        else:
            return self._generate_group_stratified(univariate, num_groups, resolution)
    
    def _generate_single_population(self, univariate: bool, time_resolution: str = 'weekly') -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], float, Optional[Dict]]:
        """Generate non-stratified epidemic time series"""
        # Random model complexity
        num_compartments = self.rng.randint(self.min_compartments, self.max_compartments)
        
        # Random simulation parameters
        simulation_weeks = self.rng.randint(self.min_weeks, self.max_weeks)
        population = self.rng.uniform(*self.population_range)
        R0 = self.rng.uniform(*self.R0_range)
        initial_infected = self.rng.randint(1, max(2, int(population * 0.0001)))
        
        # Configure with random parameters
        config_overrides = {
            'max_time_steps': simulation_weeks,
            'population': population,
            'R0': R0,
            'initial_infected': initial_infected,
        }
        
        # Generate model using the new pipeline API
        model = self.pipeline.generate_model(
            num_compartments=num_compartments,
            overrides=config_overrides,
            max_time_steps=simulation_weeks
        )
        
        # Run simulation using new API (returns weekly and daily dataframes)
        df_weekly, df_daily = self.pipeline.run_simulation(model, visualize=False)
        
        # Select appropriate dataframe based on time resolution
        if time_resolution == 'daily':
            df = df_daily
            time_col = 'day' if 'day' in df.columns else 'week'
        else:
            df = df_weekly
            time_col = 'week' if 'week' in df.columns else 'day'
        
        time = df[time_col].values if time_col in df.columns else np.arange(len(df))
        
        # Extract R_t (dynamic reproduction number)
        R_t = df['R_t'].values if 'R_t' in df.columns else np.full(len(df), R0)
        
        if univariate:
            # Only return infected compartment
            if 'I' in df.columns:
                data = df['I'].values.reshape(-1, 1)
            else:
                # Find any column that might be infected
                I_cols = [c for c in df.columns if c == 'I' or c.startswith('I_')]
                if I_cols:
                    data = df[I_cols[0]].values.reshape(-1, 1)
                else:
                    raise ValueError("No infected compartment found in simulation output")
            compartments = ['I']
        else:
            # Return all compartments (multivariate)
            compartment_names = [c for c in model.compartments if c in df.columns]
            if not compartment_names:
                # Fallback: use all numeric columns except time and R_t
                exclude_cols = {'week', 'day', 'R_t', 'time'}
                compartment_names = [c for c in df.columns if c not in exclude_cols and df[c].dtype in [np.float64, np.float32, np.int64, np.int32]]
            data = df[compartment_names].values  # Shape: [num_timesteps, num_compartments]
            compartments = compartment_names
        
        # Return metadata with time resolution
        metadata = {'time_resolution': time_resolution}
        return data, time, R_t, compartments, R0, metadata
    
    def _generate_group_stratified(
        self, 
        univariate: bool,
        num_groups: Optional[int] = None,
        time_resolution: str = 'weekly'
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], np.ndarray, Dict]:
        """Generate group-stratified epidemic time series"""
        
        # Random number of groups
        if num_groups is None:
            num_groups = self.rng.randint(2, 5)
        
        # Create random groups with different characteristics
        groups = []
        group_names = [f"group{i}" for i in range(num_groups)]
        
        # Ensure population fractions sum to 1
        fractions = np.random.dirichlet(np.ones(num_groups) * 2)
        
        for i in range(num_groups):
            groups.append(GroupConfig(
                name=group_names[i],
                population_fraction=float(fractions[i]),
                contact_rate_multiplier=self.rng.uniform(0.5, 1.5),
                susceptibility_multiplier=self.rng.uniform(0.8, 1.2),
                infectivity_multiplier=self.rng.uniform(0.8, 1.2)
            ))
        
        # Random mixing pattern
        mixing_types = ["homogeneous", "assortative", "hierarchical", "spatial"]
        mixing_type = self.rng.choice(mixing_types)
        assortativity = self.rng.uniform(0.5, 0.9) if mixing_type == "assortative" else 0.7
        
        # Use cached mixing matrix
        matrix_seed = hash((num_groups, mixing_type, round(assortativity, 2))) % (2**31)
        mixing_matrix = _get_cached_mixing_matrix(
            num_groups, mixing_type, round(assortativity, 2), matrix_seed
        )

        # Random model complexity
        num_compartments = self.rng.randint(self.min_compartments, self.max_compartments)
        
        # Random simulation parameters
        simulation_weeks = self.rng.randint(self.min_weeks, self.max_weeks)
        population = self.rng.uniform(*self.population_range)
        R0 = self.rng.uniform(*self.R0_range)
        initial_infected = self.rng.randint(1, max(2, int(population * 0.0001)))

        # Random infectious period between 3-14 days
        infectious_period_days = self.rng.uniform(3, 14)
        gamma = 1.0 / infectious_period_days

        config_overrides = {
            'max_time_steps': simulation_weeks,
            'population': population,
            'R0': R0,
            'initial_infected': initial_infected,
            'gamma': gamma,
            'beta': R0 * gamma,  # beta = R0 * gamma for SIR models
        }

        # Generate model using the new pipeline API
        model = self.pipeline.generate_model(
            num_compartments=num_compartments,
            overrides=config_overrides,
            max_time_steps=simulation_weeks
        )
        
        base_compartments = model.compartments
        
        # Run group-stratified simulation using new API
        df_weekly, df_daily, metadata = self.pipeline.run_simulation_stratified(
            model=model,
            groups=groups,
            mixing_type=mixing_type,
            assortativity=assortativity,
            mixing_matrix=mixing_matrix,
            visualize=False
        )
        
        # Select appropriate dataframe based on time resolution
        if time_resolution == 'daily':
            df = df_daily
            strat_df = metadata.get('stratified_daily', df_daily)
            time_col = 'day' if 'day' in df.columns else 'week'
        else:
            df = df_weekly
            strat_df = metadata.get('stratified_weekly', df_weekly)
            time_col = 'week' if 'week' in df.columns else 'day'
        
        time = df[time_col].values if time_col in df.columns else np.arange(len(df))
        metadata['time_resolution'] = time_resolution
        
        # Get the aggregated compartment data from the main dataframe
        # The pipeline's run_simulation_stratified returns aggregated data in df_weekly/df_daily
        base_compartment_names = list(base_compartments)
        available_compartments = [c for c in base_compartment_names if c in df.columns]
        
        if available_compartments:
            aggregated_data = df[available_compartments].values
        else:
            # Fallback: aggregate from stratified data
            aggregated_data = np.zeros((len(time), len(base_compartment_names)))
            for i, base_comp in enumerate(base_compartment_names):
                # Find all group-stratified versions of this compartment
                group_cols = [c for c in strat_df.columns if c.startswith(base_comp + '_group')]
                if group_cols:
                    aggregated_data[:, i] = strat_df[group_cols].values.sum(axis=1)
                elif base_comp in strat_df.columns:
                    aggregated_data[:, i] = strat_df[base_comp].values
            available_compartments = base_compartment_names
        
        # Extract R_t
        if 'R_t' in df.columns:
            R_t = df['R_t'].values
        else:
            # Compute R_t based on susceptible depletion
            R0_dominant = metadata.get('R0_dominant_eigenvalue', R0)
            if 'S' in df.columns:
                S_values = df['S'].values
                initial_S = S_values[0] if S_values[0] > 0 else population
                R_t = R0_dominant * (S_values / (initial_S + 1e-10))
            else:
                R_t = np.full(len(df), R0_dominant)
        
        # Use aggregated data (same format as single population)
        if univariate:
            # Only return infected compartment (aggregated across groups)
            if 'I' in available_compartments:
                I_idx = available_compartments.index('I')
                data = aggregated_data[:, I_idx].reshape(-1, 1)
            elif 'I' in df.columns:
                data = df['I'].values.reshape(-1, 1)
            else:
                raise ValueError("No infected compartment found in simulation output")
            compartments = ['I']
        else:
            # Return all compartments (aggregated across groups)
            data = aggregated_data
            compartments = available_compartments
        
        # Use scalar R0 (dominant eigenvalue) instead of matrix for consistency
        R0_scalar = metadata.get('R0_dominant_eigenvalue', R0)
        
        # Return with metadata (now includes time_resolution)
        return data, time, R_t, compartments, R0_scalar, metadata
    
    def generate_batch(
        self, 
        num_samples: int, 
        univariate: bool = False,
        use_groups: bool = False,
        group_ratio: float = 0.3,
        verbose: bool = False,
        parallel: bool = True,
        num_workers: Optional[int] = None
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], Union[float, np.ndarray], Optional[Dict]]]:
        """
        Generate multiple synthetic epidemic time series
        
        Args:
            num_samples: Number of time series to generate
            univariate: If True, return only infected compartment(s)
            use_groups: If True, generate group-stratified models
            group_ratio: Fraction of samples that should be group-stratified (if use_groups=True)
            verbose: Print generation progress
            parallel: Use multiprocessing for faster generation
            num_workers: Number of parallel workers (defaults to cpu_count)
            
        Returns:
            List of (data, time, R_t, compartments, R0, metadata) tuples
        """
        if parallel and num_samples > 10:
            return self._generate_batch_parallel(
                num_samples, univariate, use_groups, group_ratio, verbose, num_workers
            )
        else:
            return self._generate_batch_sequential(
                num_samples, univariate, use_groups, group_ratio, verbose
            )
    
    def _generate_batch_sequential(
        self,
        num_samples: int,
        univariate: bool,
        use_groups: bool,
        group_ratio: float,
        verbose: bool
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], Union[float, np.ndarray], Optional[Dict]]]:
        """Sequential batch generation (original implementation)"""
        samples = []
        for i in range(num_samples):
            if verbose and (i + 1) % 10 == 0:
                print(f"  Generated {i+1}/{num_samples} synthetic epidemics")
            
            # Decide whether to use groups for this sample
            use_groups_this_sample = use_groups and (self.rng.random() < group_ratio)
            
            try:
                result = self.generate_one(
                    univariate=univariate, 
                    use_groups=use_groups_this_sample
                )
                samples.append(result)
            except Exception as e:
                if verbose:
                    print(f"  Warning: Failed to generate sample {i+1}: {str(e)[:100]}")
                # Retry once
                try:
                    result = self.generate_one(
                        univariate=univariate,
                        use_groups=use_groups_this_sample
                    )
                    samples.append(result)
                except Exception as e2:
                    if verbose:
                        print(f"  Retry failed: {str(e2)[:100]}")
                    continue
        
        if verbose:
            print(f"  Successfully generated {len(samples)}/{num_samples} synthetic epidemics")
            if use_groups:
                num_grouped = sum(1 for s in samples if s[5] is not None)
                print(f"  {num_grouped} are group-stratified models")
        
        return samples
    
    def _generate_batch_parallel(
        self,
        num_samples: int,
        univariate: bool,
        use_groups: bool,
        group_ratio: float,
        verbose: bool,
        num_workers: Optional[int]
    ) -> List[Tuple[np.ndarray, np.ndarray, np.ndarray, List[str], Union[float, np.ndarray], Optional[Dict]]]:
        """Parallel batch generation using multiprocessing"""
        if num_workers is None:
            num_workers = min(cpu_count(), 8)  # Cap at 8 to avoid overhead
        
        if verbose:
            print(f"  Generating {num_samples} epidemics using {num_workers} workers...")
        
        # Create argument list for parallel generation
        args_list = []
        for i in range(num_samples):
            # Each worker gets a different seed based on the pipeline's rng
            worker_seed = None if self.pipeline.rng is None else hash((id(self.pipeline.rng), i)) % (2**31)
            use_groups_sample = use_groups and (random.random() < group_ratio)
            args_list.append((self._get_init_params(), univariate, use_groups_sample, worker_seed, i))
        
        # Generate in parallel
        with Pool(num_workers) as pool:
            results = pool.map(_generate_one_worker, args_list)
        
        # Filter out failed samples
        samples = [r for r in results if r is not None]
        
        if verbose:
            print(f"  Successfully generated {len(samples)}/{num_samples} synthetic epidemics")
            if use_groups:
                num_grouped = sum(1 for s in samples if s[5] is not None)
                print(f"  {num_grouped} are group-stratified models")
        
        return samples
    
    def _get_init_params(self) -> dict:
        """Get initialization parameters for creating generator in worker process"""
        return {
            'min_compartments': self.min_compartments,
            'max_compartments': self.max_compartments,
            'min_transitions': self.min_transitions,
            'max_transitions': self.max_transitions,
            'min_weeks': self.min_weeks,
            'max_weeks': self.max_weeks,
            'population_range': self.population_range,
            'R0_range': self.R0_range,
            'time_resolution': self.time_resolution,
            'daily_ratio': self.daily_ratio
        }


def _generate_one_worker(args: Tuple) -> Optional[Tuple]:
    """Worker function for parallel generation (must be top-level for pickling)"""
    init_params, univariate, use_groups, seed, idx = args
    try:
        generator = SyntheticEpidemicGenerator(**init_params, rng_seed=seed)
        return generator.generate_one(univariate=univariate, use_groups=use_groups)
    except Exception as e:
        # Silent failure - just return None
        return None


def demo():
    """Demo showing how to use the generator"""
    print("="*80)
    print("Synthetic Epidemic Data Generator Demo")
    print("="*80)
    
    generator = SyntheticEpidemicGenerator(
        min_compartments=3,
        max_compartments=6,
        min_weeks=52,
        max_weeks=156,
        rng_seed=42
    )
    
    # Generate univariate (only infected)
    print("\n1. Generating univariate (I only) time series:")
    data_uni, time_uni, R_t_uni, comps_uni, R0_uni, meta_uni = generator.generate_one(univariate=True)
    print(f"   Shape: {data_uni.shape} (weeks x features)")
    print(f"   Compartments: {comps_uni}")
    print(f"   R0: {R0_uni:.2f}, R_t range: [{R_t_uni.min():.2f}, {R_t_uni.max():.2f}]")
    print(f"   Data range: [{data_uni.min():.2f}, {data_uni.max():.2f}]")
    
    # Generate multivariate (all compartments)
    print("\n2. Generating multivariate (all compartments) time series:")
    data_multi, time_multi, R_t_multi, comps_multi, R0_multi, meta_multi = generator.generate_one(univariate=False)
    print(f"   Shape: {data_multi.shape} (weeks x features)")
    print(f"   Compartments: {comps_multi}")
    print(f"   R0: {R0_multi:.2f}, R_t range: [{R_t_multi.min():.2f}, {R_t_multi.max():.2f}]")
    print(f"   Data range: [{data_multi.min():.2f}, {data_multi.max():.2f}]")
    
    # Generate group-stratified model (aggregated output)
    print("\n3. Generating group-stratified model (internally uses groups, but output aggregated):")
    data_group, time_group, R_t_group, comps_group, R0_group, meta_group = generator.generate_one(
        univariate=False, use_groups=True, num_groups=3
    )
    print(f"   Shape: {data_group.shape} (weeks x features)")
    print(f"   Compartments: {comps_group}")
    print(f"   R0: {R0_group:.2f}, R_t range: [{R_t_group.min():.2f}, {R_t_group.max():.2f}]")
    print(f"   Data range: [{data_group.min():.2f}, {data_group.max():.2f}]")
    print(f"   Note: Output format is identical to single population model")
    
    # Generate batch with mix of regular and group-stratified
    print("\n4. Generating batch of 5 epidemics (30% group-stratified):")
    batch = generator.generate_batch(
        num_samples=5, univariate=False, use_groups=True, 
        group_ratio=0.5, verbose=True
    )
    print(f"   Generated {len(batch)} time series")
    for i, (data, time, R_t, comps, R0, metadata) in enumerate(batch):
        if metadata is None or 'num_groups' not in metadata:
            print(f"   Series {i+1}: shape={data.shape}, R0={R0:.2f}, R_t range=[{R_t.min():.2f},{R_t.max():.2f}] (single population)")
        else:
            R0_eig = metadata.get('R0_dominant_eigenvalue', R0)
            print(f"   Series {i+1}: shape={data.shape}, R0_eig={R0_eig:.2f}, R_t range=[{R_t.min():.2f},{R_t.max():.2f}] ({metadata['num_groups']} groups)")
    
    print("\n" + "="*80)
    print("Demo completed!")
    print("="*80)


if __name__ == "__main__":
    demo()
