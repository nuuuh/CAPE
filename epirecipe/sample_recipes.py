"""Random recipe sampler for 6-stage pipeline
===========================================

Utility script that generates multiple diverse compartmental models using the
6-stage pipeline, runs simulations, and stores results with metadata.
Supports optional group-stratified modeling.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional

from pipeline import EpiRecipePipeline, GroupConfig, generate_mixing_matrix


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate diverse epidemic models using 6-stage pipeline."
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=10,
        help="Number of models to generate and simulate."
    )
    parser.add_argument(
        "--min-compartments",
        type=int,
        default=3,
        help="Minimum number of compartments per model."
    )
    parser.add_argument(
        "--max-compartments",
        type=int,
        default=10,
        help="Maximum number of compartments per model."
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(__file__).with_name("simulations") / "recipes",
        help="Directory for output files."
    )
    parser.add_argument(
        "--prefix",
        type=str,
        default="recipe",
        help="Filename prefix for generated files."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Random seed for reproducibility."
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization plots for each model."
    )
    parser.add_argument(
        "--max-time-steps",
        type=int,
        default=250,
        help="Maximum time steps for simulation (default: 250 weeks)."
    )
    parser.add_argument(
        "--use-groups",
        action="store_true",
        help="Enable group-stratified modeling for simulations."
    )
    parser.add_argument(
        "--group-probability",
        type=float,
        default=0.5,
        help="Probability of using groups for each model (0-1, default: 0.5)."
    )
    parser.add_argument(
        "--min-groups",
        type=int,
        default=2,
        help="Minimum number of groups (default: 2)."
    )
    parser.add_argument(
        "--max-groups",
        type=int,
        default=5,
        help="Maximum number of groups (default: 5)."
    )
    return parser.parse_args()


def generate_random_groups(pipeline: EpiRecipePipeline, num_groups: int) -> List[GroupConfig]:
    """Generate random group configurations."""
    groups = []
    remaining_fraction = 1.0
    
    for i in range(num_groups):
        # Generate random population fraction
        if i == num_groups - 1:
            # Last group gets remaining fraction
            pop_frac = remaining_fraction
        else:
            # Random fraction of remaining population
            max_frac = remaining_fraction * 0.8  # Leave some for remaining groups
            pop_frac = pipeline.rng.uniform(0.1, max_frac)
            remaining_fraction -= pop_frac
        
        # Generate random characteristics
        contact_mult = pipeline.rng.uniform(0.3, 2.0)
        suscept_mult = pipeline.rng.uniform(0.7, 1.5)
        infect_mult = pipeline.rng.uniform(0.8, 1.2)
        
        groups.append(GroupConfig(
            name=f"group_{i}",
            population_fraction=pop_frac,
            contact_rate_multiplier=contact_mult,
            susceptibility_multiplier=suscept_mult,
            infectivity_multiplier=infect_mult
        ))
    
    return groups


def run_batch(args: argparse.Namespace) -> None:
    """Generate batch of models and simulations."""
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    pipeline = EpiRecipePipeline(rng_seed=args.seed)
    metadata = {"recipes": []}
    
    for i in range(args.num_samples):
        # Sample number of compartments
        num_comps = pipeline.rng.randint(args.min_compartments, args.max_compartments)
        
        # Decide whether to use group stratification
        use_groups_this_model = False
        groups = None
        mixing_type = None
        num_groups = 0
        
        if args.use_groups and pipeline.rng.random() < args.group_probability:
            use_groups_this_model = True
            num_groups = pipeline.rng.randint(args.min_groups, args.max_groups)
            groups = generate_random_groups(pipeline, num_groups)
            mixing_type = pipeline.rng.choice(['homogeneous', 'assortative', 'hierarchical', 'spatial'])
        
        # Generate model
        name = f"{args.prefix}_{i:04d}"
        if use_groups_this_model:
            name += f"_g{num_groups}"
        save_path = output_dir / name
        
        try:
            # Generate model structure first
            model = pipeline.generate_model(
                num_compartments=num_comps,
                name=name
            )
            
            # Run simulation (standard or stratified)
            if use_groups_this_model:
                df_weekly, df_daily, strat_metadata = pipeline.run_simulation_stratified(
                    model=model,
                    groups=groups,
                    mixing_type=mixing_type,
                    save_path=save_path,
                    visualize=args.visualize
                )
            else:
                df_weekly, df_daily = pipeline.run_simulation(
                    model=model,
                    save_path=save_path,
                    visualize=args.visualize
                )
                strat_metadata = None
            
            # Store metadata
            record = {
                "id": name,
                "csv_weekly": str(save_path) + "_weekly.csv",
                "csv_daily": str(save_path) + "_daily.csv",
                "compartments": model.compartments,
                "num_compartments": len(model.compartments),
                "transitions": [
                    {
                        "source": t.source,
                        "target": t.target,
                        "category": t.category,
                        "rule": t.rule_name
                    }
                    for t in model.transitions
                ],
                "num_transitions": len(model.transitions),
                "noise_level": model.noise_level,
                "params": {
                    k: float(v) for k, v in model.params.items()
                    if k in ["R0", "population", "gamma", "sigma", "beta", 
                            "simulation_days", "initial_infected"]
                },
                "stratified": use_groups_this_model
            }
            
            # Add group-specific metadata if applicable
            if use_groups_this_model and strat_metadata:
                record["group_stratified"] = {
                    "num_groups": num_groups,
                    "mixing_type": mixing_type,
                    "group_names": [g.name for g in groups],
                    "group_fractions": [g.population_fraction for g in groups],
                    "group_contact_rates": [g.contact_rate_multiplier for g in groups],
                    "group_susceptibilities": [g.susceptibility_multiplier for g in groups],
                    "group_infectivities": [g.infectivity_multiplier for g in groups],
                    "R0_dominant_eigenvalue": strat_metadata.get("R0_dominant_eigenvalue", None),
                    "metadata_file": str(save_path) + "_metadata.json"
                }
            
            metadata["recipes"].append(record)
            
            status_msg = f"[{i+1}/{args.num_samples}] Generated {name}: "
            status_msg += f"{len(model.compartments)} comps, "
            status_msg += f"{len(model.transitions)} trans, "
            status_msg += f"noise={model.noise_level:.3f}"
            if use_groups_this_model:
                status_msg += f", groups={num_groups} ({mixing_type})"
            print(status_msg)
        
        except Exception as e:
            print(f"[{i+1}/{args.num_samples}] Failed to generate {name}: {e}")
            continue
    
    # Save metadata
    metadata_path = output_dir / "metadata.json"
    with metadata_path.open("w") as f:
        json.dump(metadata, f, indent=2)
    
    # Calculate statistics
    num_recipes = len(metadata['recipes'])
    num_stratified = sum(1 for r in metadata['recipes'] if r.get('stratified', False))
    
    print(f"\nGenerated {num_recipes} models")
    print(f"Metadata saved to: {metadata_path}")
    print(f"\nSummary:")
    print(f"  Compartments range: {args.min_compartments}-{args.max_compartments}")
    print(f"  Average transitions: "
          f"{sum(r['num_transitions'] for r in metadata['recipes']) / num_recipes:.1f}")
    print(f"  Average noise level: "
          f"{sum(r['noise_level'] for r in metadata['recipes']) / num_recipes:.3f}")
    
    if args.use_groups:
        print(f"\nGroup Stratification:")
        print(f"  Models with groups: {num_stratified}/{num_recipes} ({num_stratified/num_recipes*100:.1f}%)")
        if num_stratified > 0:
            stratified_recipes = [r for r in metadata['recipes'] if r.get('stratified', False)]
            avg_groups = sum(r['group_stratified']['num_groups'] for r in stratified_recipes) / num_stratified
            print(f"  Average groups: {avg_groups:.1f}")
            
            # Count mixing types
            mixing_types = {}
            for r in stratified_recipes:
                mt = r['group_stratified']['mixing_type']
                mixing_types[mt] = mixing_types.get(mt, 0) + 1
            print(f"  Mixing types: {', '.join(f'{k}={v}' for k, v in mixing_types.items())}")


def main():
    args = parse_args()
    run_batch(args)


if __name__ == "__main__":
    main()
