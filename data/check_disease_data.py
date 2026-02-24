#!/usr/bin/env python
"""
Check which diseases in tycho_US.pt have sufficient data for evaluation
"""

import torch
import numpy as np

def check_disease_data(data_path='data/tycho_US.pt', train_rate=0.7, valid_rate=0.1, min_samples=64):
    """
    Check each disease for data availability using proportional split
    
    Args:
        data_path: Path to tycho data file
        train_rate: Proportion of data for training (default: 0.7)
        valid_rate: Proportion of data for validation (default: 0.1)
        min_samples: Minimum number of samples required (default: 64, typical lookback)
    """
    test_rate = 1 - train_rate - valid_rate
    
    print("="*70)
    print("Checking disease data availability")
    print("="*70)
    print(f"Data path: {data_path}")
    print(f"Split: Train={train_rate:.0%}, Valid={valid_rate:.0%}, Test={test_rate:.0%}")
    print(f"Minimum samples required per split: {min_samples}")
    print("="*70)
    print()
    
    # Load data
    data = torch.load(data_path)
    diseases = sorted(data.keys())
    
    valid_diseases = []
    invalid_diseases = []
    
    for disease in diseases:
        print(f"\nChecking: {disease}")
        print("-" * 70)
        
        try:
            disease_data = data[disease]
            
            # Aggregate all infections across states
            total_infections = {}
            for state, values in disease_data.items():
                infections = values[0][0]
                time = values[0][1]
                for w, i in enumerate(time):
                    i = i.item()
                    if i not in total_infections.keys():
                        total_infections[i] = int(infections.numpy()[w].item())
                    else:
                        total_infections[i] += int(infections.numpy()[w].item())
            
            total_infections = dict(sorted(total_infections.items(), key=lambda x: x[0]))
            
            time = np.array([i for i in total_infections.keys()])
            infections = np.array([i for i in total_infections.values()])
            
            # Proportional split
            total_len = len(infections)
            train_idx = int(total_len * train_rate)
            valid_idx = train_idx + int(total_len * valid_rate)
            
            train_data = infections[:train_idx]
            valid_data = infections[train_idx:valid_idx]
            test_data = infections[valid_idx:]
            
            # Get time range
            first_year = int(time[0] / 100) if len(time) > 0 else 0
            last_year = int(time[-1] / 100) if len(time) > 0 else 0
            
            print(f"  Total time points: {len(time)}")
            print(f"  Time range: {first_year} - {last_year}")
            print(f"  Train samples: {len(train_data)} ({len(train_data)/total_len:.1%})")
            print(f"  Valid samples: {len(valid_data)} ({len(valid_data)/total_len:.1%})")
            print(f"  Test samples: {len(test_data)} ({len(test_data)/total_len:.1%})")
            print(f"  Total infections: {infections.sum():,.0f}")
            
            # Check if sufficient data
            reasons = []
            if len(train_data) < min_samples:
                reasons.append(f'train={len(train_data)}<{min_samples}')
            if len(valid_data) < min_samples:
                reasons.append(f'valid={len(valid_data)}<{min_samples}')
            if len(test_data) < min_samples:
                reasons.append(f'test={len(test_data)}<{min_samples}')
            
            if not reasons:
                print(f"  ✓ VALID - Sufficient data for all splits")
                valid_diseases.append(disease)
            else:
                print(f"  ✗ INVALID - Insufficient data: {', '.join(reasons)}")
                invalid_diseases.append((disease, ', '.join(reasons)))
                
        except Exception as e:
            print(f"  ✗ ERROR - {str(e)}")
            invalid_diseases.append((disease, f'error: {str(e)}'))
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    print(f"Total diseases: {len(diseases)}")
    print(f"Valid diseases: {len(valid_diseases)}")
    print(f"Invalid diseases: {len(invalid_diseases)}")
    print()
    
    if valid_diseases:
        print("Valid diseases for evaluation:")
        for disease in valid_diseases:
            print(f"  - {disease}")
    
    if invalid_diseases:
        print("\nInvalid diseases (will be skipped):")
        for disease, reason in invalid_diseases:
            print(f"  - {disease} ({reason})")
    
    print("="*70)
    
    return valid_diseases, invalid_diseases


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Check disease data availability')
    parser.add_argument('--data_path', type=str, default='data/tycho_US.pt',
                       help='Path to tycho data file')
    parser.add_argument('--train_rate', type=float, default=0.7,
                       help='Proportion of data for training (default: 0.7)')
    parser.add_argument('--valid_rate', type=float, default=0.1,
                       help='Proportion of data for validation (default: 0.1)')
    parser.add_argument('--min_samples', type=int, default=64,
                       help='Minimum number of samples required (default: 64)')
    
    args = parser.parse_args()
    
    valid_diseases, invalid_diseases = check_disease_data(
        args.data_path, args.train_rate, args.valid_rate, args.min_samples
    )
