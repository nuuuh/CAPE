#!/usr/bin/env python
"""
CAPE Online Evaluation with Ensemble Strategies
=================================================
Rolling-fold evaluation pipeline:
1. Data split into folds with expanding training window
2. Each fold: baselines + frozen CAPE with ensemble strategies
3. Results aggregated across folds

Modes:
  - chronos2, moirai, moment: Foundation model baselines
  - zeroshot: Frozen CAPE with non-learnable ensemble strategies
"""

import os
import sys
import json
import argparse
import warnings
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
from typing import Dict, List, Tuple

warnings.filterwarnings('ignore')

NUM_GPUS = torch.cuda.device_count() if torch.cuda.is_available() else 1
print(f"Available GPUs: {NUM_GPUS}")

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from evaluate_utils import (
    setup_seed, load_pretrained_cape, TokenDataset, load_data,
    generate_fixed_masks, get_fixed_masks,
    collect_predictions, evaluate_ensemble, preprocess_smooth_light,
)
from ensemble_strategies import get_ensemble_strategy, AVAILABLE_STRATEGIES

# Optional model imports
try:
    from model.chronos import Chronos2, CHRONOS2_AVAILABLE, Moirai, MOIRAI_AVAILABLE
except ImportError:
    CHRONOS2_AVAILABLE = False
    MOIRAI_AVAILABLE = False

try:
    from model.Moment import Moment
    MOMENT_AVAILABLE = True
except ImportError:
    MOMENT_AVAILABLE = False

try:
    from utils.epidemic_metrics import (
        compute_outbreak_recall, compute_alert_sensitivity,
        compute_peak_underestimate_rate, compute_rising_phase_mae,
    )
    EPIDEMIC_METRICS_AVAILABLE = True
except ImportError:
    EPIDEMIC_METRICS_AVAILABLE = False

AVAILABLE_METRICS = ['mse', 'mae', 'outbreak_recall', 'alert_sensitivity',
                     'peak_underestimate_rate', 'rising_phase_mae', 'all']


# =============================================================================
# METRICS
# =============================================================================

def compute_selected_metrics(preds: np.ndarray, targets: np.ndarray,
                             metrics_list: List[str],
                             last_input: np.ndarray = None) -> Dict[str, float]:
    """Compute selected metrics from predictions and targets."""
    results = {}
    preds_flat = preds.flatten()
    targets_flat = targets.flatten()

    if 'all' in metrics_list:
        metrics_list = ['mse', 'mae', 'outbreak_recall', 'alert_sensitivity',
                        'peak_underestimate_rate', 'rising_phase_mae']

    if 'mse' in metrics_list:
        results['mse'] = float(np.mean((preds_flat - targets_flat) ** 2))
    if 'mae' in metrics_list:
        results['mae'] = float(np.mean(np.abs(preds_flat - targets_flat)))

    if EPIDEMIC_METRICS_AVAILABLE:
        if 'outbreak_recall' in metrics_list:
            val = compute_outbreak_recall(preds, targets)
            if val is not None:
                results['outbreak_recall'] = val
        if 'alert_sensitivity' in metrics_list:
            val = compute_alert_sensitivity(preds, targets)
            if val is not None:
                results['alert_sensitivity'] = val
        if 'peak_underestimate_rate' in metrics_list:
            val = compute_peak_underestimate_rate(preds, targets)
            if val is not None:
                results['peak_underestimate_rate'] = val
        if 'rising_phase_mae' in metrics_list and last_input is not None:
            val = compute_rising_phase_mae(preds, targets, last_input)
            if val is not None:
                results['rising_phase_mae'] = val

    return results


# =============================================================================
# DATASET WITH SEPARATE INPUT/LABEL SOURCES
# =============================================================================

class TokenDatasetWithOriginalLabels(Dataset):
    """Uses preprocessed tokens for input but original tokens for labels."""
    def __init__(self, input_tokens: np.ndarray, label_tokens: np.ndarray,
                 num_input: int, num_output: int):
        assert len(input_tokens) == len(label_tokens)
        total = num_input + num_output
        self.samples = [(input_tokens[i:i+num_input], label_tokens[i+num_input:i+total])
                        for i in range(len(input_tokens) - total + 1)]

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        inp, out = self.samples[idx]
        return {'input': torch.tensor(inp, dtype=torch.float32),
                'label': torch.tensor(out, dtype=torch.float32)}


# =============================================================================
# DATA LOADING
# =============================================================================

def load_data_raw(data_path: str, disease: str, token_size: int = 4):
    """Load all data without splitting (for online evaluation)."""
    data = torch.load(data_path, weights_only=False)[disease]
    total = {}
    for values in data.values():
        for w, t in enumerate(values[0][1]):
            total[t.item()] = total.get(t.item(), 0) + int(values[0][0][w].item())

    values = np.array([total[k] for k in sorted(total)], dtype=np.float32)
    scaler = StandardScaler()
    values_norm = scaler.fit_transform(values.reshape(-1, 1)).flatten()
    num_tokens = len(values_norm) // token_size
    tokens = values_norm[:num_tokens * token_size].reshape(num_tokens, token_size)
    return tokens, scaler


def prepare_cape_tokens(tokens: np.ndarray):
    """Prepare tokens for CAPE with smooth_light preprocessing."""
    return preprocess_smooth_light(tokens)


def split_into_folds(tokens: np.ndarray, num_folds: int, base_train_rate: float = 0.3):
    """Split tokens into rolling evaluation folds."""
    total_tokens = len(tokens)
    base_end = int(total_tokens * base_train_rate)
    remaining = total_tokens - base_end
    fold_size = remaining // num_folds

    folds = []
    for i in range(num_folds):
        train_end = base_end + i * fold_size
        test_start = train_end
        test_end = min(test_start + fold_size, total_tokens)
        if test_end > test_start:
            folds.append((train_end, test_start, test_end))
    return folds


# =============================================================================
# BASELINE MODEL EVALUATION (per fold)
# =============================================================================

def _eval_baseline_fold(model_fn, test_tokens, num_input, num_output, token_size,
                        device, metrics_list, desc="BASELINE"):
    """Generic baseline evaluation for a single fold."""
    total = num_input + num_output
    samples = [(test_tokens[i:i+num_input].flatten(),
                test_tokens[i+num_input:i+total].flatten())
               for i in range(len(test_tokens) - total + 1)]
    if not samples:
        return None

    all_preds, all_targets, all_inputs = [], [], []
    with torch.no_grad():
        for inp, out in tqdm(samples, desc=desc):
            x = torch.tensor(inp, dtype=torch.float32).unsqueeze(0).to(device)
            pred = model_fn(x)
            all_preds.append(pred)
            all_targets.append(out.reshape(num_output, token_size))
            all_inputs.append(inp[-token_size:])

    preds = np.array(all_preds)[:, -1, :]
    targets = np.array(all_targets)[:, -1, :]
    last_inputs = np.array(all_inputs)
    return compute_selected_metrics(preds, targets, metrics_list, last_inputs)


def evaluate_chronos2_fold(test_tokens, num_input, num_output, token_size, device,
                           metrics_list):
    if not CHRONOS2_AVAILABLE:
        print("  Chronos2 not available, skipping...")
        return None
    try:
        model = Chronos2(horizon=num_output * token_size,
                         model_name="amazon/chronos-t5-small", device=device)
    except Exception as e:
        print(f"  Chronos2 failed: {e}")
        return None
    def predict(x):
        return model(x).cpu().numpy()[0].reshape(num_output, token_size)
    return _eval_baseline_fold(predict, test_tokens, num_input, num_output,
                               token_size, device, metrics_list, "CHRONOS2")


def evaluate_moirai_fold(test_tokens, num_input, num_output, token_size, device,
                         metrics_list):
    if not MOIRAI_AVAILABLE:
        print("  Moirai not available, skipping...")
        return None
    try:
        model = Moirai(horizon=num_output * token_size,
                       model_name="Salesforce/moirai-1.1-R-small", device=device)
    except Exception as e:
        print(f"  Moirai failed: {e}")
        return None
    def predict(x):
        return model(x).cpu().numpy()[0].reshape(num_output, token_size)
    return _eval_baseline_fold(predict, test_tokens, num_input, num_output,
                               token_size, device, metrics_list, "MOIRAI")


def evaluate_moment_fold(test_tokens, num_input, num_output, token_size, device,
                         metrics_list):
    if not MOMENT_AVAILABLE:
        print("  MOMENT not available, skipping...")
        return None
    try:
        model = Moment(num_output * token_size, moment_mode='zero_shot').to(device)
        model.eval()
    except Exception as e:
        print(f"  MOMENT failed: {e}")
        return None

    total = num_input + num_output
    samples = [(test_tokens[i:i+num_input].flatten(),
                test_tokens[i+num_input:i+total].flatten())
               for i in range(len(test_tokens) - total + 1)]
    if not samples:
        return None

    all_preds, all_targets, all_inputs = [], [], []
    with torch.no_grad():
        for inp, out in tqdm(samples, desc="MOMENT"):
            x = torch.tensor(inp, dtype=torch.float32).unsqueeze(0).unsqueeze(-1).to(device)
            pred = model(x).cpu().numpy().flatten()
            all_preds.append(pred[-token_size:])
            all_targets.append(out[-token_size:])
            all_inputs.append(inp[-token_size:])

    preds = np.array(all_preds)
    targets = np.array(all_targets)
    last_inputs = np.array(all_inputs)
    return compute_selected_metrics(preds, targets, metrics_list, last_inputs)


# =============================================================================
# FOLD EVALUATION
# =============================================================================

def evaluate_fold(
    fold_idx: int,
    valid_tokens: np.ndarray,
    test_tokens: np.ndarray,
    args,
    chronos_test_tokens: np.ndarray = None,
    original_valid_tokens: np.ndarray = None,
    original_test_tokens: np.ndarray = None,
) -> Dict:
    """Evaluate a single fold: baselines + zero-shot CAPE ensembles."""
    device = args.device
    num_input = args.num_input_tokens
    num_output = args.num_output_tokens
    token_size = args.token_size
    metrics_list = getattr(args, 'metrics', ['mse', 'mae'])

    if chronos_test_tokens is None:
        chronos_test_tokens = test_tokens
    label_valid = original_valid_tokens if original_valid_tokens is not None else valid_tokens
    label_test = original_test_tokens if original_test_tokens is not None else test_tokens

    min_tokens = num_input + num_output + 1
    if len(test_tokens) < min_tokens:
        print(f"  [Fold {fold_idx}] Skipping: insufficient tokens")
        return {'fold': fold_idx, 'error': 'insufficient_tokens'}

    valid_loader = DataLoader(
        TokenDatasetWithOriginalLabels(valid_tokens, label_valid, num_input, num_output),
        batch_size=32, shuffle=False)
    test_loader = DataLoader(
        TokenDatasetWithOriginalLabels(test_tokens, label_test, num_input, num_output),
        batch_size=32, shuffle=False)

    results = {
        'fold': fold_idx,
        'valid_size': len(valid_tokens),
        'test_size': len(test_tokens),
    }

    # ----- Baselines (use original tokens) -----
    baseline_modes = {
        'moment':   lambda: evaluate_moment_fold(chronos_test_tokens, num_input, num_output, token_size, device, metrics_list),
        'chronos2': lambda: evaluate_chronos2_fold(chronos_test_tokens, num_input, num_output, token_size, device, metrics_list),
        'moirai':   lambda: evaluate_moirai_fold(chronos_test_tokens, num_input, num_output, token_size, device, metrics_list),
    }
    for mode, eval_fn in baseline_modes.items():
        if mode in args.modes:
            print(f"  [Fold {fold_idx}] {mode}...")
            result = eval_fn()
            if result is not None:
                results[mode] = result
                print(f"    {mode}: {', '.join(f'{k}={v:.4f}' for k, v in result.items())}")

    # ----- Zero-shot ensembles (frozen CAPE) -----
    if 'zeroshot' in args.modes:
        print(f"  [Fold {fold_idx}] Zero-shot ensembles...")
        model = load_pretrained_cape(args.pretrain_path, device)

        valid_preds, valid_targets = collect_predictions(
            model, valid_loader, num_output, args.num_masks,
            device, use_fixed_masks=args.use_fixed_masks)
        test_preds, test_targets = collect_predictions(
            model, test_loader, num_output, args.num_masks,
            device, use_fixed_masks=args.use_fixed_masks)

        results['zeroshot'] = {}
        results['zeroshot_valid'] = {}

        for name in AVAILABLE_STRATEGIES:
            fn = get_ensemble_strategy(name)
            results['zeroshot_valid'][name] = evaluate_ensemble(fn, valid_preds, valid_targets, metrics_list=metrics_list)
            results['zeroshot'][name] = evaluate_ensemble(fn, test_preds, test_targets, metrics_list=metrics_list)

        best = min(results['zeroshot'].items(), key=lambda x: x[1]['mse'])
        print(f"    Best zero-shot: {best[0]} MSE: {best[1]['mse']:.6f}")

        del model
        torch.cuda.empty_cache()

    return results


# =============================================================================
# MAIN ONLINE EVALUATION
# =============================================================================

def run_online_evaluation(args):
    """Run online evaluation across all folds."""
    setup_seed(args.seed)

    print("=" * 80)
    print(f"CAPE ONLINE EVALUATION: {args.disease}")
    print("=" * 80)
    print(f"  Pretrain: {args.pretrain_path}")
    print(f"  Folds: {args.num_folds}, Base train rate: {args.base_train_rate}")
    print(f"  Masks: {args.num_masks}, Modes: {args.modes}")
    print(f"  Preprocessing: smooth_light")
    print("=" * 80)

    all_tokens, scaler = load_data_raw(args.data_path, args.disease, args.token_size)
    total_tokens = len(all_tokens)
    print(f"\nTotal tokens: {total_tokens}")

    folds = split_into_folds(all_tokens, args.num_folds, args.base_train_rate)
    print(f"\n{len(folds)} evaluation folds:")
    for i, (te, ts, tend) in enumerate(folds):
        print(f"  Fold {i}: train=[0,{te}], test=[{ts},{tend}]")

    cape_tokens = prepare_cape_tokens(all_tokens)

    fold_results = []
    for fold_idx, (train_end, test_start, test_end) in enumerate(folds):
        print(f"\n{'='*60}\nFOLD {fold_idx + 1}/{len(folds)}\n{'='*60}")
        print(f"  Train={train_end} tokens, Test={test_end - test_start} tokens")

        chronos_test = all_tokens[test_start:test_end]
        cape_train = cape_tokens[:train_end]
        cape_test = cape_tokens[test_start:test_end]
        original_test = all_tokens[test_start:test_end]

        valid_rate = 0.1
        valid_end = int(len(cape_train) * (1 - valid_rate))
        original_train = all_tokens[:train_end]

        result = evaluate_fold(
            fold_idx,
            cape_train[valid_end:], cape_test,
            args,
            chronos_test_tokens=chronos_test,
            original_valid_tokens=original_train[valid_end:],
            original_test_tokens=original_test)
        fold_results.append(result)

    return aggregate_and_print_results(args, fold_results, total_tokens)


# =============================================================================
# RESULTS AGGREGATION
# =============================================================================

def aggregate_and_print_results(args, all_results, total_tokens):
    """Aggregate results across folds and output JSON summary."""
    print("\n" + "=" * 80)
    print("AGGREGATED RESULTS")
    print("=" * 80)

    aggregated = {}

    def _agg_metrics(results_list, key):
        """Aggregate numeric metrics for a top-level key across folds."""
        metric_names = set()
        for r in results_list:
            if key in r and isinstance(r[key], dict):
                metric_names.update(r[key].keys())
        metric_names -= {'params', 'type', 'best_params'}
        result = {}
        for m in metric_names:
            vals = [r[key][m] for r in results_list
                    if key in r and isinstance(r[key], dict) and m in r[key]
                    and r[key][m] is not None and not (isinstance(r[key][m], float) and np.isnan(r[key][m]))]
            if vals and isinstance(vals[0], (int, float)):
                result[f'{m}_mean'] = float(np.mean(vals))
                result[f'{m}_std'] = float(np.std(vals))
        if result:
            result['n'] = len(vals)
        return result or None

    def _agg_ensemble(results_list, ensemble_key, strategy_name):
        """Aggregate metrics for a strategy within an ensemble section."""
        metric_names = set()
        for r in results_list:
            if ensemble_key in r and strategy_name in r[ensemble_key]:
                if isinstance(r[ensemble_key][strategy_name], dict):
                    metric_names.update(r[ensemble_key][strategy_name].keys())
        metric_names -= {'params', 'type'}
        result = {}
        for m in metric_names:
            vals = [r[ensemble_key][strategy_name].get(m)
                    for r in results_list
                    if ensemble_key in r and strategy_name in r[ensemble_key]]
            vals = [v for v in vals if v is not None and not (isinstance(v, float) and np.isnan(v))]
            if vals and isinstance(vals[0], (int, float)):
                result[f'{m}_mean'] = float(np.mean(vals))
                result[f'{m}_std'] = float(np.std(vals))
        if result:
            result['n'] = len(vals)
        return result or None

    # Baselines
    for key in ['chronos2', 'moirai', 'moment']:
        agg = _agg_metrics(all_results, key)
        if agg:
            aggregated[key] = agg
            metrics_str = ', '.join(f"{m.replace('_mean','')}={agg[m]:.4f}"
                                    for m in sorted(agg) if m.endswith('_mean'))
            print(f"{key}: {metrics_str} ({agg.get('n',0)} folds)")

    # Ensemble sections: zeroshot, zeroshot_valid
    for section in ['zeroshot', 'zeroshot_valid']:
        if not any(section in r for r in all_results):
            continue
        aggregated[section] = {}
        names = set()
        for r in all_results:
            if section in r:
                names.update(r[section].keys())
        if section == 'zeroshot':
            print(f"\nZero-shot ensembles:")
        for name in sorted(names):
            agg = _agg_ensemble(all_results, section, name)
            if agg:
                aggregated[section][name] = agg
                if section == 'zeroshot':
                    metrics_str = ', '.join(f"{m.replace('_mean','')}={agg[m]:.4f}"
                                            for m in sorted(agg) if m.endswith('_mean'))
                    print(f"  {name}: {metrics_str}")

    # JSON output
    summary = {
        'disease': args.disease,
        'num_folds': args.num_folds,
        'base_train_rate': args.base_train_rate,
        'total_tokens': total_tokens,
        'aggregated': aggregated,
        'per_fold': all_results,
    }
    print("\nJSON_SUMMARY_START")
    print(json.dumps(summary, indent=2, default=str))
    print("JSON_SUMMARY_END")
    return summary


# =============================================================================
# MAIN
# =============================================================================

VALID_MODES = ['chronos2', 'moirai', 'moment', 'zeroshot', 'all']

def main():
    parser = argparse.ArgumentParser(description='CAPE Online Evaluation')
    parser.add_argument('--disease', type=str, default='Mumps')
    parser.add_argument('--data_path', type=str, default='data/tycho_US.pt')
    parser.add_argument('--pretrain_path', type=str, required=True)
    parser.add_argument('--num_folds', type=int, default=3)
    parser.add_argument('--base_train_rate', type=float, default=0.4)
    parser.add_argument('--num_input_tokens', type=int, default=8)
    parser.add_argument('--num_output_tokens', type=int, default=1)
    parser.add_argument('--token_size', type=int, default=4)
    parser.add_argument('--num_masks', type=int, default=20)
    parser.add_argument('--modes', type=str, nargs='+', default=['all'], choices=VALID_MODES)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--seed', type=int, default=15)
    parser.add_argument('--use_fixed_masks', action='store_true', default=True)
    parser.add_argument('--no_fixed_masks', dest='use_fixed_masks', action='store_false')
    parser.add_argument('--metrics', type=str, nargs='+', default=['mse', 'mae'],
                        choices=AVAILABLE_METRICS)
    args = parser.parse_args()

    if 'all' in args.modes:
        args.modes = ['chronos2', 'moirai', 'moment', 'zeroshot']
    if 'all' in args.metrics:
        args.metrics = ['mse', 'mae', 'outbreak_recall', 'alert_sensitivity',
                        'peak_underestimate_rate', 'rising_phase_mae']

    run_online_evaluation(args)


if __name__ == "__main__":
    main()
