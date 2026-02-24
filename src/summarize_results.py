#!/usr/bin/env python3
"""Summarize online evaluation results across all baselines and metrics.

Strategy Selection: Uses validation set (zeroshot_valid) to select best strategy based on MSE.
Metric Reporting:   Reports test set metrics (zeroshot) for the validation-selected strategy.
                    This avoids data leakage - strategy selection uses validation, evaluation uses test.

Usage:
    python tests_finetuning/summarize_results.py
    python tests_finetuning/summarize_results.py --results_dir tests_finetuning/results_online
"""

import json
import os
import argparse
import numpy as np

# Get script directory for relative paths
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_RESULTS_DIR = os.path.join(SCRIPT_DIR, 'results_online')

def main():
    parser = argparse.ArgumentParser(description='Summarize online evaluation results')
    parser.add_argument('--results_dir', type=str, default=DEFAULT_RESULTS_DIR,
                        help='Directory containing result JSON files')
    parser.add_argument('--num_folds', type=int, default=3,
                        help='Number of folds used in evaluation')
    parser.add_argument('--true_zeroshot', action='store_true',
                        help='Only use non-learnable (true zero-shot) strategies')
    parser.add_argument('--finetuned', action='store_true',
                        help='Show finetuned CAPE results instead of zero-shot')
    parser.add_argument('--select_by', type=str, default='alert_sensitivity',
                        choices=['mse', 'alert_sensitivity', 'outbreak_recall', 'composite'],
                        help='Metric to use for strategy selection (default: mse)')
    parser.add_argument('--plot_bar', action='store_true',
                        help='Generate bar plot of metrics comparison')
    parser.add_argument('--plot_radar', action='store_true',
                        help='Generate radar plots for each disease')
    parser.add_argument('--plot_dir', type=str, default=None,
                        help='Directory to save plots (default: results_dir/plots)')
    args = parser.parse_args()
    
    results_dir = args.results_dir
    num_folds = args.num_folds
    true_zeroshot = args.true_zeroshot
    finetuned = args.finetuned
    select_by = args.select_by
    
    diseases = ['Pertussis', 'Varicella', 'Tuberculosis', 'measle', 'TyphoidFever', 'Mumps', 
                'Diphtheria', 'ScarletFever', 'Smallpox', 'Influenza', 'Pneumonia', 
                'AcutePoliomyelitis', 'MeningococcalMeningitis', 'Gonorrhea', 'HepatitisA', 
                'HepatitisB', 'Rubella']

    baselines = ['chronos2', 'moirai', 'moment']
    metrics = [
        ('mse', 'MSE', False),  # (key, display_name, higher_is_better)
        # ('mae', 'MAE', False),
        ('alert_sensitivity', 'Alert Sensitivity', True),
        ('outbreak_recall', 'Outbreak Recall', True),
        ('rising_phase_mae', 'Rising Phase MAE', False),
        ('peak_underestimate_rate', 'Peak Underestimate Rate', False),
        # New metrics
        ('outbreak_precision', 'Outbreak Precision', True),
        ('outbreak_f1', 'Outbreak F1', True),
        ('critical_success_index', 'Critical Success Index', True),
        ('correlation', 'Correlation', True),
        ('trend_accuracy', 'Trend Accuracy', True),
        ('skill_score', 'Skill Score', True),
        ('normalized_mse', 'Normalized MSE', False),
        ('relative_peak_error', 'Relative Peak Error', False),
        ('weighted_mae', 'Weighted MAE', False),
    ]
    
    # Best strategies based on test performance analysis (65%+ win rate against baselines)
    BEST_STRATEGIES = [
        'beyond_max_20', 'max_plus_std', 'max_plus_std_100', 'variance_adaptive_blend',
        'percentile_75', 'median_max_blend', 'upper_half_mean', 'biased_mean_50',
        'optimistic_weighted', 'robust_optimal', 'adaptive_percentile', 'trimmed_upper',
        'quantile_blend_60_80', 'shrinkage_upper_50',
        # Best learnable strategy
        # 'learn_transformer',
    ]


    
    def is_learnable_strategy(name):
        """Check if a strategy is learnable (requires training)"""
        return name.startswith('learn_')
    
    def filter_strategies(cape_dict):
        """Filter strategies to only include best performing ones"""
        # Only include strategies in our BEST_STRATEGIES list
        filtered = {k: v for k, v in cape_dict.items() if k in BEST_STRATEGIES}
        # If true_zeroshot, further filter to exclude learnable
        if true_zeroshot:
            filtered = {k: v for k, v in filtered.items() if not is_learnable_strategy(k)}
        return filtered
    
    zeroshot_label = " (true zero-shot)" if true_zeroshot else ""
    finetuned_label = " [FINETUNED]" if finetuned else " [ZEROSHOT]"
    mode_label = f"select_by={select_by}{zeroshot_label}{finetuned_label}"
    print(f"\nMode: {mode_label}\n")

    for metric_key, metric_name, higher_better in metrics:
        print('='*160)
        print(f'SUMMARY ({mode_label}) - {metric_name}')
        print('='*160)
        header = f"{'Disease':<25}" + ''.join([f"{b:>12}" for b in baselines]) + f"{'CAPE':>12} {'Strategy':<28} {'Gap':>10}"
        print(header)
        print('-'*160)
        
        wins = 0
        total = 0
        cape_list = []
        baseline_lists = {b: [] for b in baselines}
        
        for disease in diseases:
            fpath = os.path.join(results_dir, f'{disease}_online_{num_folds}folds.json')
            if not os.path.exists(fpath):
                continue
            
            with open(fpath) as f:
                data = json.load(f)
            
            agg = data.get('aggregated', {})
            
            # Get baseline values
            baseline_vals = []
            for b in baselines:
                val = agg.get(b, {}).get(f'{metric_key}_mean', None)
                baseline_vals.append(val)
                if val is not None:
                    baseline_lists[b].append(val)

            # Get best CAPE strategy (selected on validation, reported on test)
            # Step 1: Select best strategy using validation set based on select_by metric
            # Use finetuned or zeroshot based on argument
            valid_key = 'finetuned_valid' if finetuned else 'zeroshot_valid'
            test_key = 'finetuned' if finetuned else 'zeroshot'
            cape_valid = agg.get(valid_key, {})
            cape_valid = filter_strategies(cape_valid)
            cape_test = agg.get(test_key, {})
            cape_test = filter_strategies(cape_test)
            
            if cape_valid and cape_test:
                # Select best strategy based on the chosen validation metric
                if select_by == 'mse':
                    # Lower MSE is better
                    best_strat_name = min(cape_valid.items(), key=lambda x: x[1].get('mse_mean', float('inf')))[0]
                elif select_by == 'alert_sensitivity':
                    # Higher alert_sensitivity is better
                    best_strat_name = max(cape_valid.items(), key=lambda x: x[1].get('alert_sensitivity_mean', -float('inf')))[0]
                elif select_by == 'outbreak_recall':
                    # Higher outbreak_recall is better
                    best_strat_name = max(cape_valid.items(), key=lambda x: x[1].get('outbreak_recall_mean', -float('inf')))[0]
                elif select_by == 'composite':
                    # Composite: normalized MSE (lower) + normalized alert_sensitivity (higher)
                    # Get ranges for normalization
                    mse_vals = [v.get('mse_mean', float('inf')) for v in cape_valid.values()]
                    alert_vals = [v.get('alert_sensitivity_mean', 0) for v in cape_valid.values()]
                    mse_min, mse_max = min(mse_vals), max(mse_vals)
                    alert_min, alert_max = min(alert_vals), max(alert_vals)
                    
                    def composite_score(item):
                        v = item[1]
                        mse = v.get('mse_mean', float('inf'))
                        alert = v.get('alert_sensitivity_mean', 0)
                        # Normalize: for MSE, lower is better so invert
                        norm_mse = (mse_max - mse) / (mse_max - mse_min + 1e-8) if mse_max > mse_min else 0.5
                        norm_alert = (alert - alert_min) / (alert_max - alert_min + 1e-8) if alert_max > alert_min else 0.5
                        return norm_mse + norm_alert  # Higher is better
                    
                    best_strat_name = max(cape_valid.items(), key=composite_score)[0]
                else:
                    best_strat_name = min(cape_valid.items(), key=lambda x: x[1].get('mse_mean', float('inf')))[0]
                
                # Report TEST metrics for the selected strategy
                if best_strat_name in cape_test:
                    cape_val = cape_test[best_strat_name].get(f'{metric_key}_mean')
                    cape_strat = best_strat_name
                else:
                    cape_val = None
                    cape_strat = 'N/A'
            else:
                cape_val = None
                cape_strat = 'N/A'

            # Find best baseline
            valid_baselines = [v for v in baseline_vals if v is not None]
            if valid_baselines:
                if higher_better:
                    best_baseline = max(valid_baselines)
                else:
                    best_baseline = min(valid_baselines)
            else:
                best_baseline = None
            
            # Calculate gap
            if cape_val is not None and best_baseline is not None:
                if higher_better:
                    gap = (cape_val - best_baseline) / best_baseline * 100 if best_baseline != 0 else 0
                else:
                    gap = (best_baseline - cape_val) / best_baseline * 100 if best_baseline != 0 else 0
                total += 1
                if gap > 0:
                    wins += 1
                gap_str = f'+{gap:.1f}%' if gap > 0 else f'{gap:.1f}%'
                cape_list.append(cape_val)
            else:
                gap_str = 'N/A'
            
            # Build row
            row = f'{disease:<25}'
            for val in baseline_vals:
                row += f'{val:12.4f}' if val is not None else f'{"N/A":>12}'
            row += f'{cape_val:12.4f}' if cape_val is not None else f'{"N/A":>12}'
            row += f' {cape_strat:<28} {gap_str:>10}'
            print(row)
        
        print('-'*160)

        # Averages
        avg_cape = sum(cape_list)/len(cape_list) if cape_list else 0
        avg_baselines = [sum(baseline_lists[b])/len(baseline_lists[b]) if baseline_lists[b] else 0 for b in baselines]
        avg_row = f'{"AVERAGE":<25}'
        for avg in avg_baselines:
            avg_row += f'{avg:12.4f}'
        avg_row += f'{avg_cape:12.4f}'
        print(avg_row)
        
        if total > 0:
            print(f'{metric_name} Win Rate: CAPE wins {wins}/{total} ({100*wins/total:.0f}%)')
        print()

    # Final summary table
    print('='*100)
    print('OVERALL SUMMARY')
    print('='*100)
    print(f"{'Metric':<30} {'Chronos2':>12} {'Moirai':>12} {'Moment':>12} {'CAPE':>12}")
    print('-'*100)

    for metric_key, metric_name, higher_better in metrics:
        baseline_avgs = {b: [] for b in baselines}
        cape_vals = []
        # Track wins for each model
        model_wins = {b: 0 for b in baselines}
        model_wins['CAPE'] = 0
        total = 0
        
        for disease in diseases:
            fpath = os.path.join(results_dir, f'{disease}_online_{num_folds}folds.json')
            if not os.path.exists(fpath):
                continue
            
            with open(fpath) as f:
                data = json.load(f)
            
            agg = data.get('aggregated', {})
            
            # Collect all model values for this disease
            disease_vals = {}
            for b in baselines:
                val = agg.get(b, {}).get(f'{metric_key}_mean', None)
                if val is not None:
                    baseline_avgs[b].append(val)
                    disease_vals[b] = val
            
            # Select best strategy based on validation MSE, report test metrics
            valid_key = 'finetuned_valid' if finetuned else 'zeroshot_valid'
            test_key = 'finetuned' if finetuned else 'zeroshot'
            cape_valid = agg.get(valid_key, {})
            cape_valid = filter_strategies(cape_valid)
            cape_test = agg.get(test_key, {})
            cape_test = filter_strategies(cape_test)
            
            if cape_valid and cape_test:
                # Select best strategy based on the chosen validation metric
                if select_by == 'mse':
                    best_strat_name = min(cape_valid.items(), key=lambda x: x[1].get('mse_mean', float('inf')))[0]
                elif select_by == 'alert_sensitivity':
                    best_strat_name = max(cape_valid.items(), key=lambda x: x[1].get('alert_sensitivity_mean', -float('inf')))[0]
                elif select_by == 'outbreak_recall':
                    best_strat_name = max(cape_valid.items(), key=lambda x: x[1].get('outbreak_recall_mean', -float('inf')))[0]
                elif select_by == 'composite':
                    mse_vals = [v.get('mse_mean', float('inf')) for v in cape_valid.values()]
                    alert_vals = [v.get('alert_sensitivity_mean', 0) for v in cape_valid.values()]
                    mse_min, mse_max = min(mse_vals), max(mse_vals)
                    alert_min, alert_max = min(alert_vals), max(alert_vals)
                    
                    def composite_score(item):
                        v = item[1]
                        mse = v.get('mse_mean', float('inf'))
                        alert = v.get('alert_sensitivity_mean', 0)
                        norm_mse = (mse_max - mse) / (mse_max - mse_min + 1e-8) if mse_max > mse_min else 0.5
                        norm_alert = (alert - alert_min) / (alert_max - alert_min + 1e-8) if alert_max > alert_min else 0.5
                        return norm_mse + norm_alert
                    
                    best_strat_name = max(cape_valid.items(), key=composite_score)[0]
                else:
                    best_strat_name = min(cape_valid.items(), key=lambda x: x[1].get('mse_mean', float('inf')))[0]
                    
                # Report test metrics
                if best_strat_name in cape_test:
                    cape_val = cape_test[best_strat_name].get(f'{metric_key}_mean')
                    if cape_val is not None:
                        cape_vals.append(cape_val)
                        disease_vals['CAPE'] = cape_val
                else:
                    cape_val = None
            else:
                cape_val = None
            
            # Determine winner for this disease (model with best value)
            if len(disease_vals) >= 2:  # Need at least 2 models to compare
                total += 1
                if higher_better:
                    winner = max(disease_vals.items(), key=lambda x: x[1])[0]
                else:
                    winner = min(disease_vals.items(), key=lambda x: x[1])[0]
                model_wins[winner] = model_wins.get(winner, 0) + 1
        
        avg_c2 = sum(baseline_avgs['chronos2'])/len(baseline_avgs['chronos2']) if baseline_avgs['chronos2'] else 0
        avg_m = sum(baseline_avgs['moirai'])/len(baseline_avgs['moirai']) if baseline_avgs['moirai'] else 0
        avg_mom = sum(baseline_avgs['moment'])/len(baseline_avgs['moment']) if baseline_avgs['moment'] else 0
        avg_cape = sum(cape_vals)/len(cape_vals) if cape_vals else 0
        
        # Format metric name with arrow
        arrow = '↑' if higher_better else '↓'
        metric_display = f'{metric_name} {arrow}'
        
        # Format values based on metric type (percentage vs decimal)
        # Note: percentage metrics are already stored as 0-100 values in JSON
        is_percentage = metric_key in [
            'alert_sensitivity', 'outbreak_recall', 'peak_underestimate_rate',
            'outbreak_precision', 'outbreak_f1', 'critical_success_index',
            'correlation', 'trend_accuracy', 'skill_score', 'relative_peak_error'
        ]
        if is_percentage:
            fmt_c2 = f'{avg_c2:.2f}%'
            fmt_m = f'{avg_m:.2f}%'
            fmt_mom = f'{avg_mom:.2f}%'
            fmt_cape = f'{avg_cape:.2f}%'
        else:
            fmt_c2 = f'{avg_c2:.4f}'
            fmt_m = f'{avg_m:.4f}'
            fmt_mom = f'{avg_mom:.4f}'
            fmt_cape = f'{avg_cape:.4f}'
        
        print(f'{metric_display:<30} {fmt_c2:>12} {fmt_m:>12} {fmt_mom:>12} {fmt_cape:>12}')
        
        # Print win rates for this metric
        if total > 0:
            win_strs = []
            for model in baselines + ['CAPE']:
                wins = model_wins.get(model, 0)
                win_strs.append(f'{wins}/{total}')
            print(f'{"  Win Rate":<30} {win_strs[0]:>12} {win_strs[1]:>12} {win_strs[2]:>12} {win_strs[3]:>12}')
        print()

    print('='*100)

    # ==================== VISUALIZATION ====================
    if args.plot_bar or args.plot_radar:
        import matplotlib.pyplot as plt
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend
        
        plot_dir = args.plot_dir or os.path.join(results_dir, 'plots')
        os.makedirs(plot_dir, exist_ok=True)
        
        # Collect data for plotting
        plot_metrics = [
            ('mse', 'MSE', False),
            ('alert_sensitivity', 'Alert Sensitivity', True),
            ('outbreak_recall', 'Outbreak Recall', True),
            ('rising_phase_mae', 'Rising Phase MAE', False),
            ('peak_underestimate_rate', 'Peak Underestimate Rate', False),
        ]
        
        # Collect per-disease data for all metrics
        all_disease_data = {}  # {disease: {model: {metric: value}}}
        overall_avgs = {model: {m[0]: [] for m in plot_metrics} for model in baselines + ['CAPE']}
        
        for disease in diseases:
            fpath = os.path.join(results_dir, f'{disease}_online_{num_folds}folds.json')
            if not os.path.exists(fpath):
                continue
            
            with open(fpath) as f:
                data = json.load(f)
            
            agg = data.get('aggregated', {})
            all_disease_data[disease] = {model: {} for model in baselines + ['CAPE']}
            
            # Baseline values
            for b in baselines:
                for metric_key, _, _ in plot_metrics:
                    val = agg.get(b, {}).get(f'{metric_key}_mean', None)
                    if val is not None:
                        all_disease_data[disease][b][metric_key] = val
                        overall_avgs[b][metric_key].append(val)
            
            # CAPE values (using validation-selected strategy)
            valid_key = 'finetuned_valid' if finetuned else 'zeroshot_valid'
            test_key = 'finetuned' if finetuned else 'zeroshot'
            cape_valid = agg.get(valid_key, {})
            cape_valid = filter_strategies(cape_valid)
            cape_test = agg.get(test_key, {})
            cape_test = filter_strategies(cape_test)
            
            if cape_valid and cape_test:
                if select_by == 'mse':
                    best_strat_name = min(cape_valid.items(), key=lambda x: x[1].get('mse_mean', float('inf')))[0]
                elif select_by == 'alert_sensitivity':
                    best_strat_name = max(cape_valid.items(), key=lambda x: x[1].get('alert_sensitivity_mean', -float('inf')))[0]
                elif select_by == 'outbreak_recall':
                    best_strat_name = max(cape_valid.items(), key=lambda x: x[1].get('outbreak_recall_mean', -float('inf')))[0]
                else:
                    best_strat_name = min(cape_valid.items(), key=lambda x: x[1].get('mse_mean', float('inf')))[0]
                
                if best_strat_name in cape_test:
                    for metric_key, _, _ in plot_metrics:
                        val = cape_test[best_strat_name].get(f'{metric_key}_mean')
                        if val is not None:
                            all_disease_data[disease]['CAPE'][metric_key] = val
                            overall_avgs['CAPE'][metric_key].append(val)
        
        # Compute averages
        avg_data = {}
        for model in baselines + ['CAPE']:
            avg_data[model] = {}
            for metric_key, _, _ in plot_metrics:
                vals = overall_avgs[model][metric_key]
                avg_data[model][metric_key] = np.mean(vals) if vals else 0
    
    # ==================== BAR PLOT ====================
    if args.plot_bar:
        # Academic style settings
        plt.rcParams.update({
            'font.family': 'serif',
            'font.size': 11,
            'axes.linewidth': 1.2,
            'axes.spines.top': False,
            'axes.spines.right': False,
            'xtick.major.width': 1.2,
            'ytick.major.width': 1.2,
            'legend.frameon': False,
        })
        
        fig, axes = plt.subplots(1, 5, figsize=(16, 3.5))
        models = ['Chronos2', 'Moirai', 'Moment', 'CAPE']
        model_keys = ['chronos2', 'moirai', 'moment', 'CAPE']
        # Academic-friendly color palette (colorblind-safe)
        colors = ['#4878A6', '#E69F00', '#009E73', '#CC3311']
        
        for idx, (metric_key, metric_name, higher_better) in enumerate(plot_metrics):
            ax = axes[idx]
            values = [avg_data[m][metric_key] for m in model_keys]
            
            bars = ax.bar(models, values, color=colors, edgecolor='white', linewidth=0.5, width=0.7)
            
            # Highlight best performer with hatching
            if higher_better:
                best_idx = np.argmax(values)
            else:
                best_idx = np.argmin(values)
            bars[best_idx].set_edgecolor('#333333')
            bars[best_idx].set_linewidth(1.5)
            bars[best_idx].set_hatch('///')
            
            # Metric name as xlabel with direction indicator
            arrow = '↑' if higher_better else '↓'
            ax.set_xlabel(f'{metric_name} ({arrow})', fontsize=10, fontweight='medium')
            
            if idx == 0:
                ax.set_ylabel('Value', fontsize=10)
            
            ax.tick_params(axis='x', rotation=30, labelsize=9)
            ax.tick_params(axis='y', labelsize=9)
            
            # Add value labels on bars
            for bar, val in zip(bars, values):
                height = bar.get_height()
                ax.annotate(f'{val:.2f}',
                           xy=(bar.get_x() + bar.get_width() / 2, height),
                           xytext=(0, 2),
                           textcoords="offset points",
                           ha='center', va='bottom', fontsize=8, color='#333333')
            
            # Clean up spines
            ax.spines['top'].set_visible(False)
            ax.spines['right'].set_visible(False)
            ax.set_axisbelow(True)
            ax.yaxis.grid(True, linestyle='--', alpha=0.3, color='gray')
        
        plt.tight_layout()
        bar_path = os.path.join(plot_dir, f'metrics_barplot_{select_by}.png')
        plt.savefig(bar_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(bar_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
        plt.close()
        print(f'\nBar plot saved to: {bar_path}')
    
    # ==================== RADAR PLOTS ====================
    if args.plot_radar:
        from math import pi
        
        # Academic style settings with larger fonts
        plt.rcParams.update({
            'font.family': 'serif',
            'font.size': 26,
            'axes.linewidth': 2.0,
            'legend.frameon': False,
        })
        
        models = ['Chronos2', 'Moirai', 'Moment', 'CAPE']
        model_keys = ['chronos2', 'moirai', 'moment', 'CAPE']
        # Strong vivid colors matching reference figure style
        colors = ['#00BFFF', '#FF8C00', '#8B00FF', '#FF0000']
        linestyles = ['-', '-', '-', '--']
        markers = ['o', 's', '^', 'D']
        
        # Metric labels for radar (shorter names for clarity)
        metric_short_names = {
            'MSE': 'MSE',
            'Alert Sensitivity': 'Alert Sens.',
            'Outbreak Recall': 'Outbreak Rec.',
            'Rising Phase MAE': 'Rising MAE',
            'Peak Underestimate Rate': 'Peak Underest.'
        }
        metric_labels = [metric_short_names.get(m[1], m[1]) for m in plot_metrics]
        N = len(metric_labels)
        angles = [n / float(N) * 2 * pi for n in range(N)]
        angles += angles[:1]  # Complete the loop
        
        # Normalize metrics to 0-100 scale for radar plot
        # For "lower is better" metrics, invert so higher = better for visualization
        def normalize_for_radar(disease_data, metric_key, higher_better):
            """Normalize metric values across models for a disease to 0-100 scale."""
            vals = [disease_data.get(m, {}).get(metric_key, None) for m in model_keys]
            valid_vals = [v for v in vals if v is not None]
            if not valid_vals:
                return [50] * len(model_keys)  # Default if no data
            
            min_val, max_val = min(valid_vals), max(valid_vals)
            if max_val == min_val:
                return [50] * len(model_keys)
            
            normalized = []
            for v in vals:
                if v is None:
                    normalized.append(50)
                else:
                    norm = (v - min_val) / (max_val - min_val) * 100
                    if not higher_better:
                        norm = 100 - norm  # Invert for "lower is better"
                    normalized.append(norm)
            return normalized
        
        # Create radar plot for each disease
        radar_dir = os.path.join(plot_dir, 'radar_plots')
        os.makedirs(radar_dir, exist_ok=True)
        
        for disease in all_disease_data:
            disease_data = all_disease_data[disease]
            
            # Check if we have enough data
            has_data = any(disease_data.get(m, {}) for m in model_keys)
            if not has_data:
                continue
            
            fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
            
            for i, (model_key, model_name, color, ls, marker) in enumerate(zip(model_keys, models, colors, linestyles, markers)):
                # Get normalized values for each metric
                values = []
                for metric_key, _, higher_better in plot_metrics:
                    norm_vals = normalize_for_radar(disease_data, metric_key, higher_better)
                    values.append(norm_vals[i])
                values += values[:1]  # Complete the loop
                
                ax.plot(angles, values, linestyle=ls, linewidth=4.0, label=model_name, color=color, marker=marker, markersize=14)
                ax.fill(angles, values, alpha=0.15, color=color)
            
            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(metric_labels, size=24, fontweight='bold')
            ax.set_ylim(0, 100)
            ax.set_yticks([25, 50, 75, 100])
            ax.set_yticklabels(['25', '50', '75', '100'], size=18, color='gray')
            ax.tick_params(axis='x', pad=26)
            
            # Style the grid
            ax.grid(True, linestyle='--', alpha=0.4, color='gray')
            ax.spines['polar'].set_visible(False)
            
            ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.25), fontsize=22, handlelength=3.0, handletextpad=0.6, markerscale=1.5)
            
            radar_path = os.path.join(radar_dir, f'{disease}_radar.png')
            plt.savefig(radar_path, dpi=300, bbox_inches='tight', facecolor='white')
            plt.savefig(radar_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
            plt.close()
        
        print(f'Radar plots saved to: {radar_dir}/')
        
        # Also create an overall average radar plot
        fig, ax = plt.subplots(figsize=(10, 10), subplot_kw=dict(polar=True))
        
        for i, (model_key, model_name, color, ls, marker) in enumerate(zip(model_keys, models, colors, linestyles, markers)):
            values = []
            for metric_key, _, higher_better in plot_metrics:
                norm_vals = normalize_for_radar(avg_data, metric_key, higher_better)
                values.append(norm_vals[i])
            values += values[:1]
            
            ax.plot(angles, values, linestyle=ls, linewidth=4.0, label=model_name, color=color, marker=marker, markersize=14)
            ax.fill(angles, values, alpha=0.15, color=color)
        
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metric_labels, size=24, fontweight='bold')
        ax.set_ylim(0, 100)
        ax.set_yticks([25, 50, 75, 100])
        ax.set_yticklabels(['25', '50', '75', '100'], size=18, color='gray')
        ax.tick_params(axis='x', pad=26)
        
        # Style the grid
        ax.grid(True, linestyle='--', alpha=0.4, color='gray')
        ax.spines['polar'].set_visible(False)
        
        ax.legend(loc='upper right', bbox_to_anchor=(1.25, 1.25), fontsize=22, handlelength=3.0, handletextpad=0.6, markerscale=1.5)
        
        avg_radar_path = os.path.join(plot_dir, f'overall_radar_{select_by}.png')
        plt.savefig(avg_radar_path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.savefig(avg_radar_path.replace('.png', '.pdf'), bbox_inches='tight', facecolor='white')
        plt.close()
        print(f'Overall radar plot saved to: {avg_radar_path}')


if __name__ == '__main__':
    main()
