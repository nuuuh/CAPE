#!/bin/bash
# =============================================================================
# CAPE Online Evaluation - Rolling Folds
# =============================================================================
# Runs strategy_online.py across diseases with rolling evaluation folds.
# Modes: chronos2, moirai, moment, zeroshot
# =============================================================================

set -e

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cape

PRETRAIN_PATH="checkpoints/pretraining/second_stage_pretrain_v2/checkpoint.pth"
RESULTS_DIR="src/results_online"
NUM_MASKS=20
NUM_FOLDS=3
BASE_TRAIN_RATE=0.4
MODES="chronos2 moirai moment zeroshot"
USE_FIXED_MASKS=true
METRICS="mse mae outbreak_recall alert_sensitivity peak_underestimate_rate rising_phase_mae"

mkdir -p "$RESULTS_DIR"

# Disease lists
INDOMAIN_DISEASES=("Pertussis" "Varicella" "Tuberculosis" "measle" "TyphoidFever" "Mumps" "Diphtheria" "ScarletFever")
OUTDOMAIN_DISEASES=("Smallpox" "Influenza" "Pneumonia" "AcutePoliomyelitis" "MeningococcalMeningitis" "Gonorrhea" "HepatitisA" "HepatitisB" "Rubella")

ALL_DISEASES=("TyphoidFever" "measle")
DISEASES=("${ALL_DISEASES[@]}")

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --num_folds)       NUM_FOLDS=$2;       shift 2 ;;
        --base_train_rate) BASE_TRAIN_RATE=$2;  shift 2 ;;
        --modes)
            shift; MODES=""
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                MODES="$MODES $1"; shift
            done
            MODES=$(echo $MODES | xargs) ;;
        --metrics)
            shift; METRICS=""
            while [[ $# -gt 0 ]] && [[ ! "$1" =~ ^-- ]]; do
                METRICS="$METRICS $1"; shift
            done
            METRICS=$(echo $METRICS | xargs) ;;
        --indomain)      DISEASES=("${INDOMAIN_DISEASES[@]}");  shift ;;
        --outdomain)     DISEASES=("${OUTDOMAIN_DISEASES[@]}"); shift ;;
        --all_diseases)  DISEASES=("${INDOMAIN_DISEASES[@]}" "${OUTDOMAIN_DISEASES[@]}"); shift ;;
        *)
            ALL_AVAILABLE=("${INDOMAIN_DISEASES[@]}" "${OUTDOMAIN_DISEASES[@]}")
            if [[ " ${ALL_AVAILABLE[*]} " =~ " ${1} " ]]; then
                DISEASES=("$1")
            fi
            shift ;;
    esac
done

echo "============================================================================="
echo "CAPE ONLINE EVALUATION - ROLLING FOLDS"
echo "============================================================================="
echo "  Diseases:    ${DISEASES[*]}"
echo "  Folds:       $NUM_FOLDS  |  Base train rate: $BASE_TRAIN_RATE"
echo "  Modes:       $MODES"
echo "  Metrics:     $METRICS"
echo "============================================================================="

for disease in "${DISEASES[@]}"; do
    echo ""
    echo ">>> $disease"

    LOG_FILE="$RESULTS_DIR/${disease}_online_${NUM_FOLDS}folds.log"
    JSON_FILE="$RESULTS_DIR/${disease}_online_${NUM_FOLDS}folds.json"

    CMD="python src/strategy_online.py \
        --disease $disease \
        --pretrain_path $PRETRAIN_PATH \
        --num_folds $NUM_FOLDS \
        --base_train_rate $BASE_TRAIN_RATE \
        --num_masks $NUM_MASKS \
        --modes $MODES \
        --device cuda"

    [ "$USE_FIXED_MASKS" = true ] && CMD="$CMD --use_fixed_masks"
    [ -n "$METRICS" ] && CMD="$CMD --metrics $METRICS"

    eval $CMD 2>&1 | tee "$LOG_FILE"

    # Extract JSON summary
    if [ -f "$LOG_FILE" ]; then
        sed -n '/JSON_SUMMARY_START/,/JSON_SUMMARY_END/p' "$LOG_FILE" | \
            grep -v "JSON_SUMMARY" > "$JSON_FILE" 2>/dev/null || true
    fi
done

echo ""
echo "============================================================================="
echo "SUMMARY"
echo "============================================================================="

NUM_FOLDS=$NUM_FOLDS DISEASES="${DISEASES[*]}" python3 << 'SUMMARY_SCRIPT'
import json, os, glob
import numpy as np

results_dir = "src/results_online"
diseases_env = os.environ.get('DISEASES', '')
diseases = diseases_env.split() if diseases_env else []
num_folds = int(os.environ.get('NUM_FOLDS', 3))

if not diseases:
    pattern = f"{results_dir}/*_online_{num_folds}folds.json"
    diseases = [os.path.basename(f).replace(f'_online_{num_folds}folds.json', '')
                for f in glob.glob(pattern)]

header = f"{'Disease':<15} {'Chronos2':>10} {'Moirai':>10} {'Moment':>10} {'Best ZS':>10} {'Gap':>10}"
print(header)
print("-" * len(header))

wins, total = 0, 0
for disease in diseases:
    json_path = f"{results_dir}/{disease}_online_{num_folds}folds.json"
    if not os.path.exists(json_path):
        continue
    try:
        with open(json_path) as f:
            data = json.load(f)
        agg = data.get('aggregated', {})

        chronos2 = agg.get('chronos2', {}).get('mse_mean', float('nan'))
        moirai = agg.get('moirai', {}).get('mse_mean', float('nan'))
        moment = agg.get('moment', {}).get('mse_mean', float('nan'))

        # Best zeroshot ensemble
        best_ens, best_name = float('inf'), 'N/A'
        for name, vals in agg.get('zeroshot', {}).items():
            mse = vals.get('mse_mean', float('inf'))
            if mse < best_ens:
                best_ens, best_name = mse, name
        if best_ens == float('inf'):
            best_ens = float('nan')

        # Best baseline
        baselines = [v for v in [chronos2, moirai, moment] if v == v]
        best_bl = min(baselines) if baselines else float('nan')

        if best_bl == best_bl and best_ens == best_ens:
            total += 1
            if best_ens < best_bl:
                wins += 1
                gap_str = f"+{(best_bl - best_ens)/best_bl*100:.1f}%"
            else:
                gap_str = f"-{(best_ens - best_bl)/best_bl*100:.1f}%"
        else:
            gap_str = "N/A"

        fmt = lambda v: f"{v:.4f}" if v == v else "N/A"
        print(f"{disease:<15} {fmt(chronos2):>10} {fmt(moirai):>10} {fmt(moment):>10} {fmt(best_ens):>10} {gap_str:>10}")
    except Exception as e:
        print(f"{disease:<15} Error: {e}")

print("-" * len(header))
if total > 0:
    print(f"CAPE wins vs best baseline: {wins}/{total} ({100*wins/total:.0f}%)")
SUMMARY_SCRIPT

echo "============================================================================="
echo "Results saved to: $RESULTS_DIR"
echo "============================================================================="
