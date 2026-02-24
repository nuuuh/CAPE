#!/bin/bash
# =============================================================================
# Second-Stage Pretraining on Real-World Disease Data
# =============================================================================
# This script runs second-stage pretraining on the first 20% of time series
# from IN-DOMAIN diseases in the Tycho dataset, using the synthetic-pretrained 
# model as the backbone.
#
# DISEASE SPLIT (based on temporal coverage):
# -----------------------------------------
# IN-DOMAIN (8 diseases for pretraining - long time spans 50+ years):
#   Pertussis, Varicella, Tuberculosis, measle, TyphoidFever, Mumps, Diphtheria, ScarletFever
#
# OUT-OF-DOMAIN (9 diseases held out for evaluation):
#   Historical-only: Smallpox, Influenza, Pneumonia, AcutePoliomyelitis, MeningococcalMeningitis
#   Modern-only: Gonorrhea, HepatitisA, HepatitisB, Rubella
#
# Usage:
#   bash run_second_stage_pretrain.sh              # Run with default settings (in-domain only)
#   bash run_second_stage_pretrain.sh --quick      # Quick test (5 epochs)
#   bash run_second_stage_pretrain.sh --epochs 100 # Custom epochs
#   bash run_second_stage_pretrain.sh --use_all_diseases  # Use all diseases (not recommended)
#
# After pretraining, update PRETRAIN_PATH in run_improved_comparison_varied_trainrate.sh:
#   PRETRAIN_PATH="checkpoints/pretraining/second_stage_pretrain/checkpoint.pth"
# =============================================================================

set -e

cd /home/ubuntu/zw/CAPE_new

source ~/miniconda3/etc/profile.d/conda.sh
conda activate cape

# Configuration
BACKBONE_PATH="checkpoints/pretraining/next_token_pretrain_v5/checkpoint.pth"
DATA_PATH="data/tycho_US.pt"
OUTPUT_DIR="checkpoints/pretraining/second_stage_pretrain_v2"

# Training hyperparameters
PRETRAIN_RATIO=0.3   
VALID_RATIO=0.2      
EPOCHS=5
LR=1e-5
WEIGHT_DECAY=1e-3
BATCH_SIZE=256        # Reduced from 128 for memory efficiency with multi-mask training
SEED=15

# Parse command line arguments
QUICK_MODE=false
while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            QUICK_MODE=true
            EPOCHS=5
            shift
            ;;
        --epochs)
            EPOCHS=$2
            shift 2
            ;;
        --lr)
            LR=$2
            shift 2
            ;;
        --pretrain_ratio)
            PRETRAIN_RATIO=$2
            shift 2
            ;;
        --output_dir)
            OUTPUT_DIR=$2
            shift 2
            ;;
        *)
            shift
            ;;
    esac
done

echo "============================================================================="
echo "SECOND-STAGE PRETRAINING ON REAL-WORLD DATA"
echo "============================================================================="
echo "  Backbone: $BACKBONE_PATH"
echo "  Data: $DATA_PATH"
echo "  Output: $OUTPUT_DIR"
echo "  Pretrain ratio: $PRETRAIN_RATIO"
echo "  Epochs: $EPOCHS"
echo "  Learning rate: $LR"
echo "  Batch size: $BATCH_SIZE"
[ "$QUICK_MODE" = true ] && echo "  Mode: QUICK (testing)"
echo "============================================================================="

mkdir -p $OUTPUT_DIR

CMD="python tests_finetuning/second_stage_pretrain.py \
    --backbone_path $BACKBONE_PATH \
    --data_path $DATA_PATH \
    --output_dir $OUTPUT_DIR \
    --pretrain_ratio $PRETRAIN_RATIO \
    --valid_ratio $VALID_RATIO \
    --epochs $EPOCHS \
    --lr $LR \
    --weight_decay $WEIGHT_DECAY \
    --batch_size $BATCH_SIZE \
    --seed $SEED \
    --device cuda"

[ "$QUICK_MODE" = true ] && CMD="$CMD --quick"

eval $CMD 2>&1 | tee "$OUTPUT_DIR/training.log"

echo ""
echo "============================================================================="
echo "TRAINING COMPLETE"
echo "============================================================================="
echo "Checkpoint saved to: $OUTPUT_DIR/checkpoint.pth"
echo ""
echo "To use in run_improved_comparison_varied_trainrate.sh, update:"
echo '  PRETRAIN_PATH="'"$OUTPUT_DIR"'/checkpoint.pth"'
echo "============================================================================="
