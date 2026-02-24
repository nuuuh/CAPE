#!/bin/bash

export CUDA_VISIBLE_DEVICES=0,1,2,3,4

# quick version
python next_token_pretraining.py \
    --exp next_token_pretrain \
    --use_synthetic_data True \
    --use_compartmental True \
    --compute_R_t True \
    --R_t_loss_weight 0.5 \
    --synthetic_num_train 100000 \
    --synthetic_num_valid 5000 \
    --synthetic_num_test 5000 \
    --synthetic_streaming True \
    --synthetic_use_groups True \
    --synthetic_group_ratio 0.5 \
    --time_resolution mixed \
    --daily_ratio 0.5 \
    --epochs 25 \
    --batch_size 64 \
    --learning_rate 1e-4 \
    --weight_decay 1e-3 \
    --hidden_size 256 \
    --layers 6 \
    --token_size 4 \
    --patch_encoder_type transformer \
    --eval_interval 100 \
    --synthetic_seed 42 \
    --device cuda

# full version
# python next_token_pretraining.py \
#     --exp next_token_pretrain_v5 \
#     --use_synthetic_data True \
#     --use_compartmental True \
#     --compute_R_t True \
#     --R_t_loss_weight 0.5 \
#     --synthetic_num_train 100000 \
#     --synthetic_num_valid 5000 \
#     --synthetic_num_test 5000 \
#     --synthetic_streaming True \
#     --synthetic_use_groups True \
#     --synthetic_group_ratio 0.5 \
#     --use_seasonal_forcing True \
#     --seasonal_forcing_ratio 0.5 \
#     --use_gp_augmentation True \
#     --gp_ratio 0.3 \
#     --time_resolution mixed \
#     --daily_ratio 0.5 \
#     --epochs 159 \
#     --batch_size 64 \
#     --learning_rate 1e-4 \
#     --weight_decay 1e-3 \
#     --mae_loss_weight 0 \
#     --hidden_size 512 \
#     --layers 4 \
#     --token_size 4 \
#     --patch_encoder_type transformer \
#     --eval_interval 100 \
#     --synthetic_seed 42 \
#     --device cuda