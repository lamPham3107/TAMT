#!/bin/bash
# Script chạy meta-training với hyperparameters đã optimize
# Copy toàn bộ script này vào Kaggle notebook cell

echo "=========================================="
echo "META-TRAINING VỚI HYPERPARAMETERS ĐÃ FIX"
echo "=========================================="

# CD vào thư mục TAMT trước
cd /kaggle/working/TAMT

# OPTION 1: CHỈ ĐỔI HYPERPARAMETERS (KHUYẾN NGHỊ THỬ TRƯỚC)
echo ""
echo "🚀 OPTION 1: Chỉ đổi hyperparameters (không sửa code)"
echo ""

python3 meta_train.py \
    --dataset kinetics400_mini \
    --data_path /kaggle/working/TAMT/filelist/kinetics400_mini/machine_01 \
    --model VideoMAES \
    --method meta_deepbdc \
    --lr 1e-5 \
    --weight_decay 0.005 \
    --reduce_dim 256 \
    --backbone_lr_scale 0.1 \
    --train_n_way 5 \
    --val_n_way 5 \
    --n_shot 5 \
    --n_query 10 \
    --train_n_episode 500 \
    --val_n_episode 600 \
    --warmup_epochs 5 \
    --patience 8 \
    --epoch 10 \
    --pretrain_path /kaggle/working/112112vit-s-140epoch.pt \
    --gpu 0,1 \
    --seed 1 \
    --clip_norm 1.0 \
    --ema_decay 1.0

# HOẶC OPTION 2: VỚI FREEZE BACKBONE (nếu Option 1 chưa đủ)
# Uncomment các dòng dưới để chạy Option 2:

# echo ""
# echo "🚀 OPTION 2: Với freeze backbone (cần chạy fix_freeze_backbone.py trước)"
# echo ""
# 
python3 fix_freeze_backbone.py
cp meta_train_with_freeze.py meta_train.py

python3 meta_train.py \
    --freeze_backbone_epochs 3 \
    --dataset kinetics400_mini \
    --data_path /kaggle/working/TAMT/filelist/kinetics400_mini/machine_01 \
    --model VideoMAES \
    --method meta_deepbdc \
    --lr 1e-5 \
    --weight_decay 0.005 \
    --reduce_dim 256 \
    --backbone_lr_scale 0.1 \
    --train_n_way 5 \
    --val_n_way 5 \
    --n_shot 5 \
    --n_query 10 \
    --train_n_episode 500 \
    --val_n_episode 600 \
    --warmup_epochs 5 \
    --patience 8 \
    --epoch 10 \
    --pretrain_path /kaggle/working/112112vit-s-140epoch.pt \
    --gpu 0,1 \
    --seed 1 \
    --clip_norm 1.0 \
    --ema_decay 1.0

echo ""
echo "=========================================="
echo "HOÀN THÀNH!"
echo "=========================================="
