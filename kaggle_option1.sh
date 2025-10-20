#!/bin/bash
# Script cho Kaggle - OPTION 1: Chỉ đổi hyperparameters (KHÔNG freeze)

echo "=========================================="
echo "🚀 OPTION 1: Chỉ đổi hyperparameters"
echo "=========================================="

# CD vào thư mục TAMT
cd /kaggle/working/TAMT || { echo "❌ Không tìm thấy /kaggle/working/TAMT"; exit 1; }

echo "Current directory: $(pwd)"
echo ""

# Kiểm tra files
echo "Checking files..."
[ -f "meta_train.py" ] && echo "✅ meta_train.py found" || echo "❌ meta_train.py NOT found"
[ -f "filelist/kinetics400_mini/machine_01/base.json" ] && echo "✅ Dataset found" || echo "❌ Dataset NOT found"
[ -f "/kaggle/working/112112vit-s-140epoch.pt" ] && echo "✅ Pretrained weights found" || echo "❌ Pretrained weights NOT found"
echo ""

echo "Starting training with optimized hyperparameters..."
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

echo ""
echo "=========================================="
echo "✅ HOÀN THÀNH!"
echo "=========================================="
