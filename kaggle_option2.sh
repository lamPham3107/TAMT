#!/bin/bash
# Script cho Kaggle - OPTION 2: Với freeze backbone

echo "=========================================="
echo "🚀 OPTION 2: Với freeze backbone"
echo "=========================================="

# CD vào thư mục TAMT
cd /kaggle/working/TAMT || { echo "❌ Không tìm thấy /kaggle/working/TAMT"; exit 1; }

echo "Current directory: $(pwd)"
echo ""

# Kiểm tra files
echo "Checking files..."
[ -f "meta_train.py" ] && echo "✅ meta_train.py found" || echo "❌ meta_train.py NOT found"
[ -f "fix_freeze_backbone.py" ] && echo "✅ fix_freeze_backbone.py found" || echo "❌ fix_freeze_backbone.py NOT found"
[ -f "filelist/kinetics400_mini/machine_01/base.json" ] && echo "✅ Dataset found" || echo "❌ Dataset NOT found"
[ -f "/kaggle/working/112112vit-s-140epoch.pt" ] && echo "✅ Pretrained weights found" || echo "❌ Pretrained weights NOT found"
echo ""

# Step 1: Thêm freeze functionality
echo "Step 1: Adding freeze backbone functionality..."
python3 fix_freeze_backbone.py
if [ $? -eq 0 ]; then
    echo "✅ fix_freeze_backbone.py completed"
else
    echo "❌ fix_freeze_backbone.py failed"
    exit 1
fi
echo ""

# Step 2: Backup original và dùng version có freeze
echo "Step 2: Backing up and replacing meta_train.py..."
if [ -f "meta_train_with_freeze.py" ]; then
    cp meta_train.py meta_train_original_backup.py
    cp meta_train_with_freeze.py meta_train.py
    echo "✅ Replaced meta_train.py with freeze version"
else
    echo "❌ meta_train_with_freeze.py not found"
    exit 1
fi
echo ""

# Step 3: Train với freeze
echo "Step 3: Starting training with freeze backbone (3 epochs)..."
echo ""

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
echo "✅ HOÀN THÀNH!"
echo "=========================================="
