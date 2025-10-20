"""
Script chẩn đoán vấn đề overfitting trong meta-training
Chạy script này để kiểm tra:
1. Dataset có load được không
2. Pretrained weights có load đúng không
3. Learning rates có hợp lý không
"""

import torch
import json
import os
from collections import Counter
import sys

print("="*60)
print("KIỂM TRA DATASET")
print("="*60)

# Kiểm tra base.json
base_json_path = './filelist/kinetics400_mini/machine_01/base.json'
val_json_path = './filelist/kinetics400_mini/machine_01/val.json'

try:
    with open(base_json_path, 'r') as f:
        base_data = json.load(f)
    
    print(f"\n✓ Đọc được {base_json_path}")
    print(f"  - Số lượng videos: {len(base_data['image_names'])}")
    print(f"  - Số lượng labels: {len(base_data['image_labels'])}")
    
    # Kiểm tra class distribution
    label_counts = Counter(base_data['image_labels'])
    unique_classes = len(label_counts)
    print(f"  - Số class (base): {unique_classes}")
    print(f"  - Samples mỗi class: min={min(label_counts.values())}, max={max(label_counts.values())}, avg={len(base_data['image_labels'])/unique_classes:.1f}")
    
    # Kiểm tra 3 file .pt đầu tiên
    print(f"\n  Kiểm tra 3 file .pt đầu tiên:")
    for i, path in enumerate(base_data['image_names'][:3]):
        exists = os.path.exists(path)
        status = "✓" if exists else "✗"
        print(f"    {status} {path}")
        if exists:
            try:
                tensor = torch.load(path)
                print(f"       → shape: {tensor.shape}, dtype: {tensor.dtype}")
            except Exception as e:
                print(f"       → Lỗi load: {e}")
    
except Exception as e:
    print(f"\n✗ Lỗi đọc {base_json_path}: {e}")
    sys.exit(1)

try:
    with open(val_json_path, 'r') as f:
        val_data = json.load(f)
    
    print(f"\n✓ Đọc được {val_json_path}")
    print(f"  - Số lượng videos: {len(val_data['image_names'])}")
    val_label_counts = Counter(val_data['image_labels'])
    val_unique_classes = len(val_label_counts)
    print(f"  - Số class (val): {val_unique_classes}")
    
    # Kiểm tra overlap giữa base và val classes
    base_classes = set(base_data['image_labels'])
    val_classes = set(val_data['image_labels'])
    overlap = base_classes & val_classes
    if overlap:
        print(f"\n  ⚠ CẢNH BÁO: {len(overlap)} classes xuất hiện ở CẢ base VÀ val!")
        print(f"    → Đây có thể là vấn đề nếu đây là few-shot learning setup")
    else:
        print(f"\n  ✓ Base và val có class hoàn toàn riêng biệt (đúng cho few-shot)")
    
except Exception as e:
    print(f"\n✗ Lỗi đọc {val_json_path}: {e}")

print("\n" + "="*60)
print("KIỂM TRA PRETRAINED CHECKPOINT")
print("="*60)

pretrain_path = '/kaggle/working/112112vit-s-140epoch.pt'
# Nếu chạy local, thử path khác
if not os.path.exists(pretrain_path):
    pretrain_path = './112112vit-s-140epoch.pt'
    if not os.path.exists(pretrain_path):
        print(f"\n✗ Không tìm thấy pretrained checkpoint")
        print(f"  Đã thử: /kaggle/working/112112vit-s-140epoch.pt và ./112112vit-s-140epoch.pt")
    else:
        print(f"\n✓ Tìm thấy checkpoint tại: {pretrain_path}")
else:
    print(f"\n✓ Tìm thấy checkpoint tại: {pretrain_path}")

if os.path.exists(pretrain_path):
    try:
        checkpoint = torch.load(pretrain_path, map_location='cpu')
        
        if isinstance(checkpoint, dict):
            keys = list(checkpoint.keys())
            print(f"  - Checkpoint type: dict với {len(keys)} keys")
            print(f"  - Keys: {keys[:5]}..." if len(keys) > 5 else f"  - Keys: {keys}")
            
            # Kiểm tra state_dict
            if 'model' in checkpoint:
                state_dict = checkpoint['model']
            elif 'state_dict' in checkpoint:
                state_dict = checkpoint['state_dict']
            else:
                state_dict = checkpoint
            
            param_count = sum(p.numel() for p in state_dict.values() if isinstance(p, torch.Tensor))
            print(f"  - Tổng số parameters: {param_count:,}")
            
            # Kiểm tra một vài layer
            layer_names = list(state_dict.keys())[:10]
            print(f"\n  10 layers đầu tiên:")
            for name in layer_names:
                param = state_dict[name]
                if isinstance(param, torch.Tensor):
                    print(f"    - {name}: shape={param.shape}, mean={param.mean().item():.4f}, std={param.std().item():.4f}")
        else:
            print(f"  - Checkpoint type: {type(checkpoint)}")
            
    except Exception as e:
        print(f"\n✗ Lỗi load checkpoint: {e}")

print("\n" + "="*60)
print("PHÂN TÍCH VẤN ĐỀ VÀ ĐỀ XUẤT GIẢI PHÁP")
print("="*60)

print("""
NGUYÊN NHÂN OVERFITTING (Train 99%, Val 36%):

1. ⚠️ LEARNING RATE QUÁ CAO cho head
   - backbone_lr_scale=0.1 nghĩa là backbone LR = 5e-6
   - Nhưng HEAD LR = 5e-5 (cao gấp 10 lần backbone)
   - Head học quá nhanh → ghi nhớ training data

2. ⚠️ WEIGHT DECAY QUÁ THẤP
   - weight_decay=0.00075 rất nhỏ
   - Không đủ để regularize model

3. ⚠️ EPISODIC SAMPLING có thể không random đủ
   - 500 episodes/epoch có thể lặp lại nhiều combinations
   - Model có thể nhớ các episode cụ thể

4. ⚠️ REDUCE_DIM=512 có thể quá lớn
   - Với few-shot (5-shot), feature dim 512 có thể overkill
   - Dễ overfit với ít samples

GIẢI PHÁP ƯU TIÊN (thử theo thứ tự):

A. GIÁ TRỊ DỄ SỬA NHẤT (chỉ cần đổi args):
   --lr 1e-5 \\                    # Giảm head LR từ 5e-5 → 1e-5
   --weight_decay 0.005 \\         # Tăng từ 0.00075 → 0.005
   --reduce_dim 256 \\             # Giảm từ 512 → 256
   --warmup_epochs 5 \\            # Tăng từ 3 → 5

B. FREEZE BACKBONE (code đơn giản):
   Freeze backbone 3-5 epoch đầu, chỉ train head
   → Head ổn định trước, sau đó mới fine-tune backbone

C. TĂNG AUGMENTATION:
   Kiểm tra data/transforms.py, tăng cường augmentation
   (flip, crop, color jitter, etc.)

D. THÊM DROPOUT:
   Trong methods/meta_deepbdc.py, thêm dropout sau projection

E. GIẢM N_WAY TRAINING:
   --train_n_way 5 thay vì 10
   → Dễ học hơn, tránh confusion
""")

print("\n" + "="*60)
print("LỆNH CHẠY ĐỀ XUẤT (copy vào Kaggle)")
print("="*60)

print("""
# Option A: Chỉ đổi hyperparameters (NHANH NHẤT - THỬ ĐẦU TIÊN)
python meta_train.py \\
    --dataset kinetics400_mini \\
    --data_path /kaggle/working/TAMT/filelist/kinetics400_mini/machine_01 \\
    --model VideoMAES \\
    --method meta_deepbdc \\
    --lr 1e-5 \\
    --weight_decay 0.005 \\
    --reduce_dim 256 \\
    --warmup_epochs 5 \\
    --train_n_way 5 \\
    --val_n_way 5 \\
    --n_shot 5 \\
    --pretrain_path /kaggle/working/112112vit-s-140epoch.pt \\
    --gpu 0,1 \\
    --epoch 10

# Option B: Thêm freeze backbone (cần sửa code một chút)
# Xem script freeze_backbone_fix.py bên dưới
""")
