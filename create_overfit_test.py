"""
Quick overfit test - kiểm tra xem model CÓ THỂ học được hay không
Chạy test này với chỉ 2 classes, 20 episodes
Nếu không đạt >90% accuracy → có bug trong model/data
"""

import torch
import json
import numpy as np
from collections import defaultdict

print("="*60)
print("QUICK OVERFIT TEST")
print("="*60)

# Tạo mini subset từ base.json
base_json_path = './filelist/kinetics400_mini/machine_01/base.json'
with open(base_json_path, 'r') as f:
    base_data = json.load(f)

# Chọn 2 classes đầu tiên và lấy 20 samples mỗi class
samples_by_class = defaultdict(list)
for img, lbl in zip(base_data['image_names'], base_data['image_labels']):
    samples_by_class[lbl].append(img)

# Lấy 2 classes có nhiều samples nhất
sorted_classes = sorted(samples_by_class.items(), key=lambda x: len(x[1]), reverse=True)
selected_classes = sorted_classes[:2]

mini_data = {
    'image_names': [],
    'image_labels': []
}

for cls_idx, (cls, samples) in enumerate(selected_classes):
    # Lấy tối đa 20 samples
    selected_samples = samples[:min(20, len(samples))]
    mini_data['image_names'].extend(selected_samples)
    mini_data['image_labels'].extend([cls_idx] * len(selected_samples))

print(f"\nĐã tạo mini dataset:")
print(f"  - 2 classes")
print(f"  - {len(mini_data['image_names'])} samples total")
print(f"  - Class 0: {mini_data['image_labels'].count(0)} samples")
print(f"  - Class 1: {mini_data['image_labels'].count(1)} samples")

# Lưu mini dataset
mini_json_path = './filelist/kinetics400_mini/machine_01/mini_overfit_test.json'
with open(mini_json_path, 'w') as f:
    json.dump(mini_data, f, indent=2)

print(f"\n✓ Đã lưu: {mini_json_path}")

print("\n" + "="*60)
print("LỆNH CHẠY TEST")
print("="*60)

test_command = f"""
# Copy mini_overfit_test.json lên Kaggle, sau đó chạy:

python meta_train.py \\
    --dataset kinetics400_mini \\
    --data_path /kaggle/working/TAMT/filelist/kinetics400_mini/machine_01 \\
    --model VideoMAES \\
    --method meta_deepbdc \\
    --lr 1e-4 \\
    --weight_decay 0.001 \\
    --reduce_dim 256 \\
    --train_n_way 2 \\
    --val_n_way 2 \\
    --n_shot 5 \\
    --n_query 5 \\
    --train_n_episode 20 \\
    --val_n_episode 20 \\
    --epoch 5 \\
    --pretrain_path /kaggle/working/112112vit-s-140epoch.pt \\
    --gpu 0,1

# TRƯỚC KHI CHẠY: Sửa data/datamgr.py để load mini_overfit_test.json
# Hoặc tạo symbolic link:
#   ln -s mini_overfit_test.json base.json
#   ln -s mini_overfit_test.json val.json

KỲ VỌNG:
  - Train accuracy nên lên >90% sau 3-5 epochs
  - Val accuracy cũng nên >80% (vì cùng 2 classes)
  - Nếu KHÔNG đạt → có bug trong model hoặc data loading
"""

print(test_command)

print("\n" + "="*60)
print("TÓM TẮT CÁC GIẢI PHÁP")
print("="*60)

summary = """
GIẢI PHÁP ƯU TIÊN (thử theo thứ tự):

1. ⭐ NHANH NHẤT - CHỈ ĐỔI ARGS (không cần sửa code):
   
   python meta_train.py \\
       --lr 1e-5 \\                        # Giảm từ 5e-5 → 1e-5
       --weight_decay 0.005 \\             # Tăng từ 0.00075 → 0.005  
       --reduce_dim 256 \\                 # Giảm từ 512 → 256
       --train_n_way 5 \\                  # Giảm từ 10 → 5
       --warmup_epochs 5 \\                # Tăng từ 3 → 5
       [... các args khác giữ nguyên]

   Lý do: Head learning rate quá cao (5e-5) → head học quá nhanh → overfit

2. ⭐ FREEZE BACKBONE (cần sửa code):
   
   - Chạy: python fix_freeze_backbone.py
   - Sau đó thêm arg: --freeze_backbone_epochs 3
   
   Lý do: Head cần ổn định trước khi fine-tune backbone

3. THÊM DROPOUT (sửa methods/meta_deepbdc.py):
   
   Thêm nn.Dropout(0.3) sau projection layer
   
4. TĂNG AUGMENTATION (sửa data/transforms.py):
   
   Tăng cường random crop, color jitter, etc.

5. THAY ĐỔI SAMPLING:
   
   --train_n_episode 1000  # Tăng từ 500 → 1000 để đa dạng hơn

QUAN TRỌNG:
  - Thử giải pháp 1 TRƯỚC (dễ nhất, chỉ đổi args)
  - Monitor val loss thay vì val acc (loss stable hơn)
  - Nếu val loss vẫn tăng → thử giải pháp 2 (freeze)
  - Nếu vẫn không khá → kiểm tra pretrained weights và data
"""

print(summary)
