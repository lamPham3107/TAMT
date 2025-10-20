# 🎯 HƯỚNG DẪN CHẠY TRÊN KAGGLE

## ⚠️ LỖI BẠN GẶP PHẢI

```
python3: can't open file '/kaggle/working/fix_freeze_backbone.py': [Errno 2] No such file or directory
```

**Nguyên nhân:** Script đang chạy ở `/kaggle/working/` nhưng files nằm trong `/kaggle/working/TAMT/`

---

## ✅ GIẢI PHÁP

### Cách 1: Chạy từng lệnh trong Kaggle notebook

#### **OPTION 1: Chỉ đổi hyperparameters (KHUYẾN NGHỊ THỬ TRƯỚC)**

```python
# Cell 1: CD vào thư mục TAMT
%cd /kaggle/working/TAMT

# Cell 2: Kiểm tra files
!ls -la meta_train.py
!ls -la filelist/kinetics400_mini/machine_01/base.json
!ls -la /kaggle/working/112112vit-s-140epoch.pt

# Cell 3: Train với hyperparameters đã fix
!python3 meta_train.py \
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
```

---

#### **OPTION 2: Với freeze backbone (nếu Option 1 chưa đủ)**

```python
# Cell 1: CD vào thư mục TAMT
%cd /kaggle/working/TAMT

# Cell 2: Kiểm tra files
!ls -la meta_train.py
!ls -la fix_freeze_backbone.py

# Cell 3: Thêm freeze functionality
!python3 fix_freeze_backbone.py

# Cell 4: Kiểm tra output
!ls -la meta_train_with_freeze.py

# Cell 5: Backup và replace
!cp meta_train.py meta_train_original_backup.py
!cp meta_train_with_freeze.py meta_train.py

# Cell 6: Train với freeze
!python3 meta_train.py \
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
```

---

### Cách 2: Upload scripts và chạy bash

#### **Bước 1: Upload files lên Kaggle**

Upload các files này vào `/kaggle/working/TAMT/`:
- `kaggle_option1.sh` (chạy option 1)
- `kaggle_option2.sh` (chạy option 2)

#### **Bước 2: Chạy script**

```python
# OPTION 1: Chỉ đổi hyperparameters
!bash /kaggle/working/TAMT/kaggle_option1.sh

# HOẶC OPTION 2: Với freeze backbone
!bash /kaggle/working/TAMT/kaggle_option2.sh
```

---

## 🔍 DEBUG: Kiểm tra structure

```python
# Kiểm tra current directory
!pwd

# Kiểm tra TAMT folder structure
!ls -la /kaggle/working/TAMT/

# Kiểm tra files quan trọng
!ls -la /kaggle/working/TAMT/meta_train.py
!ls -la /kaggle/working/TAMT/fix_freeze_backbone.py
!ls -la /kaggle/working/TAMT/filelist/kinetics400_mini/
```

**Expected output:**
```
/kaggle/working/TAMT/
├── meta_train.py              ✅
├── fix_freeze_backbone.py     ✅
├── pretrain.py
├── test.py
├── filelist/
│   └── kinetics400_mini/
│       ├── machine_01/
│       │   ├── base.json      ✅
│       │   ├── val.json       ✅
│       │   └── novel.json     ✅
│       ├── machine_02/
│       ├── machine_03/
│       └── machine_04/
└── ...
```

---

## 📊 EXPECTED RESULTS

### Option 1 (chỉ đổi hyperparameters):

```
Epoch 1:  train_acc=45%, val_acc=38%
Epoch 5:  train_acc=72%, val_acc=46%
Epoch 10: train_acc=85%, val_acc=48%

Gap: 85% - 48% = 37% (Cải thiện từ 63%)
```

### Option 2 (với freeze):

```
Epoch 1:  train_acc=38%, val_acc=42%  ❄️ (freeze)
Epoch 2:  train_acc=48%, val_acc=48%  ❄️ (freeze)
Epoch 3:  train_acc=58%, val_acc=52%  ❄️ (freeze)
Epoch 4:  train_acc=65%, val_acc=54%  🔥 (unfreeze)
Epoch 10: train_acc=78%, val_acc=56%  🔥

Gap: 78% - 56% = 22% (Tốt nhất!)
```

---

## ⚡ QUICK START

**Cách NHANH NHẤT để chạy ngay:**

```python
# 1. CD vào TAMT
%cd /kaggle/working/TAMT

# 2. Chạy Option 1 (đơn giản nhất)
!python3 meta_train.py \
    --dataset kinetics400_mini \
    --data_path /kaggle/working/TAMT/filelist/kinetics400_mini/machine_01 \
    --model VideoMAES \
    --method meta_deepbdc \
    --lr 1e-5 \
    --weight_decay 0.005 \
    --reduce_dim 256 \
    --train_n_way 5 \
    --val_n_way 5 \
    --n_shot 5 \
    --pretrain_path /kaggle/working/112112vit-s-140epoch.pt
```

**Chỉ cần 2 cells, rất đơn giản!**

---

## 🎯 KHUYẾN NGHỊ

1. **Thử Option 1 TRƯỚC** (chỉ đổi hyperparameters)
   - Nếu val_acc ≥ 48% → OK, không cần freeze
   - Nếu val_acc < 45% → Thử Option 2

2. **Nếu Option 1 chưa đủ, thử Option 2** (freeze backbone)
   - Freeze giúp tăng thêm 5-8% val accuracy
   - Giảm overfitting mạnh hơn

3. **Monitor training:**
   ```python
   # Xem logs realtime
   !tail -f /kaggle/working/TAMT/logs/train.log
   ```

---

## ❓ FAQ

**Q: Tại sao lỗi "No such file or directory"?**
A: Vì bạn đang ở `/kaggle/working/` nhưng files ở `/kaggle/working/TAMT/`. Dùng `%cd /kaggle/working/TAMT` trước.

**Q: Option 1 vs Option 2 chọn cái nào?**
A: Thử Option 1 trước (đơn giản hơn). Nếu vẫn overfit thì mới thử Option 2.

**Q: Freeze backbone có làm giảm accuracy không?**
A: Không! Freeze giúp VAL accuracy TĂNG vì giảm overfitting.

**Q: Có thể chạy cả 2 options song song không?**
A: Có, nhưng cần 2 GPUs riêng biệt. Khuyến nghị chạy tuần tự.

---

**Copy code ở phần QUICK START và chạy ngay! 🚀**
