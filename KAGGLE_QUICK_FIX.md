# ⚡ KAGGLE QUICK FIX - CHỈ CẦN COPY PASTE!

## 🔥 TÓM TẮT VẤN ĐỀ

**Lỗi gặp phải:**
```bash
python3: can't open file '/kaggle/working/fix_freeze_backbone.py': [Errno 2]
```

**Nguyên nhân:** 
- Script chạy ở `/kaggle/working/`
- Nhưng files ở `/kaggle/working/TAMT/`

**Giải pháp:** 
CD vào `/kaggle/working/TAMT/` trước khi chạy!

---

## ✅ GIẢI PHÁP 1: OPTION 1 (KHUYẾN NGHỊ)

### Chỉ cần 3 cells trong Kaggle notebook:

#### **Cell 1:**
```python
%cd /kaggle/working/TAMT
```

#### **Cell 2 (optional - để kiểm tra):**
```python
!ls -la meta_train.py
!ls -la filelist/kinetics400_mini/machine_01/base.json
!ls -la /kaggle/working/112112vit-s-140epoch.pt
```

#### **Cell 3:**
```python
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

**DONE! Chỉ vậy thôi!**

---

## ✅ GIẢI PHÁP 2: OPTION 2 (NẾU CẦN FREEZE)

### Nếu Option 1 vẫn overfit, dùng Option 2:

#### **Cell 1:**
```python
%cd /kaggle/working/TAMT
```

#### **Cell 2:**
```python
!python3 fix_freeze_backbone.py
```

#### **Cell 3:**
```python
!cp meta_train.py meta_train_original_backup.py
!cp meta_train_with_freeze.py meta_train.py
```

#### **Cell 4:**
```python
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

## 📊 KẾT QUẢ MONG ĐỢI

### Option 1:
```
Val Accuracy: 45-50% (tăng từ 36%)
Gap: ~35-40%
```

### Option 2:
```
Val Accuracy: 50-56% (tăng từ 36%)
Gap: ~20-25%
✅ Tốt nhất!
```

---

## 📁 FILES ĐÃ TẠO

- ✅ `KAGGLE_GUIDE.md` - Hướng dẫn chi tiết
- ✅ `KAGGLE_COPY_PASTE_OPTION1.py` - Copy paste Option 1
- ✅ `KAGGLE_COPY_PASTE_OPTION2.py` - Copy paste Option 2
- ✅ `kaggle_option1.sh` - Bash script Option 1
- ✅ `kaggle_option2.sh` - Bash script Option 2

---

## 🎯 KHUYẾN NGHỊ

1. **Thử OPTION 1 TRƯỚC** (đơn giản, 3 cells)
2. Nếu vẫn overfit → Thử OPTION 2 (4 cells)
3. So sánh kết quả với training ban đầu

---

**Bắt đầu với Option 1 ngay! Copy 3 cells ở trên vào Kaggle! 🚀**
