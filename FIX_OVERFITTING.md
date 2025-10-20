# 🔧 Hướng Dẫn Sửa Lỗi Overfitting

## 📊 Vấn Đề Hiện Tại

Kết quả training cho thấy **overfitting nghiêm trọng**:

```
Epoch 1: Train 45.54% → Val 37.53% ✓ (hợp lý)
Epoch 2: Train 82.64% → Val 38.33% ⚠️ (train tăng nhanh)
Epoch 3: Train 97.30% → Val 37.12% ❌ (train gần 100%, val giảm)
...
Epoch 10: Train 99.97% → Val 36.60% ❌ (overfitting hoàn toàn)
```

**Nguyên nhân:**
- Model "học thuộc lòng" training episodes thay vì học features tổng quát
- Head learning rate quá cao (5e-5) so với backbone (5e-6)
- Weight decay quá thấp (0.00075)
- Reduce dimension quá lớn (512) cho few-shot learning

---

## ✅ GIẢI PHÁP 1: ĐỔI HYPERPARAMETERS (DỄ NHẤT - THỬ TRƯỚC)

### Chạy lệnh này thay vì lệnh cũ:

```bash
python meta_train.py \
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
    --n_query 10 \
    --warmup_epochs 5 \
    --pretrain_path /kaggle/working/112112vit-s-140epoch.pt \
    --gpu 0,1 \
    --epoch 10 \
    --patience 8
```

### Thay đổi so với lệnh cũ:

| Tham số | Cũ | Mới | Lý do |
|---------|-----|-----|-------|
| `--lr` | 5e-5 | **1e-5** | Giảm head LR để tránh học quá nhanh |
| `--weight_decay` | 0.00075 | **0.005** | Tăng regularization |
| `--reduce_dim` | 512 | **256** | Giảm capacity, tránh overfit |
| `--train_n_way` | 10 | **5** | Dễ học hơn, ít confusion |
| `--warmup_epochs` | 3 | **5** | Warmup lâu hơn cho stable |

### Kỳ vọng:
- Train accuracy không nên vượt 80-85%
- Val accuracy nên ổn định hoặc tăng nhẹ
- Val loss không nên tăng liên tục

---

## ✅ GIẢI PHÁP 2: FREEZE BACKBONE (NẾU GIẢI PHÁP 1 KHÔNG ĐỦ)

### Bước 1: Chạy script fix

```bash
cd /kaggle/working/TAMT
python fix_freeze_backbone.py
```

Script sẽ tạo file `meta_train_with_freeze.py` với chức năng freeze backbone.

### Bước 2: Copy file mới

```bash
cp meta_train_with_freeze.py meta_train.py
```

### Bước 3: Chạy với freeze option

```bash
python meta_train.py \
    --freeze_backbone_epochs 3 \
    --lr 1e-5 \
    --weight_decay 0.005 \
    --reduce_dim 256 \
    --train_n_way 5 \
    --val_n_way 5 \
    [... các args khác như cũ]
```

### Hoạt động:
- Epoch 0-2: **Chỉ train head**, backbone freeze
- Epoch 3+: **Train toàn bộ** (unfreeze backbone)
- Head ổn định trước → backbone fine-tune sau → tránh overfitting

---

## 🧪 GIẢI PHÁP 3: KIỂM TRA OVERFIT TEST (OPTIONAL)

Để verify model **CÓ KHẢ NĂNG** học, chạy test nhỏ:

```bash
python create_overfit_test.py
```

Script tạo mini dataset với 2 classes, 20 samples. Nếu model không overfit được dataset nhỏ này → có bug.

---

## 📋 CHECKLIST TRƯỚC KHI CHẠY

- [ ] Code đã update trên Kaggle (git clone mới nhất)
- [ ] File JSON paths đúng (`/kaggle/input/k400mc01/...`)
- [ ] Pretrained checkpoint tồn tại (`/kaggle/working/112112vit-s-140epoch.pt`)
- [ ] GPU enabled (`--gpu 0,1`)
- [ ] Đã commit + push code nếu sửa local

---

## 📊 MONITOR TRONG LÚC TRAINING

### Dấu hiệu TỐT:
- Train acc tăng dần, dừng ở ~70-85%
- Val acc tăng hoặc ổn định
- Val loss giảm hoặc ổn định
- Gap giữa train và val không quá lớn (<15%)

### Dấu hiệu XẤU (vẫn overfit):
- Train acc > 95%
- Val acc giảm sau vài epoch
- Val loss tăng liên tục
- Gap train-val > 30%

Nếu vẫn overfit → thử:
- Giảm `--lr` thêm (5e-6)
- Tăng `--weight_decay` (0.01)
- Tăng `--freeze_backbone_epochs` (5)
- Giảm `--reduce_dim` (128)

---

## 🔍 DEBUG SCRIPTS

### 1. Kiểm tra dataset và pretrained weights:

```bash
python diagnostic_check.py
```

Output:
- Dataset có load được không
- Số class, samples per class
- Pretrained checkpoint có đúng không
- Phân tích nguyên nhân overfitting

### 2. Tạo mini overfit test:

```bash
python create_overfit_test.py
```

Output:
- File `mini_overfit_test.json` (2 classes, 40 samples)
- Lệnh chạy quick test

---

## 💡 TÓM TẮT NHANH

**Nếu bạn chỉ có thời gian thử 1 thứ:**

```bash
# Chạy lệnh này (đã optimize tất cả hyperparameters):
python meta_train.py \
    --lr 1e-5 \
    --weight_decay 0.005 \
    --reduce_dim 256 \
    --train_n_way 5 \
    --warmup_epochs 5 \
    [các args khác giữ nguyên]
```

**Kỳ vọng sau 10 epochs:**
- Train: 70-85%
- Val: 42-48% (tăng từ 38%)

**Nếu val vẫn ~36-38%:** Thử Giải pháp 2 (freeze backbone)

---

## 📞 Troubleshooting

### Lỗi: FileNotFoundError cho .pt files
```bash
# Kiểm tra mount name
ls /kaggle/input/

# Sửa paths trong JSON nếu cần
python -c "import json; ..."
```

### Lỗi: Model không load pretrained weights
```bash
# Verify checkpoint
python diagnostic_check.py
```

### Val accuracy không cải thiện
1. Thử lr thấp hơn (`1e-6`)
2. Freeze backbone (`--freeze_backbone_epochs 5`)
3. Kiểm tra data augmentation
4. Tăng train episodes (`--train_n_episode 1000`)

---

## 📈 Kết Quả Mong Đợi

Sau khi apply các fixes:

| Metric | Trước | Sau (mục tiêu) |
|--------|-------|----------------|
| Train Acc | 99.97% | 70-85% |
| Val Acc | 36-38% | 42-48% |
| Val Loss | 3.76 (tăng) | ~2.5 (ổn định) |
| Overfit Gap | 63% | <15% |

---

## 📁 Files Đã Tạo

1. `diagnostic_check.py` - Kiểm tra dataset và pretrained weights
2. `fix_freeze_backbone.py` - Thêm freeze backbone vào meta_train.py
3. `create_overfit_test.py` - Tạo mini test dataset
4. `FIX_OVERFITTING.md` - File này

Chúc bạn training thành công! 🚀
