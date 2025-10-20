"""
Script sửa meta_train.py để thêm freeze backbone option
Chạy: python fix_freeze_backbone.py
"""

import re

# Đọc file
with open('meta_train.py', 'r', encoding='utf-8') as f:
    content = f.read()

# 1. Thêm argument freeze_backbone_epochs
if '--freeze_backbone_epochs' not in content:
    # Tìm vị trí thêm argument (sau --warmup_epochs)
    warmup_pattern = r"(parser\.add_argument\('--warmup_epochs'[^\)]+\))"
    if re.search(warmup_pattern, content):
        new_arg = """
    parser.add_argument('--freeze_backbone_epochs', type=int, default=0,
                        help='Freeze backbone for first N epochs (0=no freeze)')"""
        content = re.sub(warmup_pattern, r"\1" + new_arg, content, count=1)
        print("✓ Đã thêm argument --freeze_backbone_epochs")
    else:
        print("⚠ Không tìm thấy warmup_epochs argument")

# 2. Thêm freeze/unfreeze functions
freeze_functions = '''
def freeze_backbone(model):
    """Freeze backbone parameters"""
    frozen_count = 0
    for n, p in model.named_parameters():
        if ('backbone' in n) or ('encoder' in n):
            p.requires_grad = False
            frozen_count += 1
    return frozen_count

def unfreeze_backbone(model):
    """Unfreeze backbone parameters"""
    unfrozen_count = 0
    for n, p in model.named_parameters():
        if ('backbone' in n) or ('encoder' in n):
            p.requires_grad = True
            unfrozen_count += 1
    return unfrozen_count
'''

# Tìm vị trí thêm functions (sau def load_back)
load_back_pattern = r"(def load_back\(model, backup\):[\s\S]*?model\.load_state_dict\(backup, strict=False\))"
if re.search(load_back_pattern, content):
    content = re.sub(load_back_pattern, r"\1\n" + freeze_functions, content, count=1)
    print("✓ Đã thêm freeze_backbone và unfreeze_backbone functions")
else:
    print("⚠ Không tìm thấy load_back function")

# 3. Thêm freeze logic trong training loop
# Tìm vị trí: for epoch in range(stop_epoch):
epoch_loop_pattern = r"(for epoch in range\(stop_epoch\):)"
freeze_logic = '''
        # Freeze backbone for first N epochs if specified
        if epoch == 0 and params.freeze_backbone_epochs > 0:
            model_to_freeze = model.module if isinstance(model, nn.DataParallel) else model
            count = freeze_backbone(model_to_freeze)
            print(f"  ⚄ Freezing backbone ({count} params) for {params.freeze_backbone_epochs} epochs")
        elif epoch == params.freeze_backbone_epochs and params.freeze_backbone_epochs > 0:
            model_to_unfreeze = model.module if isinstance(model, nn.DataParallel) else model
            count = unfreeze_backbone(model_to_unfreeze)
            print(f"  ⚄ Unfreezing backbone ({count} params) at epoch {epoch}")
            # Rebuild optimizer with unfrozen params
            optimizer, scheduler = make_optimizer_and_scheduler(model, params, stop_epoch)
'''

if re.search(epoch_loop_pattern, content):
    # Thêm sau "for epoch in range..." và sau "t0 = time.time()"
    pattern_with_time = r"(for epoch in range\(stop_epoch\):[\s]*\n[\s]*t0 = time\.time\(\))"
    if re.search(pattern_with_time, content):
        content = re.sub(pattern_with_time, r"\1" + freeze_logic, content, count=1)
        print("✓ Đã thêm freeze logic vào training loop")
    else:
        print("⚠ Không tìm thấy pattern 'for epoch ... t0 = time.time()'")
else:
    print("⚠ Không tìm thấy epoch loop")

# Ghi file đã sửa
with open('meta_train_with_freeze.py', 'w', encoding='utf-8') as f:
    f.write(content)

print("\n✓ Đã tạo file mới: meta_train_with_freeze.py")
print("\nCách sử dụng:")
print("  python meta_train_with_freeze.py --freeze_backbone_epochs 3 [other args...]")
print("\nĐể test local trước:")
print("  cp meta_train_with_freeze.py meta_train.py")
