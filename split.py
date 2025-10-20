import sys
import os
import json
from pathlib import Path
from collections import defaultdict
import math
import random
#cd G:\TLU\BigData\Data_time\TAMT
#python mapper.py "G:\TLU\BigData\Data_time\data_down\hmdb51_org_2" --chunk_size 50 --output_dir "./my_chunks"
def collect_videos_by_label(data_root):
    """Thu thập videos theo label - Hỗ trợ cả .avi và .mp4"""
    videos_by_label = defaultdict(list)
    
    # Duyệt đệ quy để lấy toàn bộ file video, giữ nguyên đường dẫn tương đối từ data_root
    for video_file in Path(data_root).rglob("*.avi"):
        rel_path = video_file.relative_to(data_root).as_posix()
        # label là thư mục cha của video (ví dụ: train/abseiling)
        label = str(Path(rel_path).parent)
        videos_by_label[label].append(rel_path)
    for video_file in Path(data_root).rglob("*.mp4"):
        rel_path = video_file.relative_to(data_root).as_posix()
        label = str(Path(rel_path).parent)
        videos_by_label[label].append(rel_path)
    
    return videos_by_label

def create_balanced_chunks(videos_by_label, chunks_per_machine=25, dataset_name="hmdb51_org_2", split_by_label=False):
    """Tạo chunks balanced cho 4 machines - MỖI MACHINE CÓ BASE/VAL/NOVEL
    
    Chiến lược mới:
    - Chia labels thành 4 nhóm (mỗi máy 1 nhóm riêng)
    - Mỗi machine: 60% base + 20% val + 20% novel classes
    """
    
    # Create label mapping
    all_labels = sorted(videos_by_label.keys())
    label_to_idx = {label: idx for idx, label in enumerate(all_labels)}
    
    machine_groups = []
    
    print(f"🏷️  SPLIT MODE: Mỗi máy có BASE/VAL/NOVEL từ nhóm labels riêng")
    
    # Chia labels thành 4 nhóm
    labels_per_machine = len(all_labels) // 4
    
    for machine_id in range(4):
        start_label_idx = machine_id * labels_per_machine
        if machine_id == 3:  # Machine cuối lấy hết
            end_label_idx = len(all_labels)
        else:
            end_label_idx = start_label_idx + labels_per_machine
        
        # Lấy labels cho machine này
        machine_labels = all_labels[start_label_idx:end_label_idx]
        
        # Shuffle labels để random split
        shuffled_machine_labels = machine_labels.copy()
        random.shuffle(shuffled_machine_labels)
        
        # Chia labels thành base/val/novel (60/20/20)
        n_labels = len(shuffled_machine_labels)
        n_base = int(n_labels * 0.60)
        n_val = int(n_labels * 0.20)
        
        base_labels = sorted(shuffled_machine_labels[:n_base])
        val_labels = sorted(shuffled_machine_labels[n_base:n_base + n_val])
        novel_labels = sorted(shuffled_machine_labels[n_base + n_val:])
        
        print(f"\n   Machine {machine_id+1}: Total {len(machine_labels)} labels")
        print(f"      Base: {len(base_labels)} classes ({len(base_labels)/n_labels*100:.1f}%)")
        print(f"      Val: {len(val_labels)} classes ({len(val_labels)/n_labels*100:.1f}%)")
        print(f"      Novel: {len(novel_labels)} classes ({len(novel_labels)/n_labels*100:.1f}%)")
        
        # Collect videos cho từng split
        def collect_split_videos(split_labels, split_name):
            """Helper function to collect videos for a split"""
            split_videos = []
            for label in split_labels:
                videos = videos_by_label[label]
                shuffled_videos = videos.copy()
                random.shuffle(shuffled_videos)
                
                for video in shuffled_videos:
                    # Format đường dẫn cho Kaggle: /kaggle/input/kinetics400-mini/kinetics400_mini/
                    kaggle_path = f"/kaggle/input/kinetics400-mini/kinetics400_mini/train/{video}"
                    split_videos.append({
                        "kaggle_path": kaggle_path,
                        "label": label,
                        "label_idx": label_to_idx[label]
                    })
            return split_videos
        
        base_videos = collect_split_videos(base_labels, "base")
        val_videos = collect_split_videos(val_labels, "val")
        novel_videos = collect_split_videos(novel_labels, "novel")
        
        print(f"      Videos: {len(base_videos)} base + {len(val_videos)} val + {len(novel_videos)} novel")
        
        # Tạo 1 chunk chứa tất cả data của machine này
        machine_chunks = [{
            "base_data": {
                "image_names": [v["kaggle_path"] for v in base_videos],
                "image_labels": [v["label_idx"] for v in base_videos]
            },
            "val_data": {
                "image_names": [v["kaggle_path"] for v in val_videos],
                "image_labels": [v["label_idx"] for v in val_videos]
            },
            "novel_data": {
                "image_names": [v["kaggle_path"] for v in novel_videos],
                "image_labels": [v["label_idx"] for v in novel_videos]
            },
            "metadata": {
                "base_classes": base_labels,
                "val_classes": val_labels,
                "novel_classes": novel_labels,
                "total_videos": len(base_videos) + len(val_videos) + len(novel_videos)
            }
        }]
        
        machine_groups.append(machine_chunks)
    
    return machine_groups, label_to_idx

def save_machine_folders(machine_groups, label_to_idx, output_dir="./my_chunks"):
    """Tạo 4 folders cho 4 machines - MỖI MACHINE CÓ BASE/VAL/NOVEL"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Save global label mapping
    with open(output_path / "label_mapping.json", "w") as f:
        json.dump(label_to_idx, f, indent=2)
    
    machine_folders = []
    
    for machine_id, chunks in enumerate(machine_groups, 1):
        # Tạo folder cho machine
        machine_folder = output_path / f"machine_{machine_id:02d}"
        machine_folder.mkdir(exist_ok=True)
        
        # Combine tất cả chunks của machine này
        all_base_names = []
        all_base_labels = []
        all_val_names = []
        all_val_labels = []
        all_novel_names = []
        all_novel_labels = []
        
        for chunk in chunks:
            all_base_names.extend(chunk["base_data"]["image_names"])
            all_base_labels.extend(chunk["base_data"]["image_labels"])
            all_val_names.extend(chunk["val_data"]["image_names"])
            all_val_labels.extend(chunk["val_data"]["image_labels"])
            all_novel_names.extend(chunk["novel_data"]["image_names"])
            all_novel_labels.extend(chunk["novel_data"]["image_labels"])
        
        # Save base.json
        base_data = {
            "image_names": all_base_names,
            "image_labels": all_base_labels
        }
        with open(machine_folder / "base.json", "w") as f:
            json.dump(base_data, f, indent=2)
        
        # Save val.json
        val_data = {
            "image_names": all_val_names,
            "image_labels": all_val_labels
        }
        with open(machine_folder / "val.json", "w") as f:
            json.dump(val_data, f, indent=2)
        
        # Save novel.json
        novel_data = {
            "image_names": all_novel_names,
            "image_labels": all_novel_labels
        }
        with open(machine_folder / "novel.json", "w") as f:
            json.dump(novel_data, f, indent=2)
        
        # Save machine info
        machine_info = {
            "machine_id": machine_id,
            "total_chunks": len(chunks),
            "total_videos": len(all_base_names) + len(all_val_names) + len(all_novel_names),
            "training_videos": len(all_base_names),
            "validation_videos": len(all_val_names),
            "novel_videos": len(all_novel_names),
            "unique_labels": len(set(all_base_labels + all_val_labels + all_novel_labels)),
            "labels_distribution": {
                str(label): (all_base_labels + all_val_labels + all_novel_labels).count(label) 
                for label in set(all_base_labels + all_val_labels + all_novel_labels)
            }
        }
        with open(machine_folder / "machine_info.json", "w") as f:
            json.dump(machine_info, f, indent=2)
        
        machine_folders.append({
            "machine_id": machine_id,
            "folder_path": str(machine_folder),
            "training_videos": len(all_base_names),
            "validation_videos": len(all_val_names),
            "novel_videos": len(all_novel_names),
            "unique_labels": len(set(all_base_labels + all_val_labels + all_novel_labels))
        })
        
        print(f"✅ Machine {machine_id}: {len(all_base_names)} base + {len(all_val_names)} val + {len(all_novel_names)} novel")
        print(f"   📁 Folder: {machine_folder}")
        print(f"   🏷️  Labels: {len(set(all_base_labels + all_val_labels + all_novel_labels))}")
    
    # Save overall summary
    summary = {
        "total_machines": 4,
        "machine_folders": machine_folders,
        "label_mapping": label_to_idx,
        "total_labels": len(label_to_idx),
        "usage_instructions": {
            "meta_train_command": "python meta_train.py --dataset hmdb51 --data_path ./my_chunks/machine_01",
            "available_folders": [f"machine_{i:02d}" for i in range(1, 5)]
        }
    }
    
    with open(output_path / "summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    
    return machine_folders

def create_base_val_novel_split(videos_by_label, output_dir, kaggle_dataset_name, base_ratio=0.60, val_ratio=0.20):
    """Tạo split base/val/novel với đường dẫn Kaggle chuẩn
    
    Args:
        base_ratio: 60% classes cho base (training)
        val_ratio: 20% classes cho val (validation)
        novel_ratio: 20% classes cho novel (testing) - auto calculated
        kaggle_dataset_name: Tên dataset trên Kaggle (ví dụ: "k400bigdata")
    """
    
    # Create label mapping
    all_labels = sorted(videos_by_label.keys())
    
    # Shuffle labels để random split
    import random
    random.seed(42)
    shuffled_labels = all_labels.copy()
    random.shuffle(shuffled_labels)
    
    # Split labels thành base/val/novel
    n_labels = len(shuffled_labels)
    n_base = int(n_labels * base_ratio)
    n_val = int(n_labels * val_ratio)
    
    base_labels = sorted(shuffled_labels[:n_base])
    val_labels = sorted(shuffled_labels[n_base:n_base + n_val])
    novel_labels = sorted(shuffled_labels[n_base + n_val:])
    
    print(f"\n📊 Split Strategy:")
    print(f"   Base: {len(base_labels)} classes ({len(base_labels)/n_labels*100:.1f}%)")
    print(f"   Val: {len(val_labels)} classes ({len(val_labels)/n_labels*100:.1f}%)")
    print(f"   Novel: {len(novel_labels)} classes ({len(novel_labels)/n_labels*100:.1f}%)")
    
    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Process each split
    splits = {
        'base': base_labels,
        'val': val_labels,
        'novel': novel_labels
    }
    
    split_stats = {}
    
    for split_name, split_labels in splits.items():
        # Create label mapping for this split (0-indexed)
        label_to_idx = {label: idx for idx, label in enumerate(split_labels)}
        
        image_names = []
        image_labels = []
        
        print(f"\n📁 Processing {split_name} split ({len(split_labels)} classes)...")
        
        for label in split_labels:
            videos = videos_by_label[label]
            label_idx = label_to_idx[label]
            
            for video in videos:
                # ⭐ Đường dẫn Kaggle chuẩn:
                # /kaggle/input/kinetics400-mini/kinetics400_mini/train/abseiling/-WKCwDRp_jk.mp4
                kaggle_path = f"/kaggle/input/{kaggle_dataset_name}/kinetics400_mini/train/{video}"
                image_names.append(kaggle_path)
                image_labels.append(label_idx)
            
            print(f"   ✅ {label}: {len(videos)} videos (label={label_idx})")
        
        # Save JSON file
        split_data = {
            "image_names": image_names,
            "image_labels": image_labels
        }
        
        output_file = output_path / f"{split_name}.json"
        with open(output_file, "w", encoding='utf-8') as f:
            json.dump(split_data, f, indent=2)
        
        split_stats[split_name] = {
            "classes": len(split_labels),
            "videos": len(image_names),
            "avg_videos_per_class": len(image_names) / len(split_labels) if split_labels else 0
        }
        
        print(f"   💾 Saved: {output_file}")
        print(f"   📊 Total: {len(image_names)} videos, {len(split_labels)} classes")
    
    # Save class mapping
    class_mapping = {
        "base_classes": base_labels,
        "val_classes": val_labels,
        "novel_classes": novel_labels,
        "statistics": split_stats
    }
    
    with open(output_path / "class_mapping.json", "w", encoding='utf-8') as f:
        json.dump(class_mapping, f, indent=2)
    
    # Save label mapping for each split (for reference)
    for split_name, split_labels in splits.items():
        label_mapping = {label: idx for idx, label in enumerate(split_labels)}
        with open(output_path / f"{split_name}_label_mapping.json", "w", encoding='utf-8') as f:
            json.dump(label_mapping, f, indent=2)
    
    print(f"\n✅ Split completed!")
    print(f"📁 Output directory: {output_dir}")
    print(f"\n📊 Statistics:")
    for split_name, stats in split_stats.items():
        print(f"   {split_name.upper()}: {stats['classes']} classes, {stats['videos']} videos (avg {stats['avg_videos_per_class']:.1f} videos/class)")
    
    return split_stats

def main():
    if len(sys.argv) < 2:
        print("Usage: python split.py <DATA_ROOT> [OPTIONS]")
        print("\nOptions:")
        print("  --output_dir DIR           Output directory (default: ./filelist)")
        print("  --dataset_name NAME        Dataset name for Kaggle paths (default: auto-detect)")
        print("  --base_ratio FLOAT         Ratio for base classes (default: 0.60)")
        print("  --val_ratio FLOAT          Ratio for val classes (default: 0.20)")
        print("\nExample:")
        print('  python split.py "G:\\TLU\\BigData\\Data_time\\data_down\\kinetics400_mini\\train" --output_dir "./filelist/kinetics400_mini" --dataset_name "kinetics400-mini"')
        return
    
    data_root = sys.argv[1]
    
    # Default values
    output_dir = "./filelist"
    dataset_name = "kinetics400-mini"  # Default Kaggle dataset name
    base_ratio = 0.60
    val_ratio = 0.20
    
    # Parse args
    if "--output_dir" in sys.argv:
        idx = sys.argv.index("--output_dir") + 1
        output_dir = sys.argv[idx]
    
    if "--dataset_name" in sys.argv:
        idx = sys.argv.index("--dataset_name") + 1
        dataset_name = sys.argv[idx]
    
    if "--base_ratio" in sys.argv:
        idx = sys.argv.index("--base_ratio") + 1
        base_ratio = float(sys.argv[idx])
    
    if "--val_ratio" in sys.argv:
        idx = sys.argv.index("--val_ratio") + 1
        val_ratio = float(sys.argv[idx])
    
    # Set random seed for reproducible results
    random.seed(42)
    
    print(f"🔍 Scanning videos in: {data_root}")
    print(f"📦 Kaggle dataset name: {dataset_name}")
    print(f"📊 Split ratio: Base {base_ratio*100:.0f}% / Val {val_ratio*100:.0f}% / Novel {(1-base_ratio-val_ratio)*100:.0f}%")
    
    # Collect videos
    videos_by_label = collect_videos_by_label(data_root)
    print(f"📊 Found {len(videos_by_label)} labels")
    
    total_videos = sum(len(videos) for videos in videos_by_label.values())
    print(f"📼 Total videos: {total_videos}")
    
    # Create base/val/novel split
    print(f"\n🎬 Creating BASE/VAL/NOVEL split...")
    split_stats = create_base_val_novel_split(videos_by_label, output_dir, dataset_name, base_ratio, val_ratio)
    
    print(f"\n💡 Usage with meta_train.py:")
    print(f"   python meta_train.py \\")
    print(f"       --dataset kinetics400 \\")
    print(f"       --data_path {output_dir} \\")
    print(f"       --train_n_episode 300 \\")
    print(f"       --val_n_episode 300 \\")
    print(f"       --n_shot 5 \\")
    print(f"       --num_classes {split_stats['base']['classes']} \\")
    print(f"       --epoch 10")

if __name__ == "__main__":
    main()
