"""
Visualize PSG Dataset Annotations - Hiển thị Relation Triplets

Usage:
    conda activate sgg_benchmark && python visualize_psg_annotations.py
"""
import json
import random

print("=" * 80)
print("PSG DATASET - RELATION TRIPLETS VISUALIZATION")
print("=" * 80)

# Load annotation file
ann_file = 'datasets/psg/psg/psg_train_val.json'
print(f"\n[1] Loading: {ann_file}")

with open(ann_file, 'r') as f:
    dataset = json.load(f)

print(f"✓ Loaded successfully!")

# Load object classes
with open('datasets/psg/obj_classes.txt', 'r') as f:
    obj_classes = ['__background__'] + f.read().splitlines()

# Load predicates
predicates = ['__background__'] + dataset['predicate_classes']

print(f"\n[2] Dataset Statistics:")
print(f"  - Total images: {len(dataset['data'])}")
print(f"  - Object classes: {len(obj_classes)}")
print(f"  - Predicate classes: {len(predicates)}")
print(f"  - Test images: {len(dataset['test_image_ids'])}")

# Calculate statistics
total_relations = 0
total_objects = 0
images_with_relations = 0

for data_entry in dataset['data']:
    total_relations += len(data_entry['relations'])
    total_objects += len(data_entry['segments_info'])
    if len(data_entry['relations']) > 0:
        images_with_relations += 1

print(f"\n[3] Relation Statistics:")
print(f"  - Total relations: {total_relations:,}")
print(f"  - Total objects: {total_objects:,}")
print(f"  - Images with relations: {images_with_relations:,} ({images_with_relations/len(dataset['data'])*100:.1f}%)")
print(f"  - Avg relations per image: {total_relations/len(dataset['data']):.2f}")
print(f"  - Avg objects per image: {total_objects/len(dataset['data']):.2f}")

print("\n" + "=" * 80)
print("RELATION TRIPLET FORMAT")
print("=" * 80)

print("""
Trong PSG annotation file, mỗi image có:

1. "segments_info": List of objects
   └── Mỗi segment có: id, category_id, area, iscrowd, isthing
   └── Index trong list này là object_idx (0, 1, 2, ...)

2. "relations": List of relation triplets
   └── Format: [subject_idx, object_idx, predicate_id]
   └── subject_idx, object_idx: Index trong segments_info
   └── predicate_id: Index trong predicate_classes (1-indexed)

VÍ DỤ:
  segments_info = [
    {category_id: 0},  # idx=0 → person
    {category_id: 1},  # idx=1 → bicycle  
    {category_id: 101} # idx=2 → road
  ]
  
  relations = [
    [0, 1, 47],  # person (idx=0) --[riding(47)]--> bicycle (idx=1)
    [1, 2, 4],   # bicycle (idx=1) --[on(4)]--> road (idx=2)
  ]
""")

print("\n" + "=" * 80)
print("DETAILED EXAMPLES")
print("=" * 80)

# Find some interesting examples
examples_to_show = 3
shown = 0

for data_entry in dataset['data']:
    if len(data_entry['relations']) >= 3 and shown < examples_to_show:
        print(f"\n{'─'*80}")
        print(f"Image: {data_entry['file_name']}")
        print(f"Size: {data_entry['width']} x {data_entry['height']}")
        print(f"Image ID: {data_entry['image_id']}")
        print(f"{'─'*80}")
        
        # Show objects
        print(f"\nObjects ({len(data_entry['segments_info'])} total):")
        for idx, seg in enumerate(data_entry['segments_info']):
            cat_id = seg['category_id']
            obj_name = obj_classes[cat_id] if cat_id < len(obj_classes) else f"unknown_{cat_id}"
            area = seg.get('area', 0)
            is_thing = seg.get('isthing', False)
            obj_type = "thing" if is_thing else "stuff"
            print(f"  [{idx}] {obj_name:<20} (cat_id={cat_id}, area={area:>6}, type={obj_type})")
        
        # Show relations
        print(f"\nRelations ({len(data_entry['relations'])} triplets):")
        for i, (sub_idx, obj_idx, pred_id) in enumerate(data_entry['relations'], 1):
            # Get object names
            sub_cat = data_entry['segments_info'][sub_idx]['category_id']
            obj_cat = data_entry['segments_info'][obj_idx]['category_id']
            
            sub_name = obj_classes[sub_cat] if sub_cat < len(obj_classes) else f"obj_{sub_cat}"
            obj_name = obj_classes[obj_cat] if obj_cat < len(obj_classes) else f"obj_{obj_cat}"
            pred_name = predicates[pred_id] if pred_id < len(predicates) else f"pred_{pred_id}"
            
            print(f"  {i:2d}. [{sub_idx}] {sub_name:<20} --[{pred_id:2d}:{pred_name:<20}]--> [{obj_idx}] {obj_name:<20}")
        
        shown += 1

print("\n" + "=" * 80)
print("PREDICATE DISTRIBUTION")
print("=" * 80)

# Count predicate frequency
pred_counts = {}
for data_entry in dataset['data']:
    for sub_idx, obj_idx, pred_id in data_entry['relations']:
        pred_counts[pred_id] = pred_counts.get(pred_id, 0) + 1

# Sort by frequency
sorted_preds = sorted(pred_counts.items(), key=lambda x: x[1], reverse=True)

print(f"\nTop 20 Most Common Predicates:")
print(f"{'Rank':<6} {'ID':<6} {'Predicate':<25} {'Count':<12} {'Percentage':<10}")
print("-" * 80)

total = sum(pred_counts.values())
for rank, (pred_id, count) in enumerate(sorted_preds[:20], 1):
    pred_name = predicates[pred_id] if pred_id < len(predicates) else f"pred_{pred_id}"
    percentage = (count / total) * 100
    print(f"{rank:<6} {pred_id:<6} {pred_name:<25} {count:<12,} {percentage:>6.2f}%")

print(f"\nBottom 10 Rarest Predicates:")
print(f"{'Rank':<6} {'ID':<6} {'Predicate':<25} {'Count':<12} {'Percentage':<10}")
print("-" * 80)

for rank, (pred_id, count) in enumerate(sorted_preds[-10:], 1):
    pred_name = predicates[pred_id] if pred_id < len(predicates) else f"pred_{pred_id}"
    percentage = (count / total) * 100
    print(f"{rank:<6} {pred_id:<6} {pred_name:<25} {count:<12,} {percentage:>6.2f}%")

print("\n" + "=" * 80)
print("OBJECT PAIR DISTRIBUTION")
print("=" * 80)

# Sample some common object pairs
pair_counts = {}
for data_entry in dataset['data'][:5000]:  # Sample first 5000 images
    for sub_idx, obj_idx, pred_id in data_entry['relations']:
        sub_cat = data_entry['segments_info'][sub_idx]['category_id']
        obj_cat = data_entry['segments_info'][obj_idx]['category_id']
        
        sub_name = obj_classes[sub_cat] if sub_cat < len(obj_classes) else f"obj_{sub_cat}"
        obj_name = obj_classes[obj_cat] if obj_cat < len(obj_classes) else f"obj_{obj_cat}"
        
        pair = (sub_name, obj_name)
        pair_counts[pair] = pair_counts.get(pair, 0) + 1

sorted_pairs = sorted(pair_counts.items(), key=lambda x: x[1], reverse=True)

print(f"\nTop 15 Most Common Object Pairs (from 5000 sample images):")
print(f"{'Rank':<6} {'Subject':<20} {'Object':<20} {'Count':<8}")
print("-" * 80)

for rank, ((sub, obj), count) in enumerate(sorted_pairs[:15], 1):
    print(f"{rank:<6} {sub:<20} {obj:<20} {count:<8,}")

print("\n" + "=" * 80)
print("HOW TO USE THESE TRIPLETS IN TRAINING")
print("=" * 80)

print("""
TRAINING PROCESS:

1. Load Image + Annotations
   ├── Image: datasets/psg/coco/{file_name}
   ├── Objects: segments_info → bounding boxes + class labels
   └── Relations: relation triplets → ground truth relationships

2. Data Processing (trong PSGDataset.__getitem__)
   ├── Read image
   ├── Extract bounding boxes từ segments_info
   ├── Convert relations thành target format
   └── Apply transformations

3. Forward Pass
   ├── YOLO backbone → detect objects
   ├── Generate object pairs
   └── Relation head → predict predicates

4. Loss Calculation
   ├── Match predictions với ground truth triplets
   ├── Calculate cross-entropy loss
   └── Backpropagation

5. Metrics
   ├── Recall@K: Top-K predictions chứa GT relation
   ├── Mean Recall: Average recall across all predicate classes
   └── Zero-shot/Few-shot metrics

KEY FILES:
─────────────────────────────────────────────────────────────
- Annotation: datasets/psg/psg/psg_train_val.json
- Dataset code: sgg_benchmark/data/datasets/psg.py
- Training: tools/relation_train_net.py
- Config: configs/PSG/react_yolov8m.yaml

ACCESSING TRIPLETS IN CODE:
─────────────────────────────────────────────────────────────
from sgg_benchmark.data.datasets.psg import PSGDataset

dataset = PSGDataset(
    split='train',
    img_dir='datasets/psg/coco/',
    ann_file='datasets/psg/psg/psg_train_val.json',
)

# Get sample
img, target, idx = dataset[0]

# Target contains:
# - bbox: bounding boxes
# - labels: object classes  
# - relation: relation matrix or triplets
""")

print("\n" + "=" * 80)
print("✓ VISUALIZATION COMPLETE!")
print("=" * 80)

print(f"""
Summary:
  - PSG dataset có {total_relations:,} relation triplets
  - Relations được lưu dạng [subject_idx, object_idx, predicate_id]
  - subject_idx, object_idx là index trong segments_info
  - predicate_id là index trong predicate_classes (1-indexed)
  - Model học mapping từ visual features → predicate class

Files created:
  ✓ psg_predicates.json - Full predicate info
  ✓ psg_predicates_list.json - Simple predicate list
  ✓ This visualization script
""")
