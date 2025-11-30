"""
Script để extract PSG predicate labels từ annotation file

Usage:
    conda activate sgg_benchmark && python extract_psg_predicates.py
"""
import json
import os

print("=" * 80)
print("EXTRACTING PSG PREDICATE LABELS")
print("=" * 80)

# Path to PSG annotation file
ann_file = "datasets/psg/psg/psg_train_val.json"

if not os.path.exists(ann_file):
    print(f"❌ File not found: {ann_file}")
    print("\nTrying alternative locations...")
    
    # Try alternative paths
    alternatives = [
        "datasets/psg/psg/psg.json",
        "datasets/psg/psg/tiny_psg.json",
        "datasets/psg/psg/psg_val_test.json"
    ]
    
    for alt in alternatives:
        if os.path.exists(alt):
            ann_file = alt
            print(f"✓ Found: {alt}")
            break
    else:
        print("❌ No PSG annotation file found!")
        print("\nPlease check the following paths:")
        for p in [ann_file] + alternatives:
            print(f"  - {p}")
        exit(1)

print(f"\n[1] Loading annotation file: {ann_file}")
with open(ann_file, 'r') as f:
    dataset = json.load(f)

print("✓ File loaded successfully")

# Extract predicate classes
if 'predicate_classes' in dataset:
    predicates = dataset['predicate_classes']
    print(f"\n[2] Found {len(predicates)} predicate classes")
    
    # Create ind_to_predicates mapping (1-indexed for predicates, 0 for background)
    predicate_to_idx = {label: idx+1 for idx, label in enumerate(predicates)}
    predicate_to_idx['__background__'] = 0
    
    ind_to_predicates = sorted(predicate_to_idx, key=lambda k: predicate_to_idx[k])
    
    print("\n[3] PSG Predicate Classes:")
    print("-" * 80)
    for idx, pred in enumerate(ind_to_predicates):
        print(f"  {idx:3d}: {pred}")
    
    # Save to JSON file
    output_file = "psg_predicates.json"
    output_data = {
        "predicates": ind_to_predicates,
        "num_predicates": len(ind_to_predicates),
        "predicate_to_idx": predicate_to_idx,
        "note": "Index 0 is __background__, actual predicates start from index 1"
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\n[4] ✓ Saved predicates to: {output_file}")
    
    # Save a clean list version
    output_file_list = "psg_predicates_list.json"
    with open(output_file_list, 'w') as f:
        json.dump(ind_to_predicates, f, indent=2)
    
    print(f"[5] ✓ Saved clean list to: {output_file_list}")
    
    # Print summary
    print("\n" + "=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"Total predicate classes: {len(ind_to_predicates)}")
    print(f"  - Background: 1 class")
    print(f"  - Actual predicates: {len(ind_to_predicates) - 1} classes")
    print(f"\nFirst 10 predicates:")
    for i in range(min(10, len(ind_to_predicates))):
        print(f"  {i}: {ind_to_predicates[i]}")
    
    # Check config file
    config_file = "checkpoints/react_PSG/config.yml"
    if os.path.exists(config_file):
        print(f"\n[6] Verifying with config file: {config_file}")
        import yaml
        with open(config_file, 'r') as f:
            cfg = yaml.safe_load(f)
        
        num_classes_config = cfg['MODEL']['ROI_RELATION_HEAD']['NUM_CLASSES']
        print(f"  - NUM_CLASSES in config: {num_classes_config}")
        print(f"  - Predicates extracted: {len(ind_to_predicates)}")
        
        if num_classes_config == len(ind_to_predicates):
            print("  ✓ Numbers match!")
        else:
            print(f"  ⚠ Warning: Mismatch! Config has {num_classes_config} but found {len(ind_to_predicates)} predicates")
    
    print("\n" + "=" * 80)
    print("✓ EXTRACTION COMPLETE!")
    print("=" * 80)
    print(f"\nYou can now use these predicate labels in your demo:")
    print(f"  - {output_file}")
    print(f"  - {output_file_list}")
    
else:
    print("❌ 'predicate_classes' not found in dataset")
    print("Available keys:", list(dataset.keys()))
