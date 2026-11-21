"""
SiteGuard AI - Dataset Inspection Script
Reads and validates the Roboflow Construction Site Safety dataset.
"""

import os
import yaml
import glob
from collections import Counter
from pathlib import Path

def inspect_dataset(dataset_dir="dataset"):
    dataset_path = Path(dataset_dir)
    data_yaml_path = dataset_path / "data.yaml"

    if not data_yaml_path.exists():
        print(f"[ERROR] data.yaml not found at: {data_yaml_path}")
        return False

    with open(data_yaml_path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    classes = config.get("names", [])
    nc = config.get("nc", len(classes))
    print("=" * 60)
    print("           SITEGUARD AI - DATASET INSPECTION")
    print("=" * 60)
    print(f"Dataset root: {dataset_path.resolve()}")
    print(f"Total defined classes: {nc}")
    print(f"Class names: {classes}\n")

    splits = ["train", "valid", "test"]
    split_stats = {}
    class_counts = {s: Counter() for s in splits}
    corrupted_labels = 0
    total_boxes = 0

    for split in splits:
        img_dir = dataset_path / split / "images"
        lbl_dir = dataset_path / split / "labels"

        images = glob.glob(str(img_dir / "*.*"))
        labels = glob.glob(str(lbl_dir / "*.txt"))

        split_stats[split] = {
            "images": len(images),
            "labels": len(labels)
        }

        for lbl_file in labels:
            try:
                with open(lbl_file, "r", encoding="utf-8") as lf:
                    lines = lf.readlines()
                    for line in lines:
                        line = line.strip()
                        if not line:
                            continue
                        parts = line.split()
                        cls_id = int(parts[0])
                        class_counts[split][cls_id] += 1
                        total_boxes += 1
            except Exception as e:
                corrupted_labels += 1
                print(f"[WARNING] Error reading {lbl_file}: {e}")

    print("--- SPLIT SUMMARY ---")
    for split in splits:
        print(f"  {split.upper():6s} split: {split_stats[split]['images']} images | {split_stats[split]['labels']} label files")
    print(f"  Total images: {sum(s['images'] for s in split_stats.values())}")
    print(f"  Total annotations: {total_boxes}")
    print(f"  Corrupted files: {corrupted_labels}\n")

    print("--- CLASS ID & DISTRIBUTION ---")
    print(f"{'ID':>2s} | {'Class Name':18s} | {'Train':>6s} | {'Valid':>6s} | {'Test':>6s} | {'Total':>6s}")
    print("-" * 52)
    for cid, name in enumerate(classes):
        tr = class_counts["train"][cid]
        va = class_counts["valid"][cid]
        te = class_counts["test"][cid]
        tot = tr + va + te
        print(f"{cid:2d} | {name:18s} | {tr:6d} | {va:6d} | {te:6d} | {tot:6d}")

    # PPE specific availability check
    print("\n--- PPE CLASS AVAILABILITY CHECK ---")
    ppe_targets = {
        "Person": "Worker detection",
        "Hardhat": "Safety helmet",
        "NO-Hardhat": "Missing helmet violation",
        "Safety Vest": "High-visibility vest",
        "NO-Safety Vest": "Missing vest violation",
        "Gloves": "Hand protection",
        "Mask": "Face mask",
        "NO-Mask": "Missing mask",
        "Goggles": "Eye protection"
    }

    for ppe, desc in ppe_targets.items():
        found = ppe in classes
        if found:
            idx = classes.index(ppe)
            cnt = sum(class_counts[s][idx] for s in splits)
            print(f"  [SUPPORTED] {ppe:15s} (Class ID: {idx:2d}) - {desc} - {cnt} instances")
        else:
            print(f"  [NOT IN DATASET] {ppe:15s} - {desc} - NOT SUPPORTED")

    return True

if __name__ == "__main__":
    import sys
    base_dir = Path(__file__).resolve().parent.parent / "dataset"
    inspect_dataset(base_dir)
