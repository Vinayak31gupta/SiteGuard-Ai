"""
SiteGuard AI - Model Evaluation Script
Evaluates model on the validation or test split of the dataset.
Calculates Precision, Recall, mAP@0.5, mAP@0.5:0.95, per-class metrics, and generates confusion matrix.
"""

import sys
import argparse
from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

import config

def evaluate_model(model_path=None, data_yaml=None, split="test", imgsz=640, device="cpu", save_json=True):
    model_path = Path(model_path) if model_path else config.BEST_MODEL_PATH
    if not model_path.exists():
        if config.YOLOV8N_PRETRAINED.exists():
            print(f"[NOTICE] {model_path} not found. Falling back to baseline {config.YOLOV8N_PRETRAINED}")
            model_path = config.YOLOV8N_PRETRAINED
        else:
            raise FileNotFoundError(f"No model found at {model_path}")

    data_yaml = Path(data_yaml) if data_yaml else config.DATA_YAML_PATH
    if not data_yaml.exists():
        raise FileNotFoundError(f"Dataset config not found at {data_yaml}")

    print("=" * 65)
    print("           SITEGUARD AI - MODEL EVALUATION")
    print("=" * 65)
    print(f"Model Path:  {model_path}")
    print(f"Data YAML:   {data_yaml}")
    print(f"Split:       {split}")
    print(f"Device:      {device}")
    print(f"Image Size:  {imgsz}")

    model = YOLO(str(model_path))
    metrics = model.val(
        data=str(data_yaml),
        split=split,
        imgsz=imgsz,
        device=device,
        plots=True,
        save_json=save_json,
        project=str(config.REPORTS_DIR / "eval"),
        name=f"eval_{split}",
        exist_ok=True
    )

    # Extract metrics
    mp = metrics.box.mp   # Mean Precision
    mr = metrics.box.mr   # Mean Recall
    map50 = metrics.box.map50  # mAP@0.5
    map50_95 = metrics.box.map  # mAP@0.5:0.95

    report_lines = [
        "=" * 65,
        "              SITEGUARD AI - EVALUATION REPORT",
        "=" * 65,
        f"Evaluation Split : {split}",
        f"Evaluated Model  : {model_path.name}",
        f"Precision (P)    : {mp:.4f} ({mp*100:.1f}%)",
        f"Recall (R)       : {mr:.4f} ({mr*100:.1f}%)",
        f"mAP@0.5          : {map50:.4f} ({map50*100:.1f}%)",
        f"mAP@0.5:0.95     : {map50_95:.4f} ({map50_95*100:.1f}%)",
        "-" * 65,
        f"{'Class ID':8s} | {'Class Name':18s} | {'P':>6s} | {'R':>6s} | {'mAP@0.5':>8s}",
        "-" * 65
    ]

    names = model.names
    # Per-class metrics
    for i, name in names.items():
        if i < len(metrics.box.p):
            cp = metrics.box.p[i]
            cr = metrics.box.r[i]
            cmap50 = metrics.box.ap50[i]
            report_lines.append(f"{i:8d} | {name:18s} | {cp:6.3f} | {cr:6.3f} | {cmap50:8.3f}")

    report_text = "\n".join(report_lines)
    print("\n" + report_text)

    # Save report to text file
    eval_txt = config.REPORTS_DIR / f"evaluation_report_{split}.txt"
    with open(eval_txt, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"\n[REPORT] Saved evaluation report to: {eval_txt}")

    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SiteGuard AI - Evaluation Script")
    parser.add_argument("--model", type=str, default=None, help="Path to weights file")
    parser.add_argument("--split", type=str, default="test", choices=["val", "test", "train"], help="Dataset split")
    parser.add_argument("--device", type=str, default="cpu", help="Device (0 or 'cpu')")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    args = parser.parse_args()

    evaluate_model(
        model_path=args.model,
        split=args.split,
        device=args.device,
        imgsz=args.imgsz
    )
