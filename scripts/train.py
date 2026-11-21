"""
SiteGuard AI - Training & Fine-Tuning Pipeline
Trains YOLOv8-nano on the Roboflow Construction Site Safety dataset.
Supports CUDA GPU (RTX 3050 Laptop) and CPU fallback.
Saves best.pt, last.pt, training plots, and confusion matrix.
"""

import os
import sys
import shutil
import argparse
from pathlib import Path
import torch
from ultralytics import YOLO

# Add parent directory to sys.path
BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

import config

def train_model(
    data_yaml=None,
    epochs=config.TRAIN_EPOCHS,
    batch=config.BATCH_SIZE,
    imgsz=config.INFERENCE_IMG_SIZE,
    device=None,
    weights=None,
    patience=10,
    project_name="siteguard_train",
    name="yolov8n_ppe"
):
    print("=" * 65)
    print("           SITEGUARD AI - YOLOv8 TRAINING PIPELINE")
    print("=" * 65)
    
    # Resolve Paths
    data_yaml = Path(data_yaml) if data_yaml else config.DATA_YAML_PATH
    if not data_yaml.exists():
        raise FileNotFoundError(f"data.yaml not found at: {data_yaml}")
        
    weights_path = Path(weights) if weights else config.YOLOV8N_PRETRAINED
    if not weights_path.exists():
        print(f"Pretrained weights {weights_path} not found locally, will download yolov8n.pt...")
        weights_path = "yolov8n.pt"

    # Auto-detect hardware
    if device is None:
        if torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0)
            print(f"[HARDWARE] CUDA GPU detected: {gpu_name}")
            device = 0
        else:
            print("[HARDWARE] CUDA not available. Falling back to multi-core CPU.")
            device = "cpu"
            # Adjust batch size for CPU stability if needed
            if batch > 8:
                batch = 8
                print(f"[CONFIG] Adjusted CPU batch size to {batch} for responsive training.")

    print(f"[CONFIG] Data YAML    : {data_yaml}")
    print(f"[CONFIG] Base Weights : {weights_path}")
    print(f"[CONFIG] Epochs       : {epochs}")
    print(f"[CONFIG] Batch Size   : {batch}")
    print(f"[CONFIG] Image Size   : {imgsz}")
    print(f"[CONFIG] Device       : {device}")
    print(f"[CONFIG] Early Stop   : patience={patience}")

    # Load model for transfer learning
    model = YOLO(str(weights_path))

    # Output directory for Ultralytics run
    runs_dir = BASE_DIR / "runs" / "train"
    runs_dir.mkdir(parents=True, exist_ok=True)

    print("\n[TRAINING] Starting training session...")
    results = model.train(
        data=str(data_yaml),
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        device=device,
        patience=patience,
        save=True,
        workers=2,
        optimizer="auto",
        project=str(runs_dir),
        name=name,
        exist_ok=True,
        verbose=True,
        plots=True
    )

    # Locate saved weights
    train_output_dir = runs_dir / name
    best_pt = train_output_dir / "weights" / "best.pt"
    last_pt = train_output_dir / "weights" / "last.pt"

    if best_pt.exists():
        config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
        target_best = config.BEST_MODEL_PATH
        target_last = config.LAST_MODEL_PATH
        shutil.copy2(best_pt, target_best)
        print(f"\n[SUCCESS] Best model weights saved to: {target_best}")
        if last_pt.exists():
            shutil.copy2(last_pt, target_last)
            print(f"[SUCCESS] Last model weights saved to: {target_last}")

        # Copy evaluation curves and matrices to reports/
        reports_dir = config.REPORTS_DIR
        reports_dir.mkdir(parents=True, exist_ok=True)
        report_files = [
            "confusion_matrix.png",
            "confusion_matrix_normalized.png",
            "results.png",
            "F1_curve.png",
            "PR_curve.png",
            "P_curve.png",
            "R_curve.png"
        ]
        for rf in report_files:
            src = train_output_dir / rf
            if src.exists():
                shutil.copy2(src, reports_dir / rf)
                print(f"[REPORT] Saved {rf} to {reports_dir / rf}")

        print("\n" + "=" * 65)
        print("           TRAINING & EXPORT COMPLETED SUCCESSFULLY!")
        print("=" * 65)
        return str(target_best)
    else:
        print("\n[ERROR] Training completed but best.pt was not found at expected location.")
        return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SiteGuard AI - YOLOv8 Training Script")
    parser.add_argument("--epochs", type=int, default=config.TRAIN_EPOCHS, help="Number of epochs")
    parser.add_argument("--batch", type=int, default=config.BATCH_SIZE, help="Batch size")
    parser.add_argument("--imgsz", type=int, default=config.INFERENCE_IMG_SIZE, help="Image size")
    parser.add_argument("--device", type=str, default=None, help="Device (0 for GPU, 'cpu' for CPU)")
    parser.add_argument("--weights", type=str, default=None, help="Path to initial weights")
    parser.add_argument("--data", type=str, default=None, help="Path to data.yaml")
    parser.add_argument("--patience", type=int, default=10, help="Early stopping patience")
    args = parser.parse_args()

    dev = int(args.device) if args.device and args.device.isdigit() else args.device
    train_model(
        data_yaml=args.data,
        epochs=args.epochs,
        batch=args.batch,
        imgsz=args.imgsz,
        device=dev,
        weights=args.weights,
        patience=args.patience
    )
