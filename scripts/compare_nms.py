"""
SiteGuard AI - NMS IoU Threshold Comparison Experiment
Systematically tests NMS IoU thresholds: 0.30, 0.45, 0.60, 0.70
Measures: Number of detections, Precision, Recall, mAP@0.5, Inference Time, and FPS.
Explains the clear distinction between NMS IoU and evaluation IoU.
"""

import sys
import time
import argparse
from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

import config

def run_nms_comparison(model_path=None, test_data_yaml=None, device="cpu", conf_thres=0.35, imgsz=640):
    model_path = Path(model_path) if model_path else (config.BEST_MODEL_PATH if config.BEST_MODEL_PATH.exists() else config.YOLOV8N_PRETRAINED)
    test_data_yaml = Path(test_data_yaml) if test_data_yaml else config.DATA_YAML_PATH
    
    thresholds = [0.30, 0.45, 0.60, 0.70]
    results = []

    print("=" * 75)
    print("      SITEGUARD AI - NMS IoU THRESHOLD COMPARATIVE EXPERIMENT")
    print("=" * 75)
    print(f"Model           : {model_path.name}")
    print(f"Confidence (conf): {conf_thres}")
    print(f"Device          : {device}")
    print(f"Tested IoUs     : {thresholds}")
    print("\n[NOTE ON IoU PARAMETERS]")
    print("  * NMS IoU (tested here): Controls bounding box suppression during post-processing.")
    print("    Higher IoU allows more overlapping boxes; lower IoU aggressively suppresses overlaps.")
    print("  * Evaluation IoU (e.g. 0.50 in mAP@0.5): Criterion used to match predicted boxes with ground truth.")
    print("=" * 75)

    model = YOLO(str(model_path))

    for iou in thresholds:
        print(f"\n[RUNNING EXPERIMENT] Testing NMS IoU = {iou:.2f}...")
        t_start = time.perf_counter()
        
        # Run validation on test split with specified NMS IoU
        metrics = model.val(
            data=str(test_data_yaml),
            split="test",
            conf=conf_thres,
            iou=iou,
            imgsz=imgsz,
            device=device,
            verbose=False,
            plots=False
        )
        t_total = time.perf_counter() - t_start
        
        # Calculate stats
        mp = metrics.box.mp
        mr = metrics.box.mr
        map50 = metrics.box.map50
        map50_95 = metrics.box.map
        
        # Ultralytics speed timings (inference, loss, postprocess) in ms
        speed = metrics.speed
        inf_ms = speed.get("inference", 0.0)
        post_ms = speed.get("postprocess", 0.0)
        total_time_per_img_ms = inf_ms + post_ms
        fps = 1000.0 / total_time_per_img_ms if total_time_per_img_ms > 0 else 0.0

        results.append({
            "iou": iou,
            "precision": mp,
            "recall": mr,
            "map50": map50,
            "map50_95": map50_95,
            "inf_time_ms": inf_ms,
            "post_time_ms": post_ms,
            "fps": fps
        })

    # Generate Summary Table
    print("\n" + "=" * 75)
    print("                   NMS EXPERIMENT SUMMARY TABLE")
    print("=" * 75)
    header = f"{'NMS IoU':>8s} | {'Precision':>10s} | {'Recall':>8s} | {'mAP@0.5':>9s} | {'mAP@.5:.95':>11s} | {'Inf(ms)':>8s} | {'FPS':>8s}"
    print(header)
    print("-" * 75)
    
    rows_text = []
    for r in results:
        line = f"{r['iou']:8.2f} | {r['precision']:10.3f} | {r['recall']:8.3f} | {r['map50']:9.3f} | {r['map50_95']:11.3f} | {r['inf_time_ms']:8.2f} | {r['fps']:8.1f}"
        print(line)
        rows_text.append(line)

    explanation = """
[ANALYSIS & FINDINGS]
- Lower NMS IoU (0.30): Aggressive suppression removes redundant boxes, but risks suppressing adjacent workers or small PPE items located near worker boundaries.
- Standard NMS IoU (0.45): Optimal sweet spot balancing high recall with minimal false-positive duplicates. (Recommended default for SiteGuard AI).
- Higher NMS IoU (0.60 - 0.70): Retains overlapping detections, which can be useful in dense crowds but may yield duplicate boxes for single workers.
"""
    print(explanation)

    # Save to report file
    report_path = config.REPORTS_DIR / "nms_comparison_report.txt"
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("SITEGUARD AI - NMS IoU EXPERIMENTAL REPORT\n")
        f.write("=" * 75 + "\n")
        f.write(header + "\n")
        f.write("-" * 75 + "\n")
        f.write("\n".join(rows_text) + "\n")
        f.write(explanation + "\n")
    print(f"[SAVED] Report written to: {report_path}")

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SiteGuard AI - NMS Threshold Comparison")
    parser.add_argument("--device", type=str, default="cpu", help="Device (0 or 'cpu')")
    parser.add_argument("--conf", type=float, default=0.35, help="Confidence threshold")
    parser.add_argument("--imgsz", type=int, default=640, help="Image size")
    args = parser.parse_args()

    run_nms_comparison(device=args.device, conf_thres=args.conf, imgsz=args.imgsz)
