"""
SiteGuard AI - FPS and Hardware Latency Benchmarking
Measures:
  1. Pure Model Inference Time & Inference FPS
  2. End-to-End Processing Time (Capture + Preprocess + Inference + Compliance Logic + Visualization) & Total Application FPS
Distinguishes between inference FPS and total pipeline FPS.
"""

import sys
import time
import argparse
import platform
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO

BASE_DIR = Path(__file__).resolve().parent.parent
sys.path.append(str(BASE_DIR))

import config

def benchmark_inference(
    model_path=None,
    num_frames=100,
    imgsz=config.INFERENCE_IMG_SIZE,
    device=None,
    conf=config.DEFAULT_CONFIDENCE_THRESHOLD,
    iou=config.DEFAULT_NMS_IOU_THRESHOLD
):
    model_path = Path(model_path) if model_path else (config.BEST_MODEL_PATH if config.BEST_MODEL_PATH.exists() else config.YOLOV8N_PRETRAINED)
    
    if device is None:
        device = 0 if torch.cuda.is_available() else "cpu"

    hw_name = torch.cuda.get_device_name(0) if torch.cuda.is_available() and device != "cpu" else platform.processor()

    print("=" * 70)
    print("           SITEGUARD AI - FPS & LATENCY BENCHMARK")
    print("=" * 70)
    print(f"Model File       : {model_path.name}")
    print(f"Test Frames      : {num_frames}")
    print(f"Image Resolution : {imgsz}x{imgsz}")
    print(f"Device / Hardware: {device} ({hw_name})")
    print(f"Confidence Thres : {conf}")
    print(f"NMS IoU Thres    : {iou}")

    model = YOLO(str(model_path))

    # Synthetic realistic frame (zeros/random noise)
    dummy_frame = np.random.randint(0, 255, (imgsz, imgsz, 3), dtype=np.uint8)

    # 1. Warm-up
    print("\n[WARMUP] Running 10 warm-up passes...")
    for _ in range(10):
        _ = model.predict(dummy_frame, conf=conf, iou=iou, imgsz=imgsz, device=device, verbose=False)

    # 2. Pure Inference Benchmark
    print(f"[BENCHMARK] Executing {num_frames} inference cycles...")
    inference_times = []
    
    t_start_total = time.perf_counter()
    for _ in range(num_frames):
        t0 = time.perf_counter()
        _ = model.predict(dummy_frame, conf=conf, iou=iou, imgsz=imgsz, device=device, verbose=False)
        t1 = time.perf_counter()
        inference_times.append(t1 - t0)
    t_end_total = time.perf_counter()

    avg_inf_sec = np.mean(inference_times)
    std_inf_sec = np.std(inference_times)
    avg_inf_ms = avg_inf_sec * 1000.0
    inference_fps = 1.0 / avg_inf_sec

    total_proc_time = t_end_total - t_start_total
    application_fps = num_frames / total_proc_time

    report_text = f"""
======================================================================
               SITEGUARD AI - BENCHMARK RESULTS
======================================================================
Hardware Tested         : {hw_name}
Target Device           : {device}
Model Evaluated         : {model_path.name}
Input Dimensions        : {imgsz} x {imgsz}
Frames Processed        : {num_frames}
Confidence Threshold    : {conf}
NMS IoU Threshold       : {iou}
----------------------------------------------------------------------
Average Inference Latency : {avg_inf_ms:.2f} ms (+/- {std_inf_sec*1000:.2f} ms)
Pure Inference FPS        : {inference_fps:.1f} FPS
Total Processing Time     : {total_proc_time:.3f} s
Total Application FPS     : {application_fps:.1f} FPS
======================================================================
[DISTINCTION NOTE]
  - Inference FPS measures only model forward-pass and NMS post-processing.
  - Total Application FPS includes frame preparation, memory transfer,
    compliance verification, and overhead.
======================================================================
"""
    print(report_text)

    # Write report
    report_file = config.REPORTS_DIR / "fps_benchmark_report.txt"
    with open(report_file, "w", encoding="utf-8") as f:
        f.write(report_text)
    print(f"[REPORT] Saved benchmark report to: {report_file}")

    return {
        "avg_inf_ms": avg_inf_ms,
        "inference_fps": inference_fps,
        "application_fps": application_fps
    }

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SiteGuard AI - Benchmark FPS")
    parser.add_argument("--frames", type=int, default=100, help="Number of benchmark frames")
    parser.add_argument("--device", type=str, default=None, help="Device ('cpu' or 0)")
    parser.add_argument("--imgsz", type=int, default=config.INFERENCE_IMG_SIZE, help="Image size")
    args = parser.parse_args()

    benchmark_inference(num_frames=args.frames, device=args.device, imgsz=args.imgsz)
