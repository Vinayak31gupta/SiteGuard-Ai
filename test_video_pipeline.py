"""
Test script to verify video processing pipeline end-to-end.
"""

import sys
import time
from pathlib import Path
import cv2

BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

import config
from database import record_event, record_violation, record_video_history, init_db
from detection import SiteGuardDetector

def test_pipeline():
    init_db()
    input_video = BASE_DIR / "uploads" / "sample_construction_site.mp4"
    output_video = BASE_DIR / "outputs" / "test_processed_sample.mp4"

    if not input_video.exists():
        print(f"[ERROR] Sample video not found at: {input_video}")
        return

    print("=" * 65)
    print("      TESTING SITEGUARD AI - VIDEO PROCESSING PIPELINE")
    print("=" * 65)

    det = SiteGuardDetector(
        conf_thres=0.35,
        iou_thres=0.45,
        device="cpu",
        enable_tracking=True
    )

    cap = cv2.VideoCapture(str(input_video))
    fps = cap.get(cv2.CAP_PROP_FPS) or 20.0
    w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_video), fourcc, fps, (w, h))

    frames_processed = 0
    t0 = time.perf_counter()

    print(f"Input: {input_video} ({w}x{h}, {total_frames} frames @ {fps} FPS)")

    total_workers_sum = 0
    total_violations_count = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        frames_processed += 1
        annotated, stats = det.process_frame(frame)
        out.write(annotated)

        total_workers_sum += stats['total_workers']

        # Log violations to SQLite
        for v in stats['violations']:
            logged_id = record_violation(
                missing_ppe=v['missing_ppe'],
                confidence=v['confidence'],
                status=v['status'],
                worker_id=v['worker_id'],
                source_type="upload"
            )
            if logged_id:
                total_violations_count += 1

        if frames_processed % 30 == 0 or frames_processed == total_frames:
            print(f"  Frame {frames_processed}/{total_frames} | FPS: {stats['fps']} | Workers: {stats['total_workers']} | Violations: {stats['non_compliant']}")

    cap.release()
    out.release()
    t_total = time.perf_counter() - t0
    avg_fps = frames_processed / max(0.001, t_total)

    print(f"\n[SUCCESS] Processed {frames_processed} frames in {t_total:.2f}s ({avg_fps:.1f} FPS)")
    print(f"[OUTPUT] Saved annotated video to: {output_video}")
    print(f"[STATS] Total violations logged: {total_violations_count}")

    # Log to video history
    record_video_history(
        original_filename="sample_construction_site.mp4",
        output_filename="test_processed_sample.mp4",
        total_frames=frames_processed,
        processed_fps=round(avg_fps, 1),
        workers_detected=int(total_workers_sum / max(1, frames_processed)),
        violations_detected=total_violations_count
    )

if __name__ == "__main__":
    test_pipeline()
