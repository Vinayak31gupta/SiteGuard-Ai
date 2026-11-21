"""
SiteGuard AI - Sample Video Generator
Generates a realistic 10-second test video from dataset test images
so users can immediately test the Upload Video feature out-of-the-box.
"""

import os
import glob
import cv2
import numpy as np
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
TEST_IMG_DIR = BASE_DIR / "dataset" / "test" / "images"
OUTPUT_VIDEO_PATH = BASE_DIR / "uploads" / "sample_construction_site.mp4"

def create_sample_video():
    images = glob.glob(str(TEST_IMG_DIR / "*.jpg")) + glob.glob(str(TEST_IMG_DIR / "*.png"))
    if not images:
        print("[ERROR] No test images found.")
        return

    # Select 6 varied test images
    selected = images[:8]
    print(f"Creating sample video from {len(selected)} test scene images...")

    target_w, target_h = 640, 480
    fps = 20
    frames_per_img = 25  # ~1.25 seconds per image

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    OUTPUT_VIDEO_PATH.parent.mkdir(parents=True, exist_ok=True)
    out = cv2.VideoWriter(str(OUTPUT_VIDEO_PATH), fourcc, fps, (target_w, target_h))

    for img_path in selected:
        img = cv2.imread(img_path)
        if img is None:
            continue
        resized = cv2.resize(img, (target_w, target_h))

        # Create subtle smooth pan/zoom effect across frames
        for f in range(frames_per_img):
            # Subtle zoom
            scale = 1.0 + 0.05 * (f / frames_per_img)
            h, w = resized.shape[:2]
            nw, nh = int(w * scale), int(h * scale)
            zoomed = cv2.resize(resized, (nw, nh))
            # Crop center
            cx, cy = nw // 2, nh // 2
            x1 = cx - target_w // 2
            y1 = cy - target_h // 2
            frame = zoomed[y1:y1 + target_h, x1:x1 + target_w]
            out.write(frame)

    out.release()
    print(f"[SUCCESS] Sample video generated at: {OUTPUT_VIDEO_PATH}")

if __name__ == "__main__":
    create_sample_video()
