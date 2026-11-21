"""
SiteGuard AI - NMS and Inference Module
Handles single-pass Non-Maximum Suppression (NMS) and confidence filtering using Ultralytics YOLOv8 inference.
Note: Evaluates detections without redundant double-NMS.
"""

import time
import torch
from ultralytics import YOLO
import config

class NMSInferencer:
    """
    Standardized inference wrapper with configurable IoU and confidence thresholds.
    """
    def __init__(self, model_path=None, conf_thres=config.DEFAULT_CONFIDENCE_THRESHOLD, iou_thres=config.DEFAULT_NMS_IOU_THRESHOLD, device="cpu"):
        self.model_path = model_path or (config.BEST_MODEL_PATH if config.BEST_MODEL_PATH.exists() else config.YOLOV8N_PRETRAINED)
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = device
        
        # Load YOLO model
        self.model = YOLO(str(self.model_path))
        print(f"[INFERENCER] Loaded {self.model_path.name} | Conf: {self.conf_thres} | NMS IoU: {self.iou_thres} | Device: {self.device}")

    def predict(self, frame_or_path, imgsz=config.INFERENCE_IMG_SIZE):
        """
        Runs single-pass inference with the given confidence and NMS IoU threshold.
        Returns:
            results: ultralytics Results object
            inference_time_ms: inference time in milliseconds
        """
        t0 = time.perf_counter()
        results = self.model.predict(
            source=frame_or_path,
            conf=self.conf_thres,
            iou=self.iou_thres,
            imgsz=imgsz,
            device=self.device,
            verbose=False
        )
        t1 = time.perf_counter()
        inference_time_ms = (t1 - t0) * 1000.0
        return results[0], inference_time_ms
