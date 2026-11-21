"""
SiteGuard AI - Real-Time YOLOv8 PPE Detector & Visualizer
Processes video/webcam frames, runs inference, evaluates PPE compliance,
tracks workers, and renders color-coded bounding boxes and HUD stats.
"""

import time
import cv2
import numpy as np
import torch
from pathlib import Path
from ultralytics import YOLO

import config
from .tracker import WorkerTracker
from .compliance import ComplianceEngine

class SiteGuardDetector:
    def __init__(
        self,
        model_path=None,
        conf_thres=config.DEFAULT_CONFIDENCE_THRESHOLD,
        iou_thres=config.DEFAULT_NMS_IOU_THRESHOLD,
        device=None,
        enable_tracking=True
    ):
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        
        # Determine device
        if device is None:
            self.device = 0 if torch.cuda.is_available() else "cpu"
        else:
            self.device = device

        # Determine model
        if model_path:
            self.model_path = Path(model_path)
        elif config.BEST_MODEL_PATH.exists():
            self.model_path = config.BEST_MODEL_PATH
        elif config.YOLOV8N_PRETRAINED.exists():
            self.model_path = config.YOLOV8N_PRETRAINED
        else:
            self.model_path = Path("yolov8n.pt")

        print(f"[DETECTOR] Initializing SiteGuardDetector with {self.model_path} on {self.device}")
        self.model = YOLO(str(self.model_path))
        
        self.tracker = WorkerTracker(max_disappeared=25, iou_threshold=0.3) if enable_tracking else None
        self.compliance_engine = ComplianceEngine()

        # Color schemes (BGR)
        self.COLOR_COMPLIANT = (40, 200, 40)       # Green
        self.COLOR_VIOLATION = (40, 40, 235)       # Red
        self.COLOR_UNVERIFIED = (0, 190, 245)      # Amber / Yellow
        self.COLOR_PPE_POS = (60, 220, 60)         # Soft Green
        self.COLOR_PPE_NEG = (30, 30, 220)         # Deep Red
        self.COLOR_HUD_BG = (24, 24, 28)           # Dark Slate
        self.COLOR_TEXT = (255, 255, 255)

    def process_frame(self, frame, imgsz=config.INFERENCE_IMG_SIZE):
        """
        Runs complete pipeline on a single BGR frame.
        Returns:
            annotated_frame: visually annotated BGR image
            frame_stats: {
                'fps': float,
                'inference_ms': float,
                'total_workers': int,
                'compliant': int,
                'non_compliant': int,
                'unverified': int,
                'violations': list
            }
        """
        t_start = time.perf_counter()

        # 1. Run YOLO inference
        t0 = time.perf_counter()
        results = self.model.predict(
            source=frame,
            conf=self.conf_thres,
            iou=self.iou_thres,
            imgsz=imgsz,
            device=self.device,
            verbose=False
        )[0]
        inference_ms = (time.perf_counter() - t0) * 1000.0

        # Parse detections
        detections = []
        person_boxes = []

        if results.boxes is not None and len(results.boxes) > 0:
            boxes = results.boxes.xyxy.cpu().numpy()
            confs = results.boxes.conf.cpu().numpy()
            clss = results.boxes.cls.cpu().numpy().astype(int)

            for i in range(len(boxes)):
                cid = clss[i]
                cname = results.names.get(cid, config.CLASS_NAMES[cid] if cid < len(config.CLASS_NAMES) else f"cls_{cid}")
                box = [float(boxes[i][0]), float(boxes[i][1]), float(boxes[i][2]), float(boxes[i][3])]
                conf = float(confs[i])
                
                det_record = {
                    'class_id': cid,
                    'name': cname,
                    'conf': conf,
                    'box': box
                }
                detections.append(det_record)
                if cid == config.CLASS_ID_PERSON or cname.lower() in ['person', 'worker']:
                    person_boxes.append(box)

        # Fallback for close-up views: if person box was missed but PPE items were detected
        if len(person_boxes) == 0:
            for det in detections:
                if det['name'].lower() in ['no_helmet', 'helmet', 'hardhat', 'no-hardhat', 'no_vest', 'vest']:
                    pbox = det['box']
                    pw = pbox[2] - pbox[0]
                    ph = pbox[3] - pbox[1]
                    inferred_person = [
                        max(0.0, pbox[0] - pw * 0.4),
                        max(0.0, pbox[1] - ph * 0.2),
                        pbox[2] + pw * 0.4,
                        pbox[3] + ph * 3.5
                    ]
                    person_boxes.append(inferred_person)
                    break

        # 2. Update Worker Tracker
        tracked_workers = []
        if self.tracker is not None:
            tracked_workers = self.tracker.update(person_boxes)

        # 3. Evaluate Compliance
        compliance_results = self.compliance_engine.evaluate(detections, tracked_workers=tracked_workers)
        
        # 4. Render Annotations
        annotated_frame = self.render_annotations(
            frame,
            detections=detections,
            workers=compliance_results['workers']
        )

        t_end = time.perf_counter()
        total_time = max(0.001, t_end - t_start)
        fps = 1.0 / total_time

        # 5. Render Modern HUD Banner
        summary = compliance_results['summary']
        annotated_frame = self.render_hud(annotated_frame, fps, inference_ms, summary)

        frame_stats = {
            'fps': round(fps, 1),
            'inference_ms': round(inference_ms, 1),
            'total_workers': summary['total_workers'],
            'compliant': summary['compliant'],
            'non_compliant': summary['non_compliant'],
            'unverified': summary['unverified'],
            'violations': compliance_results['violations'],
            'workers': compliance_results['workers'],
            'violator_ids': summary.get('violator_ids', []),
            'violator_info': summary.get('violator_info', '')
        }

        return annotated_frame, frame_stats

    def render_annotations(self, frame, detections, workers):
        annotated = frame.copy()
        
        # 1. Draw individual PPE detections with subtle dashed/thin boxes
        for det in detections:
            cid = det['class_id']
            box = [int(v) for v in det['box']]
            conf = det['conf']
            name = det['name']

            # Skip person boxes (handled with compliance styling below)
            if cid == config.CLASS_ID_PERSON:
                continue

            # Determine color for PPE items
            name_lower = name.lower()
            if "no-" in name_lower or "no_" in name_lower or name_lower.startswith("no"):
                color = self.COLOR_PPE_NEG
            elif any(k in name_lower for k in ['helmet', 'hardhat', 'vest', 'goggle', 'glove', 'boot', 'mask']):
                color = self.COLOR_PPE_POS
            else:
                color = (180, 160, 60)  # Machinery / Vehicle / Cones

            # Draw thin bounding box for PPE
            cv2.rectangle(annotated, (box[0], box[1]), (box[2], box[3]), color, 1)
            label = f"{name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1)
            cv2.rectangle(annotated, (box[0], box[1] - th - 4), (box[0] + tw + 4, box[1]), color, -1)
            cv2.putText(annotated, label, (box[0] + 2, box[1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1, cv2.LINE_AA)

        # 2. Draw prominent worker compliance boxes
        for w in workers:
            box = [int(v) for v in w['box']]
            status = w['overall_status']
            wid = w['worker_id']

            if status == config.STATUS_NON_COMPLIANT:
                box_color = self.COLOR_VIOLATION
                missing_str = ", ".join(w['missing_items']) if w['missing_items'] else "PPE VIOLATION"
                badge_text = f"ALARM: WORKER #{wid} [{missing_str.upper()}]"
                # Draw thick bounding box for violator
                cv2.rectangle(annotated, (box[0], box[1]), (box[2], box[3]), box_color, 3)
                # Corner accents (white highlights for visibility)
                c_len = min(20, max(8, int((box[2] - box[0]) * 0.2)))
                cv2.line(annotated, (box[0], box[1]), (box[0] + c_len, box[1]), (255, 255, 255), 3)
                cv2.line(annotated, (box[0], box[1]), (box[0], box[1] + c_len), (255, 255, 255), 3)
                cv2.line(annotated, (box[2], box[3]), (box[2] - c_len, box[3]), (255, 255, 255), 3)
                cv2.line(annotated, (box[2], box[3]), (box[2], box[3] - c_len), (255, 255, 255), 3)
            elif status == config.STATUS_COMPLIANT:
                box_color = self.COLOR_COMPLIANT
                badge_text = f"Worker #{wid}: COMPLIANT"
                cv2.rectangle(annotated, (box[0], box[1]), (box[2], box[3]), box_color, 2)
            else:
                box_color = self.COLOR_UNVERIFIED
                badge_text = f"Worker #{wid}: UNVERIFIED"
                cv2.rectangle(annotated, (box[0], box[1]), (box[2], box[3]), box_color, 2)

            # Badge pill on top of worker box
            (bw, bh), _ = cv2.getTextSize(badge_text, cv2.FONT_HERSHEY_SIMPLEX, 0.52, 2)
            cv2.rectangle(annotated, (box[0], box[1] - bh - 8), (box[0] + bw + 10, box[1]), box_color, -1)
            cv2.putText(annotated, badge_text, (box[0] + 5, box[1] - 5), cv2.FONT_HERSHEY_SIMPLEX, 0.52, (255, 255, 255), 2, cv2.LINE_AA)

        return annotated

    def render_hud(self, frame, fps, inf_ms, summary):
        """Draws top professional HUD bar with real-time statistics and alarm banner."""
        h, w = frame.shape[:2]
        hud_h = 42

        # Semi-transparent overlay
        overlay = frame.copy()
        cv2.rectangle(overlay, (0, 0), (w, hud_h), self.COLOR_HUD_BG, -1)
        cv2.addWeighted(overlay, 0.85, frame, 0.15, 0, frame)

        # Draw Title
        cv2.putText(frame, "SiteGuard AI", (14, 26), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 220, 255), 2, cv2.LINE_AA)

        # Draw Stats: FPS | Workers | Compliant | Violations
        fps_text = f"FPS: {fps:.1f} ({inf_ms:.0f}ms)"
        workers_text = f"Workers: {summary['total_workers']}"
        comp_text = f"Compliant: {summary['compliant']}"
        viol_text = f"Violations: {summary['non_compliant']}"

        x_offset = int(w * 0.28)
        spacing = int(w * 0.17)

        # FPS
        cv2.putText(frame, fps_text, (x_offset, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (210, 210, 210), 1, cv2.LINE_AA)
        # Workers
        cv2.putText(frame, workers_text, (x_offset + spacing, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (255, 255, 255), 1, cv2.LINE_AA)
        # Compliant
        cv2.putText(frame, comp_text, (x_offset + 2 * spacing, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.48, self.COLOR_COMPLIANT, 2, cv2.LINE_AA)
        # Violations
        cv2.putText(frame, viol_text, (x_offset + 3 * spacing, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.48, self.COLOR_VIOLATION, 2, cv2.LINE_AA)

        # Bottom Urgent Alarm Banner when any worker is in violation
        if summary.get('non_compliant', 0) > 0:
            alert_h = 36
            alert_y = h - alert_h
            overlay_alert = frame.copy()
            cv2.rectangle(overlay_alert, (0, alert_y), (w, h), (25, 25, 210), -1)
            cv2.addWeighted(overlay_alert, 0.88, frame, 0.12, 0, frame)

            v_info = summary.get('violator_info', '')
            if v_info:
                alert_text = f"[!] CRITICAL ALARM: WORKER {v_info} SAFETY VIOLATION ACTIVE"
            else:
                alert_text = "[!] CRITICAL ALARM: SAFETY PPE VIOLATION ACTIVE"
            cv2.putText(frame, alert_text, (16, h - 11), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2, cv2.LINE_AA)

        return frame
