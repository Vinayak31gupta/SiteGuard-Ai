"""
SiteGuard AI - Configuration Module
Defines paths, thresholds, compliance rules, and hardware settings.
"""

import os
from pathlib import Path

# Base Paths
BASE_DIR = Path(__file__).resolve().parent
DATASET_DIR = BASE_DIR / "dataset"
DATA_YAML_PATH = DATASET_DIR / "data.yaml"
MODELS_DIR = BASE_DIR / "models"
BEST_MODEL_PATH = MODELS_DIR / "best.pt"
LAST_MODEL_PATH = MODELS_DIR / "last.pt"
YOLOV8N_PRETRAINED = MODELS_DIR / "yolov8n.pt"

# Database & Storage
INSTANCE_DIR = BASE_DIR / "instance"
INSTANCE_DIR.mkdir(parents=True, exist_ok=True)
DB_PATH = INSTANCE_DIR / "siteguard.db"

UPLOADS_DIR = BASE_DIR / "uploads"
UPLOADS_DIR.mkdir(parents=True, exist_ok=True)
OUTPUTS_DIR = BASE_DIR / "outputs"
OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR = BASE_DIR / "reports"
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

# Dataset Class Names (Exact mapping from Roboflow dataset v30)
CLASS_NAMES = [
    'Excavator',        # 0
    'Gloves',           # 1 (PPE)
    'Hardhat',          # 2 (PPE)
    'Ladder',           # 3
    'Mask',             # 4 (PPE)
    'NO-Hardhat',       # 5 (Violation)
    'NO-Mask',          # 6 (Violation)
    'NO-Safety Vest',   # 7 (Violation)
    'Person',           # 8 (Worker)
    'SUV',              # 9
    'Safety Cone',      # 10
    'Safety Vest',      # 11 (PPE)
    'bus',              # 12
    'dump truck',       # 13
    'fire hydrant',     # 14
    'machinery',        # 15
    'mini-van',         # 16
    'sedan',            # 17
    'semi',             # 18
    'trailer',          # 19
    'truck and trailer',# 20
    'truck',            # 21
    'van',              # 22
    'vehicle',          # 23
    'wheel loader'      # 24
]

# Key Class IDs for rapid lookup
CLASS_ID_PERSON = 8
CLASS_ID_HARDHAT = 2
CLASS_ID_NO_HARDHAT = 5
CLASS_ID_SAFETY_VEST = 11
CLASS_ID_NO_SAFETY_VEST = 7
CLASS_ID_GLOVES = 1
CLASS_ID_MASK = 4
CLASS_ID_NO_MASK = 6

# Supported Upload Video Extensions
ALLOWED_VIDEO_EXTENSIONS = {'.mp4', '.avi', '.mov', '.m4v', '.mkv'}
MAX_CONTENT_LENGTH = 200 * 1024 * 1024  # 200 MB max upload

# Inference Defaults
DEFAULT_CONFIDENCE_THRESHOLD = 0.35
DEFAULT_NMS_IOU_THRESHOLD = 0.45
INFERENCE_IMG_SIZE = 640

# Compliance Status Labels
STATUS_COMPLIANT = "COMPLIANT"
STATUS_NON_COMPLIANT = "NON-COMPLIANT"
STATUS_UNVERIFIED = "UNVERIFIED"

# Violation cooldown window (seconds) to prevent duplicate records
VIOLATION_DEBOUNCE_SECONDS = 3.0

# Hardware Settings
GPU_DEVICE = 0  # RTX 3050 Laptop GPU (cuda:0)
FALLBACK_DEVICE = "cpu"
BATCH_SIZE = 16
TRAIN_EPOCHS = 35
