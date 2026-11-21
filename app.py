"""
SiteGuard AI - Real-Time PPE and Construction Site Safety Compliance Detection System
Flask Backend Application
"""

import os
import sys
import time
import threading
from pathlib import Path
import cv2
from werkzeug.utils import secure_filename
from flask import Flask, render_template, Response, request, jsonify, send_from_directory, redirect, url_for

# Ensure project root is on sys.path
BASE_DIR = Path(__file__).resolve().parent
sys.path.append(str(BASE_DIR))

import config
from database import (
    init_db,
    record_event,
    record_violation,
    record_video_history,
    get_dashboard_stats,
    get_recent_violations,
    get_all_violations,
    clear_history,
    get_connection
)
from detection import SiteGuardDetector

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = config.MAX_CONTENT_LENGTH
app.config['SECRET_KEY'] = 'siteguard-ai-secret-key-2026'

# Initialize Database
init_db()

# Global Detector Instance
detector = None
detector_lock = threading.Lock()

def get_detector():
    global detector
    with detector_lock:
        if detector is None:
            detector = SiteGuardDetector(
                conf_thres=config.DEFAULT_CONFIDENCE_THRESHOLD,
                iou_thres=config.DEFAULT_NMS_IOU_THRESHOLD,
                enable_tracking=True
            )
        return detector

# Global Camera State
class CameraStream:
    def __init__(self, camera_index=0):
        self.camera_index = camera_index
        self.cap = None
        self.is_running = False
        self.lock = threading.Lock()
        self.current_frame = None
        self.latest_stats = {
            'fps': 0.0,
            'workers': 0,
            'compliant': 0,
            'violations': 0
        }
        self.latest_data = None

    def start(self):
        with self.lock:
            if not self.is_running:
                # Open webcam with DirectShow for faster initialization on Windows
                self.cap = cv2.VideoCapture(self.camera_index, cv2.CAP_DSHOW)
                if not self.cap.isOpened():
                    # Fallback to default backend
                    self.cap = cv2.VideoCapture(self.camera_index)
                
                if not self.cap.isOpened():
                    self.cap = None
                    return False
                
                self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
                self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
                self.is_running = True
                return True
            return True

    def stop(self):
        with self.lock:
            self.is_running = False
            if self.cap:
                self.cap.release()
                self.cap = None

    def get_frame(self):
        with self.lock:
            if self.cap and self.cap.isOpened():
                ret, frame = self.cap.read()
                if ret:
                    return frame
            return None

camera_stream = CameraStream(camera_index=0)

# -----------------------------------------------------------------------------
# Web Page Routes
# -----------------------------------------------------------------------------

@app.route('/')
def index():
    stats = get_dashboard_stats()
    recent_violations = get_recent_violations(limit=8)
    return render_template(
        'index.html',
        active_page='dashboard',
        stats=stats,
        recent_violations=recent_violations
    )

@app.route('/live')
def live_detection():
    return render_template('live_detection.html', active_page='live')

@app.route('/upload')
def upload_video():
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT * FROM video_processing_history ORDER BY id DESC LIMIT 10")
    video_history = [dict(r) for r in cursor.fetchall()]
    conn.close()
    return render_template('upload.html', active_page='upload', video_history=video_history)

@app.route('/image')
def image_detection():
    return render_template('image_detection.html', active_page='image')

@app.route('/history')
def violation_history():
    violations = get_all_violations(limit=250)
    return render_template('history.html', active_page='history', violations=violations)

@app.route('/history/clear', methods=['POST', 'GET'])
def clear_history_action():
    clear_history()
    return redirect(url_for('violation_history'))

ALLOWED_IMAGE_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp'}

@app.route('/api/image/process', methods=['POST'])
def api_process_image():
    if 'image' not in request.files:
        return jsonify({'status': 'error', 'message': 'No image file uploaded.'}), 400

    file = request.files['image']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No image selected.'}), 400

    ext = Path(file.filename).suffix.lower()
    if ext not in ALLOWED_IMAGE_EXTENSIONS:
        return jsonify({
            'status': 'error',
            'message': f'Unsupported image format {ext}. Allowed: JPG, PNG, WEBP, BMP.'
        }), 400

    filename = secure_filename(file.filename)
    timestamp = int(time.time())
    
    # Read image using OpenCV
    import numpy as np
    file_bytes = np.frombuffer(file.read(), np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
    if image is None:
        return jsonify({'status': 'error', 'message': 'Failed to decode image file.'}), 400

    det = get_detector()
    annotated_img, stats = det.process_frame(image)

    # Save annotated output
    output_filename = f"annotated_{timestamp}_{Path(filename).stem}.jpg"
    output_path = config.OUTPUTS_DIR / output_filename
    cv2.imwrite(str(output_path), annotated_img)

    # Log to SQLite
    event_id = record_event(
        source_type="image",
        workers_count=stats['total_workers'],
        compliant_count=stats['compliant'],
        non_compliant_count=stats['non_compliant'],
        unverified_count=stats['unverified'],
        fps=stats['fps']
    )

    for v in stats['violations']:
        record_violation(
            missing_ppe=v['missing_ppe'],
            confidence=v['confidence'],
            status=v['status'],
            worker_id=v['worker_id'],
            source_type="image",
            event_id=event_id
        )

    # Compliance status label
    if stats['non_compliant'] > 0:
        overall_status = "NON-COMPLIANT"
    elif stats['compliant'] > 0 and stats['non_compliant'] == 0:
        overall_status = "COMPLIANT"
    else:
        overall_status = "UNVERIFIED"

    return jsonify({
        'status': 'success',
        'download_url': f'/outputs/{output_filename}',
        'overall_status': overall_status,
        'summary': {
            'total_workers': stats['total_workers'],
            'compliant': stats['compliant'],
            'non_compliant': stats['non_compliant'],
            'unverified': stats['unverified'],
            'violations_logged': len(stats['violations'])
        },
        'violations': stats['violations']
    })

# -----------------------------------------------------------------------------
# Streaming Video Feed (Webcam with YOLOv8 Inference & Violation Recording)
# -----------------------------------------------------------------------------

def generate_webcam_frames():
    det = get_detector()
    last_db_log_time = time.time()

    while camera_stream.is_running:
        frame = camera_stream.get_frame()
        if frame is None:
            time.sleep(0.03)
            continue

        # Run detection and compliance evaluation
        annotated_frame, stats = det.process_frame(frame)

        # Update real-time camera state for alarm and worker ID telemetry
        with camera_stream.lock:
            camera_stream.latest_data = {
                'timestamp': time.time(),
                'fps': stats.get('fps', 0.0),
                'total_workers': stats.get('total_workers', 0),
                'compliant': stats.get('compliant', 0),
                'non_compliant': stats.get('non_compliant', 0),
                'unverified': stats.get('unverified', 0),
                'has_violation': len(stats.get('violations', [])) > 0,
                'violations': stats.get('violations', []),
                'violator_ids': stats.get('violator_ids', []),
                'violator_info': stats.get('violator_info', ''),
                'workers': [
                    {
                        'worker_id': w.get('worker_id'),
                        'overall_status': w.get('overall_status'),
                        'missing_items': w.get('missing_items', []),
                        'detected_items': w.get('detected_items', [])
                    }
                    for w in stats.get('workers', [])
                ]
            }

        # Log periodic summary event to SQLite (every 2.5 seconds)
        now = time.time()
        if now - last_db_log_time >= 2.5:
            last_db_log_time = now
            event_id = record_event(
                source_type="webcam",
                workers_count=stats['total_workers'],
                compliant_count=stats['compliant'],
                non_compliant_count=stats['non_compliant'],
                unverified_count=stats['unverified'],
                fps=stats['fps']
            )
            # Log any detected violations with anti-duplication debounce
            for v in stats['violations']:
                record_violation(
                    missing_ppe=v['missing_ppe'],
                    confidence=v['confidence'],
                    status=v['status'],
                    worker_id=v['worker_id'],
                    source_type="webcam",
                    event_id=event_id
                )

        # Encode frame to JPEG
        ret, buffer = cv2.imencode('.jpg', annotated_frame, [int(cv2.IMWRITE_JPEG_QUALITY), 82])
        if not ret:
            continue

        frame_bytes = buffer.tobytes()
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')

@app.route('/video_feed')
def video_feed():
    if not camera_stream.is_running:
        # Start camera automatically if not already running
        started = camera_stream.start()
        if not started:
            return "Camera unavailable", 503
    return Response(generate_webcam_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

# -----------------------------------------------------------------------------
# API Endpoints
# -----------------------------------------------------------------------------

@app.route('/api/camera/start', methods=['POST'])
def api_camera_start():
    success = camera_stream.start()
    if success:
        return jsonify({'status': 'started', 'message': 'Camera initialized successfully.'})
    else:
        return jsonify({
            'status': 'error',
            'message': 'Failed to initialize camera. Ensure a webcam is plugged in and not in use by another app.'
        }), 400

@app.route('/api/camera/stop', methods=['POST'])
def api_camera_stop():
    camera_stream.stop()
    with camera_stream.lock:
        camera_stream.latest_data = None
    return jsonify({'status': 'stopped', 'message': 'Camera stopped.'})

@app.route('/api/camera/status', methods=['GET'])
def api_camera_status():
    with camera_stream.lock:
        latest = getattr(camera_stream, 'latest_data', None)
        if not camera_stream.is_running or latest is None:
            return jsonify({
                'is_running': camera_stream.is_running,
                'has_violation': False,
                'violator_ids': [],
                'violator_info': '',
                'violations': [],
                'total_workers': 0,
                'compliant': 0,
                'non_compliant': 0,
                'unverified': 0,
                'fps': 0.0,
                'workers': []
            })
        
        now = time.time()
        data = dict(latest)
        if now - data.get('timestamp', 0) > 2.5:
            data['has_violation'] = False
            data['violator_ids'] = []
            data['violations'] = []

        data['is_running'] = camera_stream.is_running
        return jsonify(data)

@app.route('/api/stats', methods=['GET'])
def api_stats():
    stats = get_dashboard_stats()
    return jsonify(stats)

@app.route('/api/history/clear', methods=['POST'])
def api_clear_history():
    clear_history()
    return jsonify({'status': 'cleared', 'message': 'Violation history reset successfully.'})

# -----------------------------------------------------------------------------
# Uploaded Video Processing Engine
# -----------------------------------------------------------------------------

@app.route('/api/video/process', methods=['POST'])
def api_process_video():
    if 'video' not in request.files:
        return jsonify({'status': 'error', 'message': 'No video file provided.'}), 400

    file = request.files['video']
    if file.filename == '':
        return jsonify({'status': 'error', 'message': 'No selected video file.'}), 400

    ext = Path(file.filename).suffix.lower()
    if ext not in config.ALLOWED_VIDEO_EXTENSIONS:
        return jsonify({
            'status': 'error',
            'message': f'Unsupported file format {ext}. Allowed formats: MP4, AVI, MOV, MKV.'
        }), 400

    filename = secure_filename(file.filename)
    timestamp_prefix = int(time.time())
    input_filename = f"{timestamp_prefix}_{filename}"
    input_path = config.UPLOADS_DIR / input_filename
    file.save(str(input_path))

    output_filename = f"processed_{timestamp_prefix}_{Path(filename).stem}.mp4"
    output_path = config.OUTPUTS_DIR / output_filename

    # Process video with YOLOv8
    det = get_detector()
    cap = cv2.VideoCapture(str(input_path))
    if not cap.isOpened():
        return jsonify({'status': 'error', 'message': 'Unable to open and decode uploaded video.'}), 400

    fps_in = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    # Output video writer with H264 or mp4v
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(str(output_path), fourcc, fps_in, (width, height))

    processed_count = 0
    total_workers_sum = 0
    total_violations_count = 0
    t_start = time.perf_counter()

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        processed_count += 1
        annotated_frame, stats = det.process_frame(frame)
        out.write(annotated_frame)

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

    cap.release()
    out.release()
    t_total = time.perf_counter() - t_start

    processed_fps = round(processed_count / max(0.001, t_total), 1)
    avg_workers = int(total_workers_sum / max(1, processed_count))

    # Log to video processing history
    job_id = record_video_history(
        original_filename=filename,
        output_filename=output_filename,
        total_frames=processed_count,
        processed_fps=processed_fps,
        workers_detected=avg_workers,
        violations_detected=total_violations_count
    )

    return jsonify({
        'status': 'success',
        'job_id': job_id,
        'download_url': f'/outputs/{output_filename}',
        'stats': {
            'total_frames': processed_count,
            'fps': processed_fps,
            'workers_detected': avg_workers,
            'violations_detected': total_violations_count
        }
    })

@app.route('/outputs/<path:filename>')
def serve_output(filename):
    return send_from_directory(str(config.OUTPUTS_DIR), filename, as_attachment=False)

# -----------------------------------------------------------------------------
# Main Application Entry Point
# -----------------------------------------------------------------------------

if __name__ == '__main__':
    print("=" * 65)
    print("      SITEGUARD AI - SERVER INITIALIZING ON http://127.0.0.1:5000")
    print("=" * 65)
    app.run(host='127.0.0.1', port=5000, debug=False, threaded=True)
