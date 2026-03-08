"""
Flask Web Application for Video Analytics System
IP Camera configuration, live view, analytics dashboard
"""
import os
import sys
import json
import time
import threading
from datetime import datetime
from pathlib import Path

from flask import Flask, render_template, jsonify, request, Response
from flask_socketio import SocketIO, emit
from functools import wraps
import cv2
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

# Basic Authentication Configuration
AUTH_ENABLED = True  # Set to False to disable auth
AUTH_USERNAME = os.environ.get('PA_USERNAME', 'admin')
AUTH_PASSWORD = os.environ.get('PA_PASSWORD', 'admin123')

def check_auth(username, password):
    """Verify credentials"""
    return username == AUTH_USERNAME and password == AUTH_PASSWORD

def authenticate():
    """Send 401 response"""
    return Response(
        'Authentication required',
        401,
        {'WWW-Authenticate': 'Basic realm="Video Analytics System"'}
    )

def requires_auth(f):
    """Decorator for protected routes"""
    @wraps(f)
    def decorated(*args, **kwargs):
        if not AUTH_ENABLED:
            return f(*args, **kwargs)
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)
    return decorated
from src.detector import YOLO26Detector
from src.camera_handler import CameraHandler
from src.tracker import ByteTrackTracker
from src.dwell_tracker import DwellTimeTracker
from src.pose_estimator import PoseEstimator
from src.action_classifier import ActionClassifier
from src.alert_system import AlertSystem, AlertType
from src.analytics import Analytics
from src.face_recognition import FaceRecognizer

# Static folder setup
static_folder = os.path.join(os.path.dirname(__file__), '..', 'static')
os.makedirs(static_folder, exist_ok=True)

app = Flask(__name__, static_folder=static_folder)
app.config['SECRET_KEY'] = 'people-analytics-secret'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global state
class AnalyticsSystem:
    def __init__(self):
        self.detector = None
        self.tracker = None
        self.dwell_tracker = None
        self.pose_estimator = None
        self.action_classifier = None
        self.alert_system = AlertSystem()
        self.analytics = Analytics()
        self.face_recognizer = None
        self.camera = None
        self.is_running = False
        self.frame = None
        self.analytics_data = {
            'people_count': 0,
            'dwell_times': {},
            'zone_occupancy': {},
            'alerts': []
        }
        self.cameras = {}
        self.current_camera_id = None
    
    def init_models(self):
        """Initialize detection and tracking models"""
        print("Initializing models...")
        self.detector = YOLO26Detector(model_path="yolo26n.pt")
        self.tracker = ByteTrackTracker()
        self.dwell_tracker = DwellTimeTracker()
        try:
            self.pose_estimator = PoseEstimator()
            self.action_classifier = ActionClassifier()
            print("Pose and action models loaded")
        except Exception as e:
            print(f"Pose/Action models not loaded: {e}")
        
        try:
            self.face_recognizer = FaceRecognizer()
            print("Face recognition loaded")
        except Exception as e:
            print(f"Face recognition not loaded: {e}")
        print("Models initialized")
    
    def add_camera(self, camera_id: str, name: str, url: str, 
                   camera_type: str = "ip"):
        """Add IP camera configuration"""
        self.cameras[camera_id] = {
            'id': camera_id,
            'name': name,
            'url': url,
            'type': camera_type,
            'status': 'disconnected',
            'added_at': datetime.now().isoformat()
        }
        print(f"Added camera: {name} ({url})")
    
    def connect_camera(self, camera_id: str):
        """Connect to camera"""
        if camera_id not in self.cameras:
            return False
        
        camera_info = self.cameras[camera_id]
        url = camera_info['url']
        
        # Release existing camera
        if self.camera:
            self.camera.disconnect()
        
        # Connect to new camera
        self.camera = CameraHandler(url, camera_info["name"])
        if self.camera.connect():
            camera_info['status'] = 'connected'
            self.current_camera_id = camera_id
            return True
        else:
            camera_info['status'] = 'error'
            return False
    
    def start_processing(self):
        """Start video processing loop"""
        if not self.detector:
            self.init_models()
        
        self.is_running = True
        threading.Thread(target=self._processing_loop, daemon=True).start()
    
    def _processing_loop(self):
        """Main processing loop"""
        while self.is_running:
            if self.camera and self.camera.is_alive():
                frame = self.camera.get_frame()
                if frame is not None:
                    # Detect people and objects
                    detections = self.detector.detect_with_objects(frame, conf_threshold=0.3)
                    
                    # Debug: print all detections
                    all_names = [d['class_name'] for d in detections['all_detections']]
                    people_count = len(detections['people'])
                    object_count = len(detections['objects'])
                    print(f"Frame detections - People: {people_count}, Objects: {object_count}, All: {all_names}")
                    
                    # Track
                    tracks = self.tracker.update(detections['people'])
                    
                    # Update dwell times
                    self.dwell_tracker.update(tracks)
                    
                    # Draw annotations
                    annotated_frame = frame.copy()
                    
                    # Draw all detections (people + objects like dogs)
                    annotated_frame = self.detector.draw_detections(
                        annotated_frame, detections['all_detections']
                    )
                    if self.face_recognizer:
                        try:
                            annotated_frame, faces = self.face_recognizer.process_frame(annotated_frame, draw=True)
                            # Add face count to analytics
                            face_count = len(faces)
                        except Exception as e:
                            print(f"Face recognition error: {e}")
                    
                    # Pose estimation and action classification
                    actions = {}
                    if self.pose_estimator:
                        try:
                            poses = self.pose_estimator.estimate(frame)
                            annotated_frame = self.pose_estimator.draw_poses(annotated_frame, poses)
                            
                            # Classify actions
                            if self.action_classifier:
                                for pose, track in zip(poses, tracks):
                                    action = self.action_classifier.update(track.id, pose)
                                    actions[track.id] = action
                                    
                                    # Check for alerts
                                    if action == "falling":
                                        self.alert_system.check_fall(track.id, action)
                                
                                # Check for fights
                                fighter_ids = self.action_classifier.check_fighting(poses, tracks)
                                if fighter_ids:
                                    self.alert_system.check_fight(fighter_ids)
                        except Exception as e:
                            print(f"Pose/Action error: {e}")
                    
                    # Record analytics
                    current_time = time.time()
                    zone_occupancy = {
                        zid: stats['current_occupancy'] 
                        for zid, stats in self.dwell_tracker.get_zone_analytics().items()
                    }
                    dwell_times = {
                        track.id: self.dwell_tracker.get_dwell_times(track.id)
                        for track in tracks
                    }
                    self.analytics.record(current_time, len(tracks), zone_occupancy, dwell_times)
                    annotated_frame = self.dwell_tracker.draw_zones(annotated_frame)
                    
                    # Add track IDs and dwell times
                    for track in tracks:
                        x1, y1, x2, y2 = map(int, track.bbox)
                        dwell_times = self.dwell_tracker.get_dwell_times(track.id)
                        
                        # Draw track ID with better readability
                        label = f"ID: {track.id}"
                        if dwell_times:
                            dwell_str = ", ".join([f"{z}:{t:.0f}s" 
                                                   for z, t in dwell_times.items()])
                            label += f" | {dwell_str}"
                        
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.8
                        thickness = 2
                        
                        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
                        
                        # Draw black background
                        cv2.rectangle(annotated_frame, 
                                    (x1, y1 - th - 8), 
                                    (x1 + tw + 8, y1), 
                                    (0, 0, 0), -1)
                        
                        # Draw text in bright cyan
                        cv2.putText(annotated_frame, label, 
                                  (x1 + 4, y1 - 4),
                                  font, font_scale, (0, 255, 255), thickness)
                    
                    self.frame = annotated_frame
                    
                    # Update analytics
                    self.analytics_data = {
                        'people_count': len(tracks),
                        'dwell_times': self.dwell_tracker.get_dwell_times(),
                        'zone_occupancy': {
                            zid: stats['current_occupancy']
                            for zid, stats in 
                            self.dwell_tracker.get_zone_analytics().items()
                        },
                        'timestamp': time.time()
                    }
                    
                    # Emit to clients
                    socketio.emit('analytics_update', self.analytics_data)
                
            time.sleep(0.033)  # ~30 FPS
    
    def get_frame_bytes(self):
        """Get current frame as JPEG bytes"""
        if self.frame is None:
            return None
        
        ret, buffer = cv2.imencode('.jpg', self.frame)
        if ret:
            return buffer.tobytes()
        return None
    
    def stop(self):
        """Stop processing"""
        self.is_running = False
        if self.camera:
            self.camera.disconnect()

# Global system instance
system = AnalyticsSystem()

# Add default IP camera
system.add_camera(
    camera_id="cam_01",
    name="Main Camera",
    url="rtsp://admin:PRZROZ@192.168.6.93:554/Streaming/channels/102/",
    camera_type="ip"
)

# Routes
@app.route('/settings')
def settings_page():
    """Settings page"""
    return render_template('settings.html')

@app.route('/faces')
def faces_page():
    """Face management page"""
    return render_template('faces.html')

@app.route('/zones_minimal')
def zones_minimal():
    """Minimal zones page - just video"""
    return render_template('zones_minimal.html')

@app.route('/zones_simple')
def zones_simple():
    """Simplified zones page for testing"""
    return render_template('zones_simple.html')

@app.route('/test_video')
def test_video():
    """Test video feed page"""
    return render_template('test_video.html')

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files"""
    return send_from_directory(static_folder, filename)

@app.route('/')
@requires_auth
def index():
    """Main dashboard"""
    return render_template('index.html')

@app.route('/cameras')
@requires_auth
def cameras_page():
    """Camera configuration page"""
    return render_template('cameras.html')

@app.route('/analytics')
@requires_auth
def analytics_page():
    """Analytics dashboard"""
    return render_template('analytics.html')

@app.route('/zones')
@requires_auth
def zones_page():
    """Zone configuration page"""
    return render_template('zones.html')

# API Routes
@app.route('/api/cameras', methods=['GET'])
def get_cameras():
    """Get all cameras"""
    return jsonify(system.cameras)

@app.route('/api/cameras', methods=['POST'])
def add_camera():
    """Add new camera"""
    data = request.json
    camera_id = data.get('id', f"cam_{int(time.time())}")
    
    system.add_camera(
        camera_id=camera_id,
        name=data['name'],
        url=data['url'],
        camera_type=data.get('type', 'ip')
    )
    
    return jsonify({'success': True, 'camera_id': camera_id})

@app.route('/api/cameras/<camera_id>/connect', methods=['POST'])
def connect_camera(camera_id):
    """Connect to camera"""
    success = system.connect_camera(camera_id)
    return jsonify({'success': success})

@app.route('/api/cameras/<camera_id>', methods=['DELETE'])
def delete_camera(camera_id):
    """Delete camera"""
    if camera_id in system.cameras:
        del system.cameras[camera_id]
        return jsonify({'success': True})
    return jsonify({'success': False, 'error': 'Camera not found'}), 404

@app.route('/api/zones', methods=['GET'])
def get_zones():
    """Get all zones"""
    zones = {}
    for zid, zone in system.dwell_tracker.zones.items():
        zones[zid] = {
            'id': zone.id,
            'name': zone.name,
            'polygon': zone.polygon,
            'color': zone.color
        }
    return jsonify(zones)

@app.route('/api/zones', methods=['POST'])
def add_zone():
    """Add new zone"""
    data = request.json
    system.dwell_tracker.add_zone(
        zone_id=data['id'],
        name=data['name'],
        polygon=data['polygon'],
        color=tuple(data.get('color', [0, 255, 0]))
    )
    return jsonify({'success': True})

@app.route('/api/analytics', methods=['GET'])
def get_analytics():
    """Get current analytics"""
    return jsonify({
        'current': system.analytics_data,
        'zone_stats': system.dwell_tracker.get_zone_analytics() if system.dwell_tracker else {}
    })

@app.route('/api/start', methods=['POST'])
def start_system():
    """Start processing"""
    system.start_processing()
    return jsonify({'success': True, 'status': 'running'})

@app.route('/api/stop', methods=['POST'])
def stop_system():
    """Stop processing"""
    system.stop()
    return jsonify({'success': True, 'status': 'stopped'})


@app.route('/api/alerts')
def get_alerts():
    """Get active alerts"""
    severity = request.args.get('severity')
    alerts = system.alert_system.get_active_alerts(severity)
    return jsonify([{
        'id': a.id,
        'type': a.type.value,
        'message': a.message,
        'severity': a.severity,
        'timestamp': a.timestamp,
        'track_id': a.track_id,
        'zone_id': a.zone_id,
        'acknowledged': a.acknowledged
    } for a in alerts])

@app.route('/api/alerts/<alert_id>/acknowledge', methods=['POST'])
def acknowledge_alert(alert_id):
    """Acknowledge an alert"""
    success = system.alert_system.acknowledge_alert(alert_id)
    return jsonify({'success': success})

@app.route('/api/alerts/<alert_id>/resolve', methods=['POST'])
def resolve_alert(alert_id):
    """Resolve an alert"""
    success = system.alert_system.resolve_alert(alert_id)
    return jsonify({'success': success})

@app.route('/api/analytics/current')
def get_current_analytics():
    """Get current analytics"""
    return jsonify(system.analytics.get_current_stats())

@app.route('/api/analytics/daily')
def get_daily_analytics():
    """Get daily analytics report"""
    date = request.args.get('date')
    return jsonify(system.analytics.get_daily_report(date))

@app.route('/api/analytics/export')
def export_analytics():
    """Export analytics to CSV"""
    filepath = system.analytics.export_csv()
    return jsonify({'success': True, 'filepath': filepath})


@app.route('/api/settings/resolution', methods=['POST'])
def set_resolution():
    """Change camera resolution"""
    data = request.json
    resolution = data.get('resolution', 'low')
    
    if not system.current_camera_id:
        return jsonify({'success': False, 'error': 'No camera connected'})
    
    camera_info = system.cameras.get(system.current_camera_id)
    if not camera_info:
        return jsonify({'success': False, 'error': 'Camera not found'})
    
    # Update URL based on resolution
    current_url = camera_info['url']
    if resolution == 'high':
        new_url = current_url.replace('/102/', '/101/')
    else:
        new_url = current_url.replace('/101/', '/102/')
    
    camera_info['url'] = new_url
    
    # Reconnect camera with new URL
    if system.camera:
        system.camera.disconnect()
    
    success = system.connect_camera(system.current_camera_id)
    
    return jsonify({'success': success, 'url': new_url})


@app.route('/api/faces')
def get_known_faces():
    """Get list of known faces"""
    if system.face_recognizer:
        return jsonify({
            'faces': list(system.face_recognizer.known_faces.keys()),
            'stats': system.face_recognizer.get_recognition_stats()
        })
    return jsonify({'faces': [], 'stats': {}})

@app.route('/api/faces/<name>', methods=['DELETE'])
def remove_face(name):
    """Remove a known face"""
    if system.face_recognizer:
        system.face_recognizer.remove_known_face(name)
        return jsonify({'success': True})
    return jsonify({'success': False, 'error': 'Face recognizer not available'})


@app.route('/api/faces/add', methods=['POST'])
def add_face():
    """Add a new face from current frame"""
    data = request.json
    name = data.get('name')
    
    if not name:
        return jsonify({'success': False, 'error': 'Name required'})
    
    if not system.face_recognizer:
        return jsonify({'success': False, 'error': 'Face recognition not available'})
    
    # Get frame from camera or system
    frame = None
    if system.camera and system.camera.is_alive():
        frame = system.camera.get_frame()
    elif system.frame is not None:
        frame = system.frame
    
    if frame is None:
        return jsonify({'success': False, 'error': 'No video frame available. Make sure camera is connected and video is streaming.'})
    
    # Detect faces in current frame
    faces = system.face_recognizer.detect_faces(frame)
    
    if not faces:
        return jsonify({'success': False, 'error': 'No face detected. Make sure face is clearly visible.'})
    
    # Use the first detected face
    face = faces[0]
    x, y, w, h = face['bbox']
    face_img = frame[y:y+h, x:x+w]
    
    if face_img.size == 0:
        return jsonify({'success': False, 'error': 'Invalid face region'})
    
    # Add to database
    system.face_recognizer.add_known_face(name, face_img)
    
    return jsonify({'success': True, 'name': name, 'faces_detected': len(faces)})

# Video feed
@app.route('/video_feed')
def video_feed():
    """MJPEG video stream"""
    def generate():
        while True:
            frame_bytes = system.get_frame_bytes()
            if frame_bytes:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.033)
    
    return Response(generate(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

# WebSocket events
@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('connected', {'data': 'Connected to Video Analytics System'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

if __name__ == '__main__':
    # Create templates directory if needed
    templates_dir = Path(__file__).parent / 'templates'
    templates_dir.mkdir(exist_ok=True)
    
    port = int(os.environ.get('PA_PORT', 5000))
    
    print("Starting Video Analytics Web Server...")
    print(f"Open http://localhost:{port} in your browser")
    
    socketio.run(app, host='0.0.0.0', port=port, debug=True, allow_unsafe_werkzeug=True)
