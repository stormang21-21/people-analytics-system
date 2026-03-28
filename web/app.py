"""
Flask Web Application for VideoPeopleAnalytics
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
from src.demographics import DemographicsAnalyzer
from src.settings_manager import SettingsManager

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
        self.demographics = None
        self.settings_manager = SettingsManager()
        self.cameras = {}  # Multiple camera handlers
        self.camera = None  # Primary active camera
        self.is_running = False
        self.frame = None
        self.frames = {}  # Frames from all cameras
        self.analytics_data = {
            'people_count': 0,
            'dwell_times': {},
            'zone_occupancy': {},
            'alerts': []
        }
        self.cameras_config = {}  # Camera configurations
        self.current_camera_id = None
        self.active_cameras = set()  # Set of active camera IDs
    
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
        
        try:
            self.demographics = DemographicsAnalyzer()
            print("Demographics analyzer loaded")
        except Exception as e:
            print(f"Demographics analyzer not loaded: {e}")
        
        print("Models initialized")
    
    def add_camera(self, camera_id: str, name: str, url: str, 
                   camera_type: str = "ip"):
        """Add camera configuration"""
        self.cameras_config[camera_id] = {
            'id': camera_id,
            'name': name,
            'url': url,
            'type': camera_type,
            'status': 'disconnected',
            'added_at': datetime.now().isoformat()
        }
        print(f"Added camera config: {name} ({url})")
    
    def connect_camera(self, camera_id: str):
        """Connect to camera (supports multiple cameras)"""
        if camera_id not in self.cameras_config:
            return False
        
        camera_info = self.cameras_config[camera_id]
        url = camera_info['url']
        
        # Disconnect existing camera with same ID
        if camera_id in self.cameras:
            self.cameras[camera_id].disconnect()
            del self.cameras[camera_id]
        
        # Connect to new camera with 2K capture and 720p processing
        camera_handler = CameraHandler(
            url, 
            camera_info["name"],
            capture_resolution=(2560, 1440),  # 2K for display
            processing_resolution=(1280, 720)  # 720p for AI processing
        )
        if camera_handler.connect():
            self.cameras[camera_id] = camera_handler
            self.cameras_config[camera_id]['status'] = 'connected'
            self.active_cameras.add(camera_id)
            # Set as primary if no primary exists
            if self.current_camera_id is None:
                self.current_camera_id = camera_id
                self.camera = camera_handler
            return True
        else:
            camera_info['status'] = 'error'
            return False
    
    def disconnect_camera(self, camera_id: str):
        """Disconnect a specific camera"""
        if camera_id in self.cameras:
            self.cameras[camera_id].disconnect()
            del self.cameras[camera_id]
            self.active_cameras.discard(camera_id)
            self.cameras_config[camera_id]['status'] = 'disconnected'
            
            # Update primary camera if needed
            if self.current_camera_id == camera_id:
                if self.active_cameras:
                    self.current_camera_id = next(iter(self.active_cameras))
                    self.camera = self.cameras.get(self.current_camera_id)
                else:
                    self.current_camera_id = None
                    self.camera = None
            return True
        return False
    
    def switch_camera(self, camera_id: str):
        """Switch primary camera for viewing"""
        if camera_id in self.cameras and camera_id in self.active_cameras:
            self.current_camera_id = camera_id
            self.camera = self.cameras[camera_id]
            return True
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
                # Get processing frame (scaled down) for AI
                frame = self.camera.get_processing_frame()
                # Get full frame for display
                full_frame = self.camera.get_frame()
                if frame is not None:
                    # Detect people and objects
                    detections = self.detector.detect_with_objects(frame, conf_threshold=0.3)
                    
                    # Debug: print all detections with confidence
                    all_info = [(d['class_name'], f"{d['confidence']:.2f}") for d in detections['all_detections']]
                    people_count = len(detections['people'])
                    object_count = len(detections['objects'])
                    print(f"Frame detections - People: {people_count}, Objects: {object_count}")
                    print(f"All detections: {all_info}")
                    
                    # Track
                    tracks = self.tracker.update(detections['people'])
                    
                    # Update dwell times (if enabled)
                    if self.settings_manager.is_enabled('dwell'):
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
                    
                    # Pose estimation and action classification (if enabled)
                    actions = {}
                    if self.settings_manager.is_enabled('pose') and self.pose_estimator:
                        try:
                            print(f"Pose estimation enabled, estimating...")
                            poses = self.pose_estimator.estimate(frame)
                            print(f"Pose estimation complete: {len(poses)} poses detected")
                            if self.settings_manager.is_enabled('pose'):
                                annotated_frame = self.pose_estimator.draw_poses(annotated_frame, poses)
                                print(f"Pose drawing complete")
                            
                            # Classify actions (if enabled)
                            if self.settings_manager.is_enabled('actions') and self.action_classifier:
                                print(f"Action recognition enabled, classifying {len(poses)} poses...")
                                for pose, track in zip(poses, tracks):
                                    print(f"Classifying track {track.id} with pose: {len(pose.get('keypoints', []))} keypoints")
                                    action = self.action_classifier.update(track.id, pose)
                                    actions[track.id] = action
                                    print(f"Track {track.id}: action = {action}")
                                    
                                    # Check for alerts (if enabled)
                                    if self.settings_manager.is_enabled('fall') and action == "falling":
                                        self.alert_system.check_fall(track.id, action)
                                
                                # Check for fights (if enabled)
                                if self.settings_manager.is_enabled('fight'):
                                    fighter_ids = self.action_classifier.check_fighting(poses, tracks)
                                    if fighter_ids:
                                        self.alert_system.check_fight(fighter_ids)
                        except Exception as e:
                            print(f"Pose/Action error: {e}")
                    
                    # Record analytics (if enabled)
                    if self.settings_manager.is_enabled('analytics'):
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
                    for i, track in enumerate(tracks):
                        x1, y1, x2, y2 = map(int, track.bbox)
                        dwell_times = self.dwell_tracker.get_dwell_times(track.id)
                        
                        # Draw track ID with better readability
                        action_str = actions.get(track.id, "")
                        label = f"ID: {track.id}"
                        if action_str:
                            label += f" | {action_str}"
                        if dwell_times:
                            dwell_str = ", ".join([f"{z}:{t:.0f}s" 
                                                   for z, t in dwell_times.items()])
                            label += f" | {dwell_str}"
                        
                        font = cv2.FONT_HERSHEY_SIMPLEX
                        font_scale = 0.7
                        thickness = 2
                        
                        (tw, th), _ = cv2.getTextSize(label, font, font_scale, thickness)
                        
                        # Calculate vertical offset for multiple tracks
                        y_offset = i * 30
                        
                        # Draw black background
                        cv2.rectangle(annotated_frame, 
                                    (x1, y1 - th - 8 - y_offset), 
                                    (x1 + tw + 8, y1 - y_offset), 
                                    (0, 0, 0), -1)
                        
                        # Draw text in bright cyan
                        cv2.putText(annotated_frame, label, 
                                  (x1 + 4, y1 - 4 - y_offset),
                                  font, font_scale, (0, 255, 255), thickness)
                    
                    # Scale annotations back to full resolution for display
                    if full_frame is not None:
                        # Calculate scale factor
                        scale_x = full_frame.shape[1] / frame.shape[1]
                        scale_y = full_frame.shape[0] / frame.shape[0]
                        
                        # Resize annotated frame to full resolution
                        annotated_full = cv2.resize(annotated_frame, 
                                                    (full_frame.shape[1], full_frame.shape[0]),
                                                    interpolation=cv2.INTER_LINEAR)
                        self.frame = annotated_full
                    else:
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
                    print(f"Emitting analytics: people={self.analytics_data['people_count']}, zones={len(self.analytics_data['zone_occupancy'])}")
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

# Add default webcam
system.add_camera(
    camera_id="cam_01",
    name="Webcam",
    url="0",
    camera_type="usb"
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

@app.route('/zones_test')
def zones_test():
    """Test zones page"""
    return render_template('zones_test.html')

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

@app.route('/zones_debug')
def zones_debug():
    """Debug canvas clicks"""
    return render_template('zones_debug.html')

@app.route('/test')
def test_page():
    """Simple canvas test"""
    return render_template('test.html')

# API Routes
@app.route('/api/cameras', methods=['GET'])
def get_cameras():
    """Get all cameras"""
    return jsonify(system.cameras_config)

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
    # Disconnect first if connected
    system.disconnect_camera(camera_id)
    if camera_id in system.cameras_config:
        del system.cameras_config[camera_id]
        return jsonify({'success': True})
    return jsonify({'success': False, 'error': 'Camera not found'}), 404

@app.route('/api/cameras/<camera_id>/disconnect', methods=['POST'])
def disconnect_camera(camera_id):
    """Disconnect camera"""
    success = system.disconnect_camera(camera_id)
    return jsonify({'success': success})

@app.route('/api/cameras/<camera_id>/switch', methods=['POST'])
def switch_camera(camera_id):
    """Switch to camera as primary"""
    success = system.switch_camera(camera_id)
    return jsonify({'success': success})

@app.route('/api/cameras/active', methods=['GET'])
def get_active_cameras():
    """Get list of active cameras"""
    return jsonify({
        'active': list(system.active_cameras),
        'current': system.current_camera_id
    })

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

@app.route('/api/zones/<zone_id>', methods=['DELETE'])
def delete_zone(zone_id):
    """Delete a zone"""
    if zone_id in system.dwell_tracker.zones:
        del system.dwell_tracker.zones[zone_id]
        return jsonify({'success': True})
    return jsonify({'success': False, 'error': 'Zone not found'}), 404

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


@app.route('/api/settings/features', methods=['GET'])
def get_feature_settings():
    """Get current feature settings"""
    return jsonify(system.settings_manager.get_all())

@app.route('/api/settings/features', methods=['POST'])
def set_feature_settings():
    """Update feature settings"""
    data = request.json
    success = system.settings_manager.update(data)
    return jsonify({'success': success})


@app.route('/api/snapshot')
def get_snapshot():
    """Get current frame as JPEG"""
    if system.frame is not None:
        ret, buffer = cv2.imencode('.jpg', system.frame)
        if ret:
            return Response(buffer.tobytes(), mimetype='image/jpeg')
    return jsonify({'error': 'No frame available'}), 404


@app.route('/api/demographics', methods=['GET'])
def get_demographics():
    """Get demographic statistics"""
    if system.demographics:
        return jsonify(system.demographics.get_session_stats())
    return jsonify({'error': 'Demographics not available'}), 404


@app.route('/api/demographics/current', methods=['GET'])
def get_current_demographics():
    """Get current frame demographics"""
    if system.frame is None:
        return jsonify({'error': 'No video frame available'}), 404
    
    if system.demographics is None:
        return jsonify({'error': 'Demographics analyzer not loaded'}), 404
    
    frame = system.frame.copy()
    faces = system.demographics.detect_faces(frame)
    
    results = []
    for face_bbox in faces:
        x1, y1, x2, y2 = face_bbox
        face_img = frame[y1:y2, x1:x2]
        demo = system.demographics.analyze_face(face_img)
        demo['bbox'] = face_bbox
        results.append(demo)
        system.demographics.update_stats(demo)
    
    return jsonify({
        'faces_detected': len(faces),
        'demographics': results
    })

# Camera permission page
@app.route('/camera_permission')
def camera_permission():
    """Page to request camera permission from browser"""
    return '''
<!DOCTYPE html>
<html>
<head>
    <title>Camera Permission</title>
    <style>
        body { font-family: Arial, sans-serif; padding: 20px; text-align: center; }
        video { width: 640px; height: 480px; border: 2px solid #333; margin: 20px auto; display: block; }
        button { padding: 15px 30px; font-size: 18px; cursor: pointer; background: #007bff; color: white; border: none; border-radius: 5px; }
        button:hover { background: #0056b3; }
        #status { margin: 20px; font-size: 16px; padding: 10px; border-radius: 5px; }
        .success { background: #d4edda; color: #155724; }
        .error { background: #f8d7da; color: #721c24; }
    </style>
</head>
<body>
    <h1>🎥 Camera Permission Request</h1>
    <p>Click the button below to grant camera access for People Analytics</p>
    <button id="requestBtn" onclick="requestCamera()">Request Camera Permission</button>
    <div id="status"></div>
    <video id="video" autoplay playsinline></video>

    <script>
        async function requestCamera() {
            const status = document.getElementById('status');
            const video = document.getElementById('video');
            const btn = document.getElementById('requestBtn');
            
            try {
                status.textContent = 'Requesting camera access...';
                status.className = '';
                const stream = await navigator.mediaDevices.getUserMedia({ video: true });
                video.srcObject = stream;
                status.textContent = '✅ SUCCESS: Camera access granted! You can now use the webcam with People Analytics.';
                status.className = 'success';
                btn.textContent = 'Camera Active';
                btn.disabled = true;
            } catch (err) {
                status.textContent = '❌ FAILED: ' + err.message;
                status.className = 'error';
            }
        }
    </script>
</body>
</html>
    '''

# Video feed
@app.route('/video_feed')
def video_feed():
    """MJPEG video stream - primary camera"""
    def generate():
        while True:
            frame_bytes = system.get_frame_bytes()
            if frame_bytes:
                yield (b'--frame\r\n'
                       b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')
            time.sleep(0.033)
    
    return Response(generate(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_feed/<camera_id>')
def video_feed_camera(camera_id):
    """MJPEG video stream - specific camera"""
    def generate():
        while True:
            if camera_id in system.cameras:
                frame = system.cameras[camera_id].get_frame()
                if frame is not None:
                    ret, buffer = cv2.imencode('.jpg', frame)
                    if ret:
                        yield (b'--frame\r\n'
                               b'Content-Type: image/jpeg\r\n\r\n' + buffer.tobytes() + b'\r\n')
            time.sleep(0.033)
    
    return Response(generate(),
                   mimetype='multipart/x-mixed-replace; boundary=frame')

# WebSocket events
@socketio.on('connect')
def handle_connect():
    print('Client connected')
    emit('connected', {'data': 'Connected to VideoPeopleAnalytics'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

if __name__ == '__main__':
    # Create templates directory if needed
    templates_dir = Path(__file__).parent / 'templates'
    templates_dir.mkdir(exist_ok=True)
    
    port = int(os.environ.get('PA_PORT', 5000))
    
    # Add default webcam
    system.add_camera(
        camera_id="cam_01",
        name="Webcam",
        url="0",
        camera_type="usb"
    )
    
    print("Starting Video Analytics Web Server...")
    print(f"Open http://localhost:{port} in your browser")
    
    socketio.run(app, host='0.0.0.0', port=port, debug=True, allow_unsafe_werkzeug=True)
