"""
Flask Web Application for People Analytics System
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
        {'WWW-Authenticate': 'Basic realm="People Analytics System"'}
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

app = Flask(__name__)
app.config['SECRET_KEY'] = 'people-analytics-secret'
socketio = SocketIO(app, cors_allowed_origins="*")

# Global state
class AnalyticsSystem:
    def __init__(self):
        self.detector = None
        self.tracker = None
        self.dwell_tracker = None
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
                    detections = self.detector.detect_with_objects(frame)
                    
                    # Track
                    tracks = self.tracker.update(detections['people'])
                    
                    # Update dwell times
                    self.dwell_tracker.update(tracks)
                    
                    # Draw annotations
                    annotated_frame = frame.copy()
                    annotated_frame = self.detector.draw_detections(
                        annotated_frame, detections['all_detections']
                    )
                    annotated_frame = self.dwell_tracker.draw_zones(annotated_frame)
                    
                    # Add track IDs and dwell times
                    for track in tracks:
                        x1, y1, x2, y2 = map(int, track.bbox)
                        dwell_times = self.dwell_tracker.get_dwell_times(track.id)
                        
                        # Draw track ID
                        label = f"ID: {track.id}"
                        if dwell_times:
                            dwell_str = ", ".join([f"{z}:{t:.0f}s" 
                                                   for z, t in dwell_times.items()])
                            label += f" | {dwell_str}"
                        
                        cv2.putText(annotated_frame, label, 
                                  (x1, y1 - 10),
                                  cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)
                    
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
        if frame is not None:
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

# Video feed
@app.route('/video_feed')
@requires_auth
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
    emit('connected', {'data': 'Connected to People Analytics System'})

@socketio.on('disconnect')
def handle_disconnect():
    print('Client disconnected')

if __name__ == '__main__':
    # Create templates directory if needed
    templates_dir = Path(__file__).parent / 'templates'
    templates_dir.mkdir(exist_ok=True)
    
    port = int(os.environ.get('PA_PORT', 5000))
    
    print("Starting People Analytics Web Server...")
    print(f"Open http://localhost:{port} in your browser")
    
    socketio.run(app, host='0.0.0.0', port=port, debug=True, allow_unsafe_werkzeug=True)
