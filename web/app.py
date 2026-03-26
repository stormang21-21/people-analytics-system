"""
VideoPeopleAnalytics_Temi - Flask Web Application
Optimized for TEMI robot WebView
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
import cv2
import numpy as np

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from src.detector import TEMIDetector
from src.tracker import TEMITracker
from src.dwell_tracker import TEMIDwellTracker
from src.temi_bridge import TEMIBridge

# Flask setup
static_folder = os.path.join(os.path.dirname(__file__), '..', 'static')
template_folder = os.path.join(os.path.dirname(__file__), 'templates')
os.makedirs(static_folder, exist_ok=True)

app = Flask(__name__, 
            static_folder=static_folder,
            template_folder=template_folder)
app.config['SECRET_KEY'] = 'temi-analytics-secret'
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# Global state
class TEMIAnalyticsSystem:
    def __init__(self):
        self.detector = None
        self.tracker = None
        self.dwell_tracker = None
        self.temi_bridge = TEMIBridge()
        
        self.camera = None
        self.is_running = False
        self.frame = None
        self.frame_lock = threading.Lock()
        
        self.analytics_data = {
            'people_count': 0,
            'zone_occupancy': {},
            'total_detections': 0
        }
        
        self.processing_thread = None
        
    def init_models(self):
        """Initialize lightweight models"""
        print("Initializing TEMI-optimized models...")
        self.detector = TEMIDetector(conf_threshold=0.5)
        self.tracker = TEMITracker(iou_threshold=0.3, max_age=5)
        self.dwell_tracker = TEMIDwellTracker()
        
        # Register TEMI callbacks
        self.temi_bridge.register_callbacks(
            on_person_detected=self._on_person_detected,
            on_person_lost=self._on_person_lost,
            on_dwell_alert=self._on_dwell_alert
        )
        
        print("Models initialized")
    
    def _on_person_detected(self, data):
        """Handle person detected event"""
        self.analytics_data['total_detections'] += 1
        socketio.emit('person_detected', data)
    
    def _on_person_lost(self, data):
        """Handle person lost event"""
        socketio.emit('person_lost', data)
    
    def _on_dwell_alert(self, data):
        """Handle dwell alert"""
        socketio.emit('dwell_alert', data)
    
    def start_camera(self, camera_id=0):
        """Start camera capture"""
        if self.camera is not None:
            self.camera.release()
        
        self.camera = cv2.VideoCapture(camera_id)
        self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        self.camera.set(cv2.CAP_PROP_FPS, 15)
        
        self.is_running = True
        self.processing_thread = threading.Thread(target=self._process_loop)
        self.processing_thread.daemon = True
        self.processing_thread.start()
        
        return True
    
    def stop_camera(self):
        """Stop camera capture"""
        self.is_running = False
        if self.processing_thread:
            self.processing_thread.join(timeout=1)
        if self.camera:
            self.camera.release()
            self.camera = None
        return True
    
    def _process_loop(self):
        """Main processing loop"""
        while self.is_running:
            if self.camera is None:
                time.sleep(0.1)
                continue
            
            ret, frame = self.camera.read()
            if not ret:
                time.sleep(0.01)
                continue
            
            # Process frame
            processed_frame, analytics = self._process_frame(frame)
            
            with self.frame_lock:
                self.frame = processed_frame
                self.analytics_data.update(analytics)
            
            # Emit updates via WebSocket
            socketio.emit('analytics_update', analytics)
            
            time.sleep(0.033)  # ~30 FPS
    
    def _process_frame(self, frame):
        """Process a single frame"""
        # Resize for faster processing
        small_frame = cv2.resize(frame, (320, 240))
        
        # Detect people
        detections = self.detector.detect(small_frame)
        
        # Scale detections back to original size
        scale_x = frame.shape[1] / 320
        scale_y = frame.shape[0] / 240
        if len(detections) > 0:
            detections[:, [0, 2]] *= scale_x
            detections[:, [1, 3]] *= scale_y
        
        # Track
        track_ids = self.tracker.update(detections)
        
        # Update dwell tracking
        if len(detections) > 0:
            self.dwell_tracker.update(track_ids, detections[:, :4])
            
            # Trigger TEMI callbacks
            for i, track_id in enumerate(track_ids):
                if i < len(detections):
                    self.temi_bridge.on_person_detected(
                        track_id, 
                        detections[i, :4].tolist(),
                        float(detections[i, 4])
                    )
        
        # Draw detections
        if len(detections) > 0:
            frame = self.detector.draw_detections(frame, detections, track_ids)
        
        # Get analytics
        analytics = {
            'people_count': len(track_ids),
            'zone_occupancy': self.dwell_tracker.get_zone_occupancy(),
            'total_detections': self.analytics_data['total_detections']
        }
        
        return frame, analytics
    
    def get_frame(self):
        """Get current frame"""
        with self.frame_lock:
            return self.frame.copy() if self.frame is not None else None
    
    def get_frame_bytes(self):
        """Get current frame as JPEG bytes"""
        frame = self.get_frame()
        if frame is None:
            return None
        
        _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
        return buffer.tobytes()
    
    def add_zone(self, zone_id, name, polygon):
        """Add a dwell zone"""
        if self.dwell_tracker:
            self.dwell_tracker.add_zone(zone_id, name, polygon)
            return True
        return False
    
    def remove_zone(self, zone_id):
        """Remove a dwell zone"""
        if self.dwell_tracker:
            self.dwell_tracker.remove_zone(zone_id)
            return True
        return False
    
    def get_zones(self):
        """Get all zones"""
        if self.dwell_tracker:
            return self.dwell_tracker.zones
        return {}

# Global system instance
system = TEMIAnalyticsSystem()

@app.route('/')
def index():
    """Main dashboard"""
    return render_template('index.html')

@app.route('/api/status')
def status():
    """Get system status"""
    return jsonify({
        'is_running': system.is_running,
        'people_count': system.analytics_data['people_count'],
        'temi_status': system.temi_bridge.get_status()
    })

@app.route('/api/start', methods=['POST'])
def start():
    """Start camera"""
    if system.detector is None:
        system.init_models()
    success = system.start_camera()
    return jsonify({'success': success})

@app.route('/api/stop', methods=['POST'])
def stop():
    """Stop camera"""
    success = system.stop_camera()
    return jsonify({'success': success})

@app.route('/api/zones', methods=['GET', 'POST'])
def zones():
    """Get or add zones"""
    if request.method == 'GET':
        return jsonify(system.get_zones())
    
    data = request.json
    success = system.add_zone(
        data['zone_id'],
        data['name'],
        data['polygon']
    )
    return jsonify({'success': success})

@app.route('/api/zones/<zone_id>', methods=['DELETE'])
def delete_zone(zone_id):
    """Remove a zone"""
    success = system.remove_zone(zone_id)
    return jsonify({'success': success})

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

@socketio.on('connect')
def handle_connect():
    """Client connected"""
    print('TEMI client connected')
    emit('connected', {'data': 'Connected to VideoPeopleAnalytics_Temi'})

@socketio.on('disconnect')
def handle_disconnect():
    """Client disconnected"""
    print('TEMI client disconnected')

if __name__ == '__main__':
    port = int(os.environ.get('TEMI_PORT', 5002))
    
    print("Starting VideoPeopleAnalytics_Temi...")
    print(f"Open http://localhost:{port} in TEMI WebView")
    
    socketio.run(app, host='0.0.0.0', port=port, debug=False, allow_unsafe_werkzeug=True)