"""
Robust Camera Handler for RTSP/IP cameras
Prevents video hangs with buffer management and auto-reconnection
"""
import cv2
import numpy as np
import threading
import time
from typing import Optional, Callable


class CameraHandler:
    """Robust camera handler with auto-reconnection and buffer management"""
    
    def __init__(self, url: str, name: str = "Camera"):
        self.url = url
        self.name = name
        self.cap: Optional[cv2.VideoCapture] = None
        self.frame: Optional[np.ndarray] = None
        self.is_running = False
        self.is_connected = False
        self.capture_thread: Optional[threading.Thread] = None
        self.lock = threading.Lock()
        self.last_frame_time = 0
        self.reconnect_attempts = 0
        self.max_reconnect_attempts = 10
        self.reconnect_delay = 2  # seconds
        self.frame_timeout = 5  # seconds before considering connection dead
        
    def connect(self) -> bool:
        """Connect to camera with RTSP optimizations"""
        print(f"[{self.name}] Connecting to {self.url}...")
        
        # Release existing connection
        self.disconnect()
        
        # Create capture with buffer settings
        self.cap = cv2.VideoCapture(self.url)
        
        # RTSP optimizations to prevent hanging
        self.cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize buffer
        self.cap.set(cv2.CAP_PROP_FPS, 15)  # Limit FPS
        
        # Check connection
        if self.cap.isOpened():
            self.is_connected = True
            self.is_running = True
            self.reconnect_attempts = 0
            self.last_frame_time = time.time()
            
            # Start capture thread
            self.capture_thread = threading.Thread(target=self._capture_loop, daemon=True)
            self.capture_thread.start()
            
            print(f"[{self.name}] Connected successfully")
            return True
        else:
            print(f"[{self.name}] Failed to connect")
            self.cap = None
            return False
    
    def disconnect(self):
        """Disconnect from camera"""
        self.is_running = False
        self.is_connected = False
        
        if self.capture_thread:
            self.capture_thread.join(timeout=1.0)
            self.capture_thread = None
        
        if self.cap:
            self.cap.release()
            self.cap = None
        
        with self.lock:
            self.frame = None
        
        print(f"[{self.name}] Disconnected")
    
    def _capture_loop(self):
        """Background capture loop"""
        while self.is_running:
            if not self.cap or not self.cap.isOpened():
                self.is_connected = False
                break
            
            ret, frame = self.cap.read()
            
            if ret and frame is not None:
                with self.lock:
                    self.frame = frame
                self.last_frame_time = time.time()
                self.reconnect_attempts = 0
            else:
                # Check for timeout
                if time.time() - self.last_frame_time > self.frame_timeout:
                    print(f"[{self.name}] Frame timeout - attempting reconnect")
                    self.is_connected = False
                    break
            
            time.sleep(0.001)
        
        # Attempt reconnection if still running
        if self.is_running and self.reconnect_attempts < self.max_reconnect_attempts:
            self.reconnect_attempts += 1
            print(f"[{self.name}] Reconnecting (attempt {self.reconnect_attempts}/{self.max_reconnect_attempts})...")
            time.sleep(self.reconnect_delay)
            self.connect()
    
    def get_frame(self) -> Optional[np.ndarray]:
        """Get latest frame (thread-safe)"""
        with self.lock:
            return self.frame.copy() if self.frame is not None else None
    
    def is_alive(self) -> bool:
        """Check if camera connection is alive"""
        if not self.is_connected:
            return False
        return time.time() - self.last_frame_time <= self.frame_timeout
    
    def get_status(self) -> dict:
        """Get camera status"""
        return {
            'name': self.name,
            'url': self.url,
            'connected': self.is_connected,
            'alive': self.is_alive(),
            'reconnect_attempts': self.reconnect_attempts,
            'last_frame_time': self.last_frame_time
        }
