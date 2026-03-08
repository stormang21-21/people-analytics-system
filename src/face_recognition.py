"""
Face Recognition Module
Detect and recognize faces in video streams
"""
import cv2
import numpy as np
import pickle
import os
from typing import List, Dict, Tuple, Optional
from datetime import datetime
from collections import defaultdict
import time


class FaceRecognizer:
    """Face detection and recognition system"""
    
    def __init__(self, data_dir: str = "data/faces"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        # Load face detection model (OpenCV DNN)
        self.face_detector = None
        self._load_face_detector()
        
        # Known faces database
        self.known_faces = {}  # name -> face encoding
        self.face_database_file = os.path.join(data_dir, "face_database.pkl")
        self._load_database()
        
        # Recognition tracking
        self.recognition_history = defaultdict(list)
        self.last_seen = {}
        
    def _load_face_detector(self):
        """Load OpenCV face detection model"""
        try:
            # Use OpenCV's DNN face detector
            model_file = "models/opencv_face_detector_uint8.pb"
            config_file = "models/opencv_face_detector.pbtxt"
            
            if os.path.exists(model_file) and os.path.exists(config_file):
                self.face_detector = cv2.dnn.readNetFromTensorflow(model_file, config_file)
                print("Face detector loaded (DNN)")
            else:
                # Fallback to Haar cascade
                cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
                self.face_detector = cv2.CascadeClassifier(cascade_path)
                print("Face detector loaded (Haar Cascade)")
        except Exception as e:
            print(f"Failed to load face detector: {e}")
            # Fallback to Haar
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_detector = cv2.CascadeClassifier(cascade_path)
    
    def _load_database(self):
        """Load known faces from database"""
        if os.path.exists(self.face_database_file):
            try:
                with open(self.face_database_file, 'rb') as f:
                    self.known_faces = pickle.load(f)
                print(f"Loaded {len(self.known_faces)} known faces")
            except Exception as e:
                print(f"Failed to load face database: {e}")
                self.known_faces = {}
    
    def _save_database(self):
        """Save known faces to database"""
        try:
            with open(self.face_database_file, 'wb') as f:
                pickle.dump(self.known_faces, f)
            print(f"Saved {len(self.known_faces)} faces to database")
        except Exception as e:
            print(f"Failed to save face database: {e}")
    
    def detect_faces(self, frame: np.ndarray, conf_threshold: float = 0.5) -> List[Dict]:
        """
        Detect faces in frame
        
        Returns:
            List of face detections: {
                'bbox': [x, y, w, h],
                'confidence': float,
                'face_id': str
            }
        """
        faces = []
        h, w = frame.shape[:2]
        
        if self.face_detector is None:
            return faces
        
        # Check if using DNN or Haar
        if hasattr(self.face_detector, 'setInput'):
            # DNN detector
            blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104.0, 177.0, 123.0])
            self.face_detector.setInput(blob)
            detections = self.face_detector.forward()
            
            for i in range(detections.shape[2]):
                confidence = detections[0, 0, i, 2]
                if confidence > conf_threshold:
                    x1 = int(detections[0, 0, i, 3] * w)
                    y1 = int(detections[0, 0, i, 4] * h)
                    x2 = int(detections[0, 0, i, 5] * w)
                    y2 = int(detections[0, 0, i, 6] * h)
                    
                    # Ensure valid coordinates
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(w, x2), min(h, y2)
                    
                    face_id = f"face_{int(time.time() * 1000)}_{i}"
                    faces.append({
                        'bbox': [x1, y1, x2 - x1, y2 - y1],
                        'confidence': float(confidence),
                        'face_id': face_id
                    })
        else:
            # Haar cascade
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            detections = self.face_detector.detectMultiScale(
                gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30)
            )
            
            for i, (x, y, w, h) in enumerate(detections):
                face_id = f"face_{int(time.time() * 1000)}_{i}"
                faces.append({
                    'bbox': [x, y, w, h],
                    'confidence': 0.8,  # Haar doesn't give confidence
                    'face_id': face_id
                })
        
        return faces
    
    def extract_face_encoding(self, face_img: np.ndarray) -> np.ndarray:
        """
        Extract face encoding/feature vector
        Simple implementation using histogram and basic features
        """
        # Resize to standard size
        face_img = cv2.resize(face_img, (128, 128))
        
        # Convert to grayscale
        if len(face_img.shape) == 3:
            gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = face_img
        
        # Extract features (simple histogram-based encoding)
        hist = cv2.calcHist([gray], [0], None, [256], [0, 256])
        hist = cv2.normalize(hist, hist).flatten()
        
        # Add some spatial features
        h, w = gray.shape
        center_region = gray[h//4:3*h//4, w//4:3*w//4]
        center_hist = cv2.calcHist([center_region], [0], None, [128], [0, 256]).flatten()
        
        # Combine features
        encoding = np.concatenate([hist, center_hist])
        
        return encoding
    
    def compare_faces(self, encoding1: np.ndarray, encoding2: np.ndarray, 
                      threshold: float = 0.6) -> Tuple[bool, float]:
        """
        Compare two face encodings
        Returns: (is_match, similarity_score)
        """
        # Calculate cosine similarity
        similarity = np.dot(encoding1, encoding2) / (np.linalg.norm(encoding1) * np.linalg.norm(encoding2))
        is_match = similarity > threshold
        return is_match, float(similarity)
    
    def recognize_face(self, face_img: np.ndarray) -> Tuple[Optional[str], float]:
        """
        Recognize a face against known database
        Returns: (name or None, confidence)
        """
        if not self.known_faces:
            return None, 0.0
        
        encoding = self.extract_face_encoding(face_img)
        
        best_match = None
        best_score = 0.0
        
        for name, known_encoding in self.known_faces.items():
            is_match, score = self.compare_faces(encoding, known_encoding)
            if is_match and score > best_score:
                best_match = name
                best_score = score
        
        return best_match, best_score
    
    def add_known_face(self, name: str, face_img: np.ndarray):
        """Add a new face to the database"""
        encoding = self.extract_face_encoding(face_img)
        self.known_faces[name] = encoding
        self._save_database()
        print(f"Added face: {name}")
    
    def remove_known_face(self, name: str):
        """Remove a face from the database"""
        if name in self.known_faces:
            del self.known_faces[name]
            self._save_database()
            print(f"Removed face: {name}")
    
    def process_frame(self, frame: np.ndarray, draw: bool = True) -> Tuple[np.ndarray, List[Dict]]:
        """
        Process frame for face detection and recognition
        Returns: (annotated_frame, face_data)
        """
        faces = self.detect_faces(frame)
        annotated = frame.copy() if draw else frame
        
        for face in faces:
            x, y, w, h = face['bbox']
            
            # Extract face region
            face_img = frame[y:y+h, x:x+w]
            if face_img.size == 0:
                continue
            
            # Try to recognize
            name, confidence = self.recognize_face(face_img)
            
            face['recognized_name'] = name
            face['recognition_confidence'] = confidence
            
            # Track recognition
            if name:
                self.recognition_history[name].append({
                    'timestamp': time.time(),
                    'confidence': confidence
                })
                self.last_seen[name] = time.time()
            
            # Draw on frame
            if draw:
                color = (0, 255, 0) if name else (0, 165, 255)
                cv2.rectangle(annotated, (x, y), (x+w, y+h), color, 2)
                
                label = name if name else "Unknown"
                label += f" ({confidence:.2f})" if name else ""
                
                cv2.putText(annotated, label, (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        return annotated, faces
    
    def get_recognition_stats(self) -> Dict:
        """Get recognition statistics"""
        stats = {
            'known_faces': len(self.known_faces),
            'recent_recognitions': {}
        }
        
        current_time = time.time()
        for name, history in self.recognition_history.items():
            # Count recognitions in last hour
            recent = [h for h in history if current_time - h['timestamp'] < 3600]
            if recent:
                stats['recent_recognitions'][name] = {
                    'count': len(recent),
                    'last_seen': self.last_seen.get(name, 0),
                    'avg_confidence': sum(h['confidence'] for h in recent) / len(recent)
                }
        
        return stats


if __name__ == "__main__":
    recognizer = FaceRecognizer()
    print(f"Face recognizer ready with {len(recognizer.known_faces)} known faces")
