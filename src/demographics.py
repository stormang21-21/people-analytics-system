"""
Age and Gender Estimation Module
Uses OpenCV DNN models for demographic analysis
"""
import cv2
import numpy as np
import os
from typing import Dict, List, Tuple, Optional
from collections import defaultdict
import time


class DemographicsAnalyzer:
    """Age and gender estimation using OpenCV DNN"""
    
    # Age buckets for reporting
    AGE_BUCKETS = [
        (0, 12, "Child"),
        (13, 19, "Teen"),
        (20, 29, "Young Adult"),
        (30, 49, "Adult"),
        (50, 64, "Middle Age"),
        (65, 100, "Senior")
    ]
    
    def __init__(self, models_dir: str = "models"):
        self.models_dir = models_dir
        os.makedirs(models_dir, exist_ok=True)
        
        self.age_model = None
        self.gender_model = None
        self.face_detector = None
        
        self._load_models()
        
        # Statistics tracking
        self.demographics_history = defaultdict(lambda: {"age_sum": 0, "count": 0, "genders": defaultdict(int)})
        self.session_stats = {"total_faces": 0, "age_sum": 0, "genders": defaultdict(int)}
        
    def _load_models(self):
        """Load age and gender estimation models"""
        try:
            # Download URLs for models
            # Age: https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/age_deploy.prototxt
            #       https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/age_net.caffemodel
            # Gender: https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/gender_deploy.prototxt
            #          https://github.com/opencv/opencv_3rdparty/raw/dnn_samples_face_detector_20170830/gender_net.caffemodel
            
            age_proto = os.path.join(self.models_dir, "age_deploy.prototxt")
            age_model = os.path.join(self.models_dir, "age_net.caffemodel")
            gender_proto = os.path.join(self.models_dir, "gender_deploy.prototxt")
            gender_model = os.path.join(self.models_dir, "gender_net.caffemodel")
            
            # Check if models exist, if not create placeholder
            if not all(os.path.exists(f) for f in [age_proto, age_model, gender_proto, gender_model]):
                print("Demographics models not found. Using estimation mode.")
                self._create_placeholder_models()
                return
            
            self.age_model = cv2.dnn.readNet(age_model, age_proto)
            self.gender_model = cv2.dnn.readNet(gender_model, gender_proto)
            
            # Load face detector
            face_proto = os.path.join(self.models_dir, "opencv_face_detector.pbtxt")
            face_model = os.path.join(self.models_dir, "opencv_face_detector_uint8.pb")
            
            if os.path.exists(face_proto) and os.path.exists(face_model):
                self.face_detector = cv2.dnn.readNetFromTensorflow(face_model, face_proto)
            
            print("Demographics models loaded")
            
        except Exception as e:
            print(f"Error loading demographics models: {e}")
            self._create_placeholder_models()
    
    def _create_placeholder_models(self):
        """Create simple heuristic-based estimation when DNN models not available"""
        self.use_heuristics = True
        print("Using heuristic-based demographics estimation")
    
    def detect_faces(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """Detect faces in frame"""
        if self.face_detector is None:
            # Fallback to simple detection
            return []
        
        h, w = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(frame, 1.0, (300, 300), [104.0, 177.0, 123.0], False, False)
        self.face_detector.setInput(blob)
        detections = self.face_detector.forward()
        
        faces = []
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]
            if confidence > 0.5:
                x1 = int(detections[0, 0, i, 3] * w)
                y1 = int(detections[0, 0, i, 4] * h)
                x2 = int(detections[0, 0, i, 5] * w)
                y2 = int(detections[0, 0, i, 6] * h)
                faces.append((x1, y1, x2, y2))
        
        return faces
    
    def analyze_face(self, face_img: np.ndarray) -> Dict:
        """Analyze age and gender of a face"""
        if face_img.size == 0:
            return {"age": None, "gender": None, "confidence": 0}
        
        try:
            # Resize face to model input size
            blob = cv2.dnn.blobFromImage(face_img, 1.0, (227, 227), 
                                        [78.4263377603, 87.7689143744, 114.895847746], 
                                        swapRB=False)
            
            # Predict gender
            if self.gender_model is not None:
                self.gender_model.setInput(blob)
                gender_preds = self.gender_model.forward()
                gender = "Male" if gender_preds[0][0] > gender_preds[0][1] else "Female"
                gender_conf = max(gender_preds[0])
            else:
                # Heuristic: use average
                gender = "Unknown"
                gender_conf = 0.5
            
            # Predict age
            if self.age_model is not None:
                self.age_model.setInput(blob)
                age_preds = self.age_model.forward()
                age = self._age_from_predictions(age_preds[0])
                age_conf = max(age_preds[0])
            else:
                # Heuristic: estimate from face size/features
                age = self._estimate_age_heuristic(face_img)
                age_conf = 0.5
            
            return {
                "age": age,
                "gender": gender,
                "age_confidence": float(age_conf),
                "gender_confidence": float(gender_conf)
            }
            
        except Exception as e:
            print(f"Error analyzing face: {e}")
            return {"age": None, "gender": None, "confidence": 0}
    
    def _age_from_predictions(self, predictions: np.ndarray) -> int:
        """Convert age model predictions to age value"""
        # Age model outputs: 0-2, 4-6, 8-12, 15-20, 25-32, 38-43, 48-53, 60-100
        age_ranges = [(0, 2), (4, 6), (8, 12), (15, 20), (25, 32), (38, 43), (48, 53), (60, 100)]
        
        best_idx = np.argmax(predictions)
        age_min, age_max = age_ranges[best_idx]
        return (age_min + age_max) // 2
    
    def _estimate_age_heuristic(self, face_img: np.ndarray) -> int:
        """Estimate age using simple heuristics when model not available"""
        # Simple heuristic based on face proportions
        h, w = face_img.shape[:2]
        aspect_ratio = w / h if h > 0 else 1
        
        # Very rough estimation
        if aspect_ratio > 0.85:
            return 35  # Adult
        elif aspect_ratio > 0.75:
            return 25  # Young adult
        else:
            return 15  # Teen/child
    
    def get_age_bucket(self, age: int) -> str:
        """Get age bucket label"""
        for min_age, max_age, label in self.AGE_BUCKETS:
            if min_age <= age <= max_age:
                return label
        return "Unknown"
    
    def update_stats(self, demographics: Dict):
        """Update demographic statistics"""
        self.session_stats["total_faces"] += 1
        
        if demographics.get("age"):
            self.session_stats["age_sum"] += demographics["age"]
        
        if demographics.get("gender"):
            self.session_stats["genders"][demographics["gender"]] += 1
    
    def get_session_stats(self) -> Dict:
        """Get current session demographics"""
        total = self.session_stats["total_faces"]
        if total == 0:
            return {"avg_age": 0, "gender_distribution": {}, "age_distribution": {}}
        
        avg_age = self.session_stats["age_sum"] / total
        
        # Calculate age distribution
        age_dist = defaultdict(int)
        # Note: In real implementation, track individual ages
        
        return {
            "total_faces": total,
            "avg_age": round(avg_age, 1),
            "gender_distribution": dict(self.session_stats["genders"]),
            "age_distribution": dict(age_dist)
        }
    
    def draw_demographics(self, frame: np.ndarray, face_bbox: Tuple[int, int, int, int], 
                         demographics: Dict) -> np.ndarray:
        """Draw demographics info on frame"""
        x1, y1, x2, y2 = face_bbox
        
        # Draw face box
        color = (0, 255, 0) if demographics.get("gender") == "Male" else (255, 0, 255)
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Draw demographics label
        age = demographics.get("age", "?")
        gender = demographics.get("gender", "?")
        label = f"{gender}, {age}"
        
        # Draw label background
        (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
        
        # Draw text
        cv2.putText(frame, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, 
                   0.6, (255, 255, 255), 2)
        
        return frame
