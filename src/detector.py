"""
YOLO26 Detector Wrapper
Handles person detection + object classification
"""
import cv2
import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


class YOLO26Detector:
    """YOLO26 detector for people and objects"""
    
    def __init__(self, model_path: str = "yolo26n.pt", device: str = "auto"):
        self.model_path = model_path
        self.device = device
        self.model = None
        self.classes = None
        self._load_model()
    
    def _load_model(self):
        """Load YOLO26 model"""
        if YOLO is None:
            raise ImportError("ultralytics not installed. Run: pip install ultralytics")
        
        print(f"Loading YOLO26 model: {self.model_path}")
        self.model = YOLO(self.model_path)
        self.classes = self.model.names
        print(f"Model loaded. Classes: {len(self.classes)}")
    
    def detect(self, frame: np.ndarray, conf_threshold: float = 0.5) -> List[Dict]:
        """
        Detect objects in frame
        
        Returns:
            List of detections: {
                'bbox': [x1, y1, x2, y2],
                'confidence': float,
                'class_id': int,
                'class_name': str
            }
        """
        results = self.model(frame, verbose=False)
        detections = []
        
        for result in results:
            boxes = result.boxes
            if boxes is None:
                continue
                
            for box in boxes:
                conf = float(box.conf[0])
                if conf < conf_threshold:
                    continue
                
                cls_id = int(box.cls[0])
                bbox = box.xyxy[0].cpu().numpy().tolist()
                
                detections.append({
                    'bbox': bbox,
                    'confidence': conf,
                    'class_id': cls_id,
                    'class_name': self.classes.get(cls_id, 'unknown')
                })
        
        return detections
    
    def detect_people(self, frame: np.ndarray, conf_threshold: float = 0.5) -> List[Dict]:
        """Detect only people"""
        all_detections = self.detect(frame, conf_threshold)
        return [d for d in all_detections if d['class_name'] == 'person']
    
    def detect_with_objects(self, frame: np.ndarray, conf_threshold: float = 0.5) -> Dict:
        """
        Detect people and objects they might be holding
        
        Returns:
            {
                'people': [...],
                'objects': [...],
                'all_detections': [...]
            }
        """
        all_detections = self.detect(frame, conf_threshold)
        
        people = [d for d in all_detections if d['class_name'] == 'person']
        objects = [d for d in all_detections if d['class_name'] != 'person']
        
        return {
            'people': people,
            'objects': objects,
            'all_detections': all_detections
        }
    
    def draw_detections(self, frame: np.ndarray, detections: List[Dict], 
                       color_map: Optional[Dict] = None) -> np.ndarray:
        """Draw bounding boxes on frame"""
        if color_map is None:
            color_map = {
                'person': (0, 255, 0),
                'cell phone': (255, 0, 0),
                'laptop': (0, 0, 255),
                'backpack': (255, 255, 0),
                'default': (128, 128, 128)
            }
        
        for det in detections:
            x1, y1, x2, y2 = map(int, det['bbox'])
            class_name = det['class_name']
            conf = det['confidence']
            
            color = color_map.get(class_name, color_map['default'])
            
            # Draw box
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            
            # Draw label
            label = f"{class_name} {conf:.2f}"
            (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            cv2.rectangle(frame, (x1, y1-th-10), (x1+tw, y1), color, -1)
            cv2.putText(frame, label, (x1, y1-5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)
        
        return frame


if __name__ == "__main__":
    # Test
    detector = YOLO26Detector()
    print(f"Loaded {len(detector.classes)} classes")
    print("Available classes:", list(detector.classes.values())[:20])
