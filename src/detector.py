"""
Lightweight YOLOv8n detector for TEMI robot
Person detection only, minimal resource usage
"""
import cv2
import numpy as np
from ultralytics import YOLO


class TEMIDetector:
    """Lightweight person detector using YOLOv8n"""
    
    def __init__(self, conf_threshold=0.5):
        self.conf_threshold = conf_threshold
        # Use YOLOv8n (nano) - smallest and fastest
        self.model = YOLO('yolov8n.pt')
        self.person_class = 0  # COCO class for person
        
    def detect(self, frame):
        """
        Detect people in frame
        Returns: list of detections [x1, y1, x2, y2, conf]
        """
        results = self.model(frame, verbose=False, classes=[self.person_class])
        
        detections = []
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    conf = float(box.conf[0])
                    if conf >= self.conf_threshold:
                        x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()
                        detections.append([x1, y1, x2, y2, conf])
        
        return np.array(detections) if detections else np.array([])
    
    def draw_detections(self, frame, detections, track_ids=None):
        """Draw detection boxes on frame"""
        for i, det in enumerate(detections):
            x1, y1, x2, y2, conf = det
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            
            # Color based on track ID
            if track_ids and i < len(track_ids):
                color = self._get_color(track_ids[i])
                label = f"Person {track_ids[i]}: {conf:.2f}"
            else:
                color = (0, 255, 0)
                label = f"Person: {conf:.2f}"
            
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
            cv2.putText(frame, label, (x1, y1 - 10),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        return frame
    
    def _get_color(self, track_id):
        """Generate consistent color for track ID"""
        colors = [
            (0, 255, 0),    # Green
            (255, 0, 0),    # Blue
            (0, 0, 255),    # Red
            (255, 255, 0),  # Cyan
            (255, 0, 255),  # Magenta
            (0, 255, 255),  # Yellow
        ]
        return colors[track_id % len(colors)]
