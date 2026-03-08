"""
Pose Estimator - Extract 17 keypoints for posture analysis
Uses YOLOv8 pose model
"""
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None


class PoseEstimator:
    """Estimate human pose with 17 keypoints"""
    
    # COCO keypoint format
    KEYPOINTS = [
        'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
        'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
        'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
        'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
    ]
    
    # Skeleton connections for visualization
    SKELETON = [
        [0, 1], [0, 2], [1, 3], [2, 4],  # Head
        [5, 6], [5, 7], [7, 9], [6, 8], [8, 10],  # Arms
        [5, 11], [6, 12], [11, 12],  # Torso
        [11, 13], [13, 15], [12, 14], [14, 16]  # Legs
    ]
    
    def __init__(self, model_path: str = "yolov8n-pose.pt"):
        self.model_path = model_path
        self.model = None
        self._load_model()
    
    def _load_model(self):
        """Load YOLOv8 pose model"""
        if YOLO is None:
            raise ImportError("ultralytics not installed")
        
        try:
            self.model = YOLO(self.model_path)
            print(f"Pose model loaded: {self.model_path}")
        except Exception as e:
            print(f"Failed to load pose model: {e}")
            # Try downloading
            print("Attempting to download model...")
            self.model = YOLO("yolov8n-pose.pt")
    
    def estimate(self, frame: np.ndarray, conf_threshold: float = 0.5) -> List[Dict]:
        """Estimate pose for all people in frame"""
        if self.model is None:
            return []
        
        results = self.model(frame, verbose=False)
        poses = []
        
        for result in results:
            if result.keypoints is None:
                continue
            
            boxes = result.boxes
            keypoints = result.keypoints
            
            if boxes is None or keypoints is None:
                continue
            
            for i, (box, kpts) in enumerate(zip(boxes, keypoints)):
                conf = float(box.conf[0])
                if conf < conf_threshold:
                    continue
                
                bbox = box.xyxy[0].cpu().numpy().tolist()
                
                # Extract keypoints - handle different tensor formats
                try:
                    if hasattr(kpts, 'data'):
                        kp_data = kpts.data.cpu().numpy()
                    else:
                        kp_data = kpts.xy.cpu().numpy()
                    
                    print(f"Pose {i}: kp_data shape = {kp_data.shape}, ndim = {kp_data.ndim}")
                    
                    keypoint_list = []
                    
                    # Handle different possible shapes
                    if kp_data.ndim == 3 and kp_data.shape[0] > i:
                        # Shape: (num_people, num_keypoints, 3)
                        for j in range(kp_data.shape[1]):
                            point = kp_data[i, j]
                            if len(point) >= 3:
                                keypoint_list.append([float(point[0]), float(point[1]), float(point[2])])
                            elif len(point) == 2:
                                keypoint_list.append([float(point[0]), float(point[1]), 1.0])
                    elif kp_data.ndim == 2:
                        # Shape: (num_keypoints, 3)
                        for j in range(kp_data.shape[0]):
                            point = kp_data[j]
                            if len(point) >= 3:
                                keypoint_list.append([float(point[0]), float(point[1]), float(point[2])])
                            elif len(point) == 2:
                                keypoint_list.append([float(point[0]), float(point[1]), 1.0])
                    
                    print(f"Pose {i}: extracted {len(keypoint_list)} keypoints")
                    
                except Exception as e:
                    print(f"Error extracting keypoints: {e}")
                    keypoint_list = []
                
                poses.append({
                    'bbox': bbox,
                    'keypoints': keypoint_list,
                    'confidence': conf
                })
        
        return poses
    
    def draw_poses(self, frame: np.ndarray, poses: List[Dict], 
                   show_skeleton: bool = False, show_keypoints: bool = True) -> np.ndarray:
        """Draw poses on frame"""
        for pose in poses:
            keypoints = pose['keypoints']
            
            if show_skeleton:
                for connection in self.SKELETON:
                    kp1_idx, kp2_idx = connection
                    if kp1_idx < len(keypoints) and kp2_idx < len(keypoints):
                        kp1 = keypoints[kp1_idx]
                        kp2 = keypoints[kp2_idx]
                        
                        if kp1[2] > 0.5 and kp2[2] > 0.5:
                            pt1 = (int(kp1[0]), int(kp1[1]))
                            pt2 = (int(kp2[0]), int(kp2[1]))
                            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
            
            if show_keypoints:
                for i, kp in enumerate(keypoints):
                    if kp[2] > 0.5:
                        x, y = int(kp[0]), int(kp[1])
                        color = (0, 0, 255) if i < 5 else (255, 0, 0)
                        cv2.circle(frame, (x, y), 4, color, -1)
        
        return frame


if __name__ == "__main__":
    estimator = PoseEstimator()
    print(f"Loaded pose estimator with {len(PoseEstimator.KEYPOINTS)} keypoints")
