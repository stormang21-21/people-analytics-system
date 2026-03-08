"""
Action Classifier - Detect actions like texting, fighting, falling
Uses pose keypoints to classify actions
"""
import numpy as np
from typing import List, Dict, Optional
from collections import deque
import time


class ActionClassifier:
    """Classify human actions from pose sequences"""
    
    ACTIONS = ['standing', 'walking', 'sitting', 'laying_down', 'texting', 'falling', 'fighting', 'loitering']
    
    def __init__(self, history_size: int = 30):
        self.history_size = history_size
        self.pose_history = {}  # track_id -> deque of poses
        self.action_history = {}  # track_id -> deque of actions
        self.last_update = {}  # track_id -> timestamp
    
    def update(self, track_id: int, pose: Dict) -> str:
        """
        Update action classification for a track
        
        Args:
            track_id: Person tracking ID
            pose: Pose dict with 'keypoints' and 'bbox'
        
        Returns:
            Classified action string
        """
        current_time = time.time()
        
        # Initialize history for new tracks
        if track_id not in self.pose_history:
            self.pose_history[track_id] = deque(maxlen=self.history_size)
            self.action_history[track_id] = deque(maxlen=10)
            self.last_update[track_id] = current_time
        
        # Add pose to history
        self.pose_history[track_id].append(pose)
        self.last_update[track_id] = current_time
        
        # Classify action
        action = self._classify(pose, self.pose_history[track_id])
        self.action_history[track_id].append(action)
        
        # Return most common recent action
        if len(self.action_history[track_id]) >= 3:
            from collections import Counter
            recent_actions = list(self.action_history[track_id])[-5:]
            most_common = Counter(recent_actions).most_common(1)[0][0]
            return most_common
        
        return action
    
    def _classify(self, current_pose: Dict, pose_history: deque) -> str:
        """Classify action from pose"""
        keypoints = current_pose.get('keypoints', [])
        
        if len(keypoints) < 13:
            return 'unknown'
        
        # Get key points
        nose = keypoints[0]
        left_shoulder = keypoints[5]
        right_shoulder = keypoints[6]
        left_wrist = keypoints[9]
        right_wrist = keypoints[10]
        left_hip = keypoints[11]
        right_hip = keypoints[12]
        left_knee = keypoints[13]
        right_knee = keypoints[14]
        left_ankle = keypoints[15]
        right_ankle = keypoints[16]
        
        # Check for falling (head below hips)
        head_y = nose[1]
        hip_y = (left_hip[1] + right_hip[1]) / 2
        
        if head_y > hip_y + 100 and nose[2] > 0.5:
            return 'falling'
        
        # Check for texting (hands near face)
        if (left_wrist[2] > 0.5 and right_wrist[2] > 0.5):
            face_y = (nose[1] + keypoints[1][1] + keypoints[2][1]) / 3
            face_x = (nose[0] + keypoints[1][0] + keypoints[2][0]) / 3
            
            left_hand_near_face = abs(left_wrist[1] - face_y) < 80 and abs(left_wrist[0] - face_x) < 80
            right_hand_near_face = abs(right_wrist[1] - face_y) < 80 and abs(right_wrist[0] - face_x) < 80
            
            if left_hand_near_face or right_hand_near_face:
                return 'texting'
        
        # Check for sitting (hips below knees)
        knee_y = (left_knee[1] + right_knee[1]) / 2
        hip_y_avg = (left_hip[1] + right_hip[1]) / 2
        
        if hip_y_avg > knee_y - 20 and all(kp[2] > 0.5 for kp in [left_hip, right_hip, left_knee, right_knee]):
            return 'sitting'
        
        # Check for laying down (torso is horizontal - shoulder and hip at similar height, both low)
        shoulder_y = (left_shoulder[1] + right_shoulder[1]) / 2
        torso_angle = abs(shoulder_y - hip_y_avg)
        
        # If torso is relatively flat (shoulders and hips at similar height) and low in frame
        if torso_angle < 50 and hip_y_avg > 200 and all(kp[2] > 0.5 for kp in [left_shoulder, right_shoulder, left_hip, right_hip]):
            return 'laying_down'
        
        # Check movement from history
        if len(pose_history) >= 5:
            old_pose = pose_history[0]
            old_keypoints = old_pose.get('keypoints', [])
            
            if len(old_keypoints) >= len(keypoints):
                # Calculate movement
                movement = 0
                count = 0
                for i in range(min(len(keypoints), len(old_keypoints))):
                    if keypoints[i][2] > 0.5 and old_keypoints[i][2] > 0.5:
                        dx = keypoints[i][0] - old_keypoints[i][0]
                        dy = keypoints[i][1] - old_keypoints[i][1]
                        movement += np.sqrt(dx**2 + dy**2)
                        count += 1
                
                if count > 0:
                    avg_movement = movement / count
                    if avg_movement > 30:
                        return 'walking'
        
        return 'standing'
    
    def check_fighting(self, poses: List[Dict], tracks: List) -> List[int]:
        """Detect potential fighting between people"""
        fighters = []
        
        if len(poses) < 2:
            return fighters
        
        for i, (pose1, track1) in enumerate(zip(poses, tracks)):
            for j, (pose2, track2) in enumerate(zip(poses, tracks)):
                if i >= j:
                    continue
                
                # Calculate distance between people
                bbox1 = pose1.get('bbox', [0, 0, 0, 0])
                bbox2 = pose2.get('bbox', [0, 0, 0, 0])
                
                center1 = ((bbox1[0] + bbox1[2]) / 2, (bbox1[1] + bbox1[3]) / 2)
                center2 = ((bbox2[0] + bbox2[2]) / 2, (bbox2[1] + bbox2[3]) / 2)
                
                distance = np.sqrt((center1[0] - center2[0])**2 + (center1[1] - center2[1])**2)
                
                # If people are very close and moving rapidly
                if distance < 100:
                    # Check for rapid movement in both
                    action1 = self.action_history.get(track1.id, deque(maxlen=10))
                    action2 = self.action_history.get(track2.id, deque(maxlen=10))
                    
                    if len(action1) >= 3 and len(action2) >= 3:
                        recent1 = list(action1)[-3:]
                        recent2 = list(action2)[-3:]
                        
                        # If both are moving/walking and close together
                        if ('walking' in recent1 or 'texting' in recent1) and \
                           ('walking' in recent2 or 'texting' in recent2):
                            fighters.extend([track1.id, track2.id])
        
        return list(set(fighters))
    
    def check_loitering(self, track_id: int, current_zone: Optional[str], 
                        dwell_time: float, threshold: float = 300) -> bool:
        """Check if person is loitering (in same zone too long)"""
        return dwell_time > threshold and current_zone is not None
    
    def cleanup_old_tracks(self, max_age: float = 30):
        """Remove old tracks that haven't been updated"""
        current_time = time.time()
        to_remove = []
        
        for track_id, last_time in self.last_update.items():
            if current_time - last_time > max_age:
                to_remove.append(track_id)
        
        for track_id in to_remove:
            del self.pose_history[track_id]
            del self.action_history[track_id]
            del self.last_update[track_id]


if __name__ == "__main__":
    classifier = ActionClassifier()
    print(f"Action classifier ready for: {classifier.ACTIONS}")
