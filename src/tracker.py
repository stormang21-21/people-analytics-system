"""
Simple IOU tracker for TEMI robot
Lightweight alternative to ByteTrack
"""
import numpy as np


class TEMITracker:
    """Simple IOU-based tracker for minimal resource usage"""
    
    def __init__(self, iou_threshold=0.3, max_age=5):
        self.iou_threshold = iou_threshold
        self.max_age = max_age
        self.next_id = 1
        self.tracks = {}  # id -> {bbox, age}
        
    def update(self, detections):
        """
        Update tracks with new detections
        Returns: list of track IDs corresponding to detections
        """
        if len(detections) == 0:
            # Age all tracks
            for track_id in list(self.tracks.keys()):
                self.tracks[track_id]['age'] += 1
                if self.tracks[track_id]['age'] > self.max_age:
                    del self.tracks[track_id]
            return []
        
        # Get current track bboxes
        track_ids = list(self.tracks.keys())
        track_bboxes = [self.tracks[tid]['bbox'] for tid in track_ids]
        
        # Calculate IOU matrix
        iou_matrix = self._compute_iou_matrix(track_bboxes, detections[:, :4])
        
        # Assign detections to tracks
        assigned_tracks = set()
        assigned_detections = set()
        new_track_ids = []
        
        # Greedy assignment by IOU
        if len(iou_matrix) > 0:
            for det_idx in range(len(detections)):
                best_iou = 0
                best_track = -1
                
                for track_idx in range(len(track_ids)):
                    if track_idx in assigned_tracks:
                        continue
                    iou = iou_matrix[track_idx, det_idx]
                    if iou > best_iou:
                        best_iou = iou
                        best_track = track_idx
                
                if best_iou >= self.iou_threshold:
                    track_id = track_ids[best_track]
                    self.tracks[track_id]['bbox'] = detections[det_idx, :4]
                    self.tracks[track_id]['age'] = 0
                    assigned_tracks.add(best_track)
                    assigned_detections.add(det_idx)
                    new_track_ids.append(track_id)
                else:
                    # Create new track
                    new_id = self.next_id
                    self.next_id += 1
                    self.tracks[new_id] = {
                        'bbox': detections[det_idx, :4],
                        'age': 0
                    }
                    new_track_ids.append(new_id)
        else:
            # No existing tracks, create all new
            for det_idx in range(len(detections)):
                new_id = self.next_id
                self.next_id += 1
                self.tracks[new_id] = {
                    'bbox': detections[det_idx, :4],
                    'age': 0
                }
                new_track_ids.append(new_id)
        
        # Age unassigned tracks
        for track_id in track_ids:
            if track_id not in [track_ids[list(assigned_tracks)[i]] for i in range(len(assigned_tracks)) if i < len(track_ids)]:
                self.tracks[track_id]['age'] += 1
                if self.tracks[track_id]['age'] > self.max_age:
                    del self.tracks[track_id]
        
        return new_track_ids
    
    def _compute_iou(self, bbox1, bbox2):
        """Compute IOU between two bboxes"""
        x1_1, y1_1, x2_1, y2_1 = bbox1
        x1_2, y1_2, x2_2, y2_2 = bbox2
        
        xi1 = max(x1_1, x1_2)
        yi1 = max(y1_1, y1_2)
        xi2 = min(x2_1, x2_2)
        yi2 = min(y2_1, y2_2)
        
        inter_area = max(0, xi2 - xi1) * max(0, yi2 - yi1)
        
        box1_area = (x2_1 - x1_1) * (y2_1 - y1_1)
        box2_area = (x2_2 - x1_2) * (y2_2 - y1_2)
        
        union_area = box1_area + box2_area - inter_area
        
        return inter_area / union_area if union_area > 0 else 0
    
    def _compute_iou_matrix(self, bboxes1, bboxes2):
        """Compute IOU matrix between two sets of bboxes"""
        if len(bboxes1) == 0 or len(bboxes2) == 0:
            return np.array([])
        
        matrix = np.zeros((len(bboxes1), len(bboxes2)))
        for i, bbox1 in enumerate(bboxes1):
            for j, bbox2 in enumerate(bboxes2):
                matrix[i, j] = self._compute_iou(bbox1, bbox2)
        
        return matrix
    
    def get_active_tracks(self):
        """Get number of active tracks"""
        return len(self.tracks)
