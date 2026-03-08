"""
ByteTrack Multi-Object Tracker
Persistent ID tracking across frames
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
import time


@dataclass
class Track:
    """Single track (person) with persistent ID"""
    id: int
    bbox: np.ndarray  # [x1, y1, x2, y2]
    confidence: float
    class_name: str
    age: int  # Frames since last update
    hits: int  # Total detections
    start_time: float
    last_update: float
    
    @property
    def center(self) -> Tuple[float, float]:
        """Get bounding box center"""
        return ((self.bbox[0] + self.bbox[2]) / 2, 
                (self.bbox[1] + self.bbox[3]) / 2)
    
    @property
    def dwell_time(self) -> float:
        """Time since track started"""
        return time.time() - self.start_time


class ByteTrackTracker:
    """
    Simplified ByteTrack implementation
    For production, use official ByteTrack or supervision.ByteTrack
    """
    
    def __init__(self, 
                 track_thresh: float = 0.5,
                 match_thresh: float = 0.8,
                 track_buffer: int = 30,
                 frame_rate: int = 30):
        self.track_thresh = track_thresh
        self.match_thresh = match_thresh
        self.track_buffer = track_buffer
        self.frame_rate = frame_rate
        
        self.tracks: List[Track] = []
        self.next_id = 1
        self.frame_count = 0
        
        # Track history for analytics
        self.track_history: Dict[int, List[Dict]] = defaultdict(list)
    
    def update(self, detections: List[Dict]) -> List[Track]:
        """
        Update tracks with new detections
        
        Args:
            detections: List of detection dicts with 'bbox', 'confidence', 'class_name'
        
        Returns:
            List of active tracks
        """
        self.frame_count += 1
        current_time = time.time()
        
        # Separate high and low confidence detections
        high_dets = [d for d in detections if d['confidence'] >= self.track_thresh]
        low_dets = [d for d in detections if d['confidence'] < self.track_thresh]
        
        # Predict new locations for existing tracks (simple linear prediction)
        for track in self.tracks:
            track.age += 1
        
        # Match high confidence detections to existing tracks
        matched_tracks = []
        unmatched_dets = []
        
        if self.tracks and high_dets:
            # Calculate IoU matrix
            iou_matrix = self._calc_iou_matrix(self.tracks, high_dets)
            
            # Hungarian algorithm or greedy matching
            matched_indices = self._greedy_match(iou_matrix)
            
            matched_track_indices = set()
            for track_idx, det_idx in matched_indices:
                if iou_matrix[track_idx, det_idx] >= self.match_thresh:
                    track = self.tracks[track_idx]
                    det = high_dets[det_idx]
                    
                    # Update track
                    track.bbox = np.array(det['bbox'])
                    track.confidence = det['confidence']
                    track.age = 0
                    track.hits += 1
                    track.last_update = current_time
                    
                    matched_tracks.append(track)
                    matched_track_indices.add(track_idx)
                    
                    # Save to history
                    self.track_history[track.id].append({
                        'frame': self.frame_count,
                        'time': current_time,
                        'bbox': det['bbox'],
                        'confidence': det['confidence']
                    })
            
            # Unmatched detections become new tracks
            for i, det in enumerate(high_dets):
                if i not in [m[1] for m in matched_indices]:
                    unmatched_dets.append(det)
        else:
            unmatched_dets = high_dets
        
        # Create new tracks for unmatched high-confidence detections
        for det in unmatched_dets:
            new_track = Track(
                id=self.next_id,
                bbox=np.array(det['bbox']),
                confidence=det['confidence'],
                class_name=det.get('class_name', 'person'),
                age=0,
                hits=1,
                start_time=current_time,
                last_update=current_time
            )
            self.tracks.append(new_track)
            matched_tracks.append(new_track)
            self.next_id += 1
        
        # Remove old tracks
        self.tracks = [t for t in self.tracks if t.age < self.track_buffer]
        
        return matched_tracks
    
    def _calc_iou(self, bbox1: np.ndarray, bbox2: np.ndarray) -> float:
        """Calculate IoU between two bboxes"""
        x1 = max(bbox1[0], bbox2[0])
        y1 = max(bbox1[1], bbox2[1])
        x2 = min(bbox1[2], bbox2[2])
        y2 = min(bbox1[3], bbox2[3])
        
        intersection = max(0, x2 - x1) * max(0, y2 - y1)
        area1 = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
        area2 = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
        union = area1 + area2 - intersection
        
        return intersection / union if union > 0 else 0
    
    def _calc_iou_matrix(self, tracks: List[Track], detections: List[Dict]) -> np.ndarray:
        """Calculate IoU matrix between tracks and detections"""
        matrix = np.zeros((len(tracks), len(detections)))
        for i, track in enumerate(tracks):
            for j, det in enumerate(detections):
                matrix[i, j] = self._calc_iou(track.bbox, np.array(det['bbox']))
        return matrix
    
    def _greedy_match(self, iou_matrix: np.ndarray) -> List[Tuple[int, int]]:
        """Greedy matching based on IoU"""
        matches = []
        if iou_matrix.size == 0:
            return matches
        
        iou_copy = iou_matrix.copy()
        while True:
            max_idx = np.unravel_index(np.argmax(iou_copy), iou_copy.shape)
            max_val = iou_copy[max_idx]
            
            if max_val < 0.1:  # Minimum IoU threshold
                break
            
            matches.append(max_idx)
            iou_copy[max_idx[0], :] = 0
            iou_copy[:, max_idx[1]] = 0
        
        return matches
    
    def get_active_tracks(self, max_age: int = 5) -> List[Track]:
        """Get tracks that have been updated recently"""
        return [t for t in self.tracks if t.age <= max_age]
    
    def get_track_stats(self) -> Dict:
        """Get tracking statistics"""
        return {
            'active_tracks': len(self.tracks),
            'total_tracks_created': self.next_id - 1,
            'frame_count': self.frame_count
        }


if __name__ == "__main__":
    # Test
    tracker = ByteTrackTracker()
    
    # Simulate detections
    test_dets = [
        {'bbox': [100, 100, 200, 300], 'confidence': 0.9, 'class_name': 'person'},
        {'bbox': [300, 150, 400, 350], 'confidence': 0.85, 'class_name': 'person'},
    ]
    
    tracks = tracker.update(test_dets)
    print(f"Created {len(tracks)} tracks")
    for t in tracks:
        print(f"  Track {t.id}: dwell={t.dwell_time:.2f}s")
