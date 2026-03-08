"""
Dwell Time Tracker
Track how long people spend in defined zones
"""
import time
import cv2
import numpy as np
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass, field
from collections import defaultdict


@dataclass
class Zone:
    """Defined area for dwell time tracking"""
    id: str
    name: str
    polygon: List[Tuple[int, int]]  # List of (x, y) points
    color: Tuple[int, int, int] = (0, 255, 0)
    
    def contains_point(self, point: Tuple[float, float]) -> bool:
        """Check if point is inside zone polygon"""
        return self._point_in_polygon(point, self.polygon)
    
    def contains_bbox(self, bbox: List[float]) -> bool:
        """Check if bbox center is inside zone"""
        center_x = (bbox[0] + bbox[2]) / 2
        center_y = (bbox[1] + bbox[3]) / 2
        return self.contains_point((center_x, center_y))
    
    @staticmethod
    def _point_in_polygon(point: Tuple[float, float], 
                          polygon: List[Tuple[int, int]]) -> bool:
        """Ray casting algorithm for point-in-polygon"""
        x, y = point
        n = len(polygon)
        inside = False
        
        p1x, p1y = polygon[0]
        for i in range(1, n + 1):
            p2x, p2y = polygon[i % n]
            if y > min(p1y, p2y):
                if y <= max(p1y, p2y):
                    if x <= max(p1x, p2x):
                        if p1y != p2y:
                            xinters = (y - p1y) * (p2x - p1x) / (p2y - p1y) + p1x
                        if p1x == p2x or x <= xinters:
                            inside = not inside
            p1x, p1y = p2x, p2y
        
        return inside


@dataclass
class DwellRecord:
    """Record of time spent in a zone"""
    track_id: int
    zone_id: str
    enter_time: float
    exit_time: Optional[float] = None
    
    @property
    def duration(self) -> float:
        """Time spent in zone"""
        if self.exit_time:
            return self.exit_time - self.enter_time
        return time.time() - self.enter_time
    
    def to_dict(self) -> Dict:
        return {
            'track_id': self.track_id,
            'zone_id': self.zone_id,
            'enter_time': self.enter_time,
            'exit_time': self.exit_time,
            'duration': self.duration
        }


class DwellTimeTracker:
    """Track dwell time for multiple zones"""
    
    def __init__(self):
        self.zones: Dict[str, Zone] = {}
        # track_id -> zone_id -> DwellRecord
        self.active_dwells: Dict[int, Dict[str, DwellRecord]] = defaultdict(dict)
        # Historical records
        self.completed_dwells: List[DwellRecord] = []
        # Analytics
        self.zone_stats: Dict[str, Dict] = defaultdict(lambda: {
            'total_visits': 0,
            'total_dwell_time': 0,
            'current_occupancy': 0
        })
    
    def add_zone(self, zone_id: str, name: str, 
                 polygon: List[Tuple[int, int]],
                 color: Tuple[int, int, int] = (0, 255, 0)):
        """Add a zone for tracking"""
        self.zones[zone_id] = Zone(zone_id, name, polygon, color)
        print(f"Added zone '{name}' with {len(polygon)} points")
    
    def remove_zone(self, zone_id: str):
        """Remove a zone"""
        if zone_id in self.zones:
            del self.zones[zone_id]
    
    def update(self, tracks: List):
        """
        Update dwell times based on current track positions
        
        Args:
            tracks: List of Track objects with bbox and id
        """
        current_time = time.time()
        active_track_ids = set()
        
        for track in tracks:
            track_id = track.id
            active_track_ids.add(track_id)
            bbox = track.bbox
            
            # Check which zones the person is in
            for zone_id, zone in self.zones.items():
                is_in_zone = zone.contains_bbox(bbox)
                current_dwell = self.active_dwells[track_id].get(zone_id)
                
                if is_in_zone:
                    if current_dwell is None:
                        # Person entered zone
                        self.active_dwells[track_id][zone_id] = DwellRecord(
                            track_id=track_id,
                            zone_id=zone_id,
                            enter_time=current_time
                        )
                        self.zone_stats[zone_id]['total_visits'] += 1
                        print(f"Track {track_id} entered zone '{zone.name}'")
                else:
                    if current_dwell is not None:
                        # Person left zone
                        current_dwell.exit_time = current_time
                        self.completed_dwells.append(current_dwell)
                        self.zone_stats[zone_id]['total_dwell_time'] += current_dwell.duration
                        del self.active_dwells[track_id][zone_id]
                        print(f"Track {track_id} left zone '{zone.name}' "
                              f"(duration: {current_dwell.duration:.1f}s)")
        
        # Clean up tracks that no longer exist
        for track_id in list(self.active_dwells.keys()):
            if track_id not in active_track_ids:
                # Close out any active dwells
                for zone_id, dwell in list(self.active_dwells[track_id].items()):
                    dwell.exit_time = current_time
                    self.completed_dwells.append(dwell)
                    self.zone_stats[zone_id]['total_dwell_time'] += dwell.duration
                del self.active_dwells[track_id]
        
        # Update current occupancy
        for zone_id in self.zones:
            occupancy = sum(
                1 for track_dwells in self.active_dwells.values()
                if zone_id in track_dwells
            )
            self.zone_stats[zone_id]['current_occupancy'] = occupancy
    
    def get_dwell_times(self, track_id: Optional[int] = None) -> Dict:
        """Get dwell times for a track or all tracks"""
        if track_id:
            return {
                zone_id: record.duration
                for zone_id, record in self.active_dwells.get(track_id, {}).items()
            }
        
        # Return all active dwell times
        result = {}
        for track_id, zones in self.active_dwells.items():
            result[track_id] = {
                zone_id: record.duration
                for zone_id, record in zones.items()
            }
        return result
    
    def get_zone_analytics(self) -> Dict:
        """Get analytics for all zones"""
        analytics = {}
        for zone_id, zone in self.zones.items():
            stats = self.zone_stats[zone_id]
            avg_dwell = (stats['total_dwell_time'] / stats['total_visits'] 
                        if stats['total_visits'] > 0 else 0)
            
            analytics[zone_id] = {
                'name': zone.name,
                'total_visits': stats['total_visits'],
                'current_occupancy': stats['current_occupancy'],
                'total_dwell_time': stats['total_dwell_time'],
                'average_dwell_time': avg_dwell
            }
        return analytics
    
    def draw_zones(self, frame: np.ndarray, draw_labels: bool = True) -> np.ndarray:
        """Draw zones on frame"""
        for zone_id, zone in self.zones.items():
            # Draw polygon
            pts = np.array(zone.polygon, np.int32)
            pts = pts.reshape((-1, 1, 2))
            cv2.polylines(frame, [pts], True, zone.color, 2)
            
            if draw_labels:
                # Draw zone name
                centroid_x = int(np.mean([p[0] for p in zone.polygon]))
                centroid_y = int(np.mean([p[1] for p in zone.polygon]))
                
                label = f"{zone.name}"
                (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                cv2.rectangle(frame, 
                            (centroid_x - tw//2 - 5, centroid_y - th - 5),
                            (centroid_x + tw//2 + 5, centroid_y + 5),
                            zone.color, -1)
                cv2.putText(frame, label, 
                          (centroid_x - tw//2, centroid_y),
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        return frame
    
    def export_data(self, filepath: str):
        """Export dwell time data to JSON"""
        import json
        
        data = {
            'completed_dwells': [d.to_dict() for d in self.completed_dwells],
            'active_dwells': {
                str(tid): {
                    zid: d.to_dict() 
                    for zid, d in zones.items()
                }
                for tid, zones in self.active_dwells.items()
            },
            'zone_analytics': self.get_zone_analytics()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        
        print(f"Exported dwell data to {filepath}")


if __name__ == "__main__":
    # Test
    tracker = DwellTimeTracker()
    
    # Add a test zone (rectangle)
    tracker.add_zone("entrance", "Main Entrance", 
                    [(100, 100), (300, 100), (300, 300), (100, 300)])
    
    # Simulate tracks
    class FakeTrack:
        def __init__(self, id, bbox):
            self.id = id
            self.bbox = bbox
    
    tracks = [FakeTrack(1, [150, 150, 200, 250])]
    tracker.update(tracks)
    
    print(f"Active dwells: {tracker.get_dwell_times()}")
    print(f"Zone analytics: {tracker.get_zone_analytics()}")
