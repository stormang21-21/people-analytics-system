"""
Lightweight dwell time tracker for TEMI
Tracks time spent in defined zones
"""
import time
from collections import defaultdict


class TEMIDwellTracker:
    """Track how long people stay in specific zones"""
    
    def __init__(self):
        self.zones = {}  # zone_id -> {name, polygon, color}
        self.zone_entries = defaultdict(dict)  # zone_id -> {track_id -> entry_time}
        self.dwell_times = defaultdict(lambda: defaultdict(float))  # zone_id -> {track_id -> total_time}
        self.current_zone = {}  # track_id -> zone_id (which zone they're currently in)
        
    def add_zone(self, zone_id, name, polygon, color="#4a9eff"):
        """
        Add a zone
        polygon: list of (x, y) points defining the zone
        """
        self.zones[zone_id] = {
            'name': name,
            'polygon': polygon,
            'color': color
        }
    
    def remove_zone(self, zone_id):
        """Remove a zone"""
        if zone_id in self.zones:
            del self.zones[zone_id]
            if zone_id in self.zone_entries:
                del self.zone_entries[zone_id]
    
    def update(self, track_ids, bboxes):
        """
        Update dwell tracking based on current positions
        track_ids: list of track IDs
        bboxes: list of [x1, y1, x2, y2]
        """
        current_time = time.time()
        
        # Check which zone each person is in
        for track_id, bbox in zip(track_ids, bboxes):
            center_x = (bbox[0] + bbox[2]) / 2
            center_y = (bbox[1] + bbox[3]) / 2
            
            # Find which zone they're in (if any)
            in_zone = None
            for zone_id, zone in self.zones.items():
                if self._point_in_polygon(center_x, center_y, zone['polygon']):
                    in_zone = zone_id
                    break
            
            # Handle zone transitions
            prev_zone = self.current_zone.get(track_id)
            
            if in_zone != prev_zone:
                # Left previous zone
                if prev_zone is not None and track_id in self.zone_entries[prev_zone]:
                    entry_time = self.zone_entries[prev_zone][track_id]
                    self.dwell_times[prev_zone][track_id] += current_time - entry_time
                    del self.zone_entries[prev_zone][track_id]
                
                # Entered new zone
                if in_zone is not None:
                    self.zone_entries[in_zone][track_id] = current_time
                
                self.current_zone[track_id] = in_zone
        
        # Clean up tracks that disappeared
        active_tracks = set(track_ids)
        for track_id in list(self.current_zone.keys()):
            if track_id not in active_tracks:
                prev_zone = self.current_zone[track_id]
                if prev_zone is not None and track_id in self.zone_entries[prev_zone]:
                    entry_time = self.zone_entries[prev_zone][track_id]
                    self.dwell_times[prev_zone][track_id] += time.time() - entry_time
                    del self.zone_entries[prev_zone][track_id]
                del self.current_zone[track_id]
    
    def get_zone_occupancy(self):
        """Get current number of people in each zone"""
        occupancy = {}
        for zone_id in self.zones:
            occupancy[zone_id] = len(self.zone_entries[zone_id])
        return occupancy
    
    def get_dwell_times(self, zone_id=None):
        """Get dwell times, optionally filtered by zone"""
        if zone_id:
            return dict(self.dwell_times[zone_id])
        return {z: dict(t) for z, t in self.dwell_times.items()}
    
    def get_average_dwell(self, zone_id):
        """Get average dwell time for a zone"""
        times = list(self.dwell_times[zone_id].values())
        # Add current dwell times for people still in zone
        current_time = time.time()
        for track_id, entry_time in self.zone_entries[zone_id].items():
            times.append(current_time - entry_time)
        
        return sum(times) / len(times) if times else 0
    
    def reset(self):
        """Reset all tracking data"""
        self.zone_entries.clear()
        self.dwell_times.clear()
        self.current_zone.clear()
    
    def _point_in_polygon(self, x, y, polygon):
        """Ray casting algorithm to check if point is in polygon"""
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
