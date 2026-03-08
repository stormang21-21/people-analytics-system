"""
Analytics Module - Data aggregation and reporting
"""
import json
import time
from typing import Dict, List, Optional
from datetime import datetime, timedelta
from collections import defaultdict
import csv
import os


class Analytics:
    """Analytics data aggregation and reporting"""
    
    def __init__(self, data_dir: str = "data"):
        self.data_dir = data_dir
        os.makedirs(data_dir, exist_ok=True)
        
        self.hourly_data = defaultdict(lambda: {
            'people_count': [],
            'zone_occupancy': defaultdict(list),
            'dwell_times': defaultdict(list)
        })
        
        self.daily_stats = {}
        self.current_hour = datetime.now().hour
    
    def record(self, timestamp: float, people_count: int, 
               zone_occupancy: Dict, dwell_times: Dict):
        """Record analytics data point"""
        hour = datetime.fromtimestamp(timestamp).hour
        hour_key = datetime.fromtimestamp(timestamp).strftime('%Y-%m-%d_%H')
        
        self.hourly_data[hour_key]['people_count'].append(people_count)
        
        for zone_id, count in zone_occupancy.items():
            self.hourly_data[hour_key]['zone_occupancy'][zone_id].append(count)
        
        for track_id, dwell_data in dwell_times.items():
            for zone_id, duration in dwell_data.items():
                self.hourly_data[hour_key]['dwell_times'][zone_id].append(duration)
    
    def get_current_stats(self) -> Dict:
        """Get current hour statistics"""
        hour_key = datetime.now().strftime('%Y-%m-%d_%H')
        data = self.hourly_data[hour_key]
        
        return {
            'hour': hour_key,
            'avg_people_count': self._avg(data['people_count']),
            'max_people_count': max(data['people_count']) if data['people_count'] else 0,
            'total_detections': len(data['people_count']),
            'zone_stats': {
                zone_id: {
                    'avg_occupancy': self._avg(counts),
                    'max_occupancy': max(counts) if counts else 0
                }
                for zone_id, counts in data['zone_occupancy'].items()
            },
            'dwell_stats': {
                zone_id: {
                    'avg_dwell': self._avg(times),
                    'max_dwell': max(times) if times else 0
                }
                for zone_id, times in data['dwell_times'].items()
            }
        }
    
    def get_daily_report(self, date: Optional[str] = None) -> Dict:
        """Get daily analytics report"""
        if date is None:
            date = datetime.now().strftime('%Y-%m-%d')
        
        day_data = {
            'people_count': [],
            'zone_occupancy': defaultdict(list),
            'dwell_times': defaultdict(list)
        }
        
        for hour in range(24):
            hour_key = f"{date}_{hour:02d}"
            if hour_key in self.hourly_data:
                data = self.hourly_data[hour_key]
                day_data['people_count'].extend(data['people_count'])
                
                for zone_id, counts in data['zone_occupancy'].items():
                    day_data['zone_occupancy'][zone_id].extend(counts)
                
                for zone_id, times in data['dwell_times'].items():
                    day_data['dwell_times'][zone_id].extend(times)
        
        return {
            'date': date,
            'avg_people_count': self._avg(day_data['people_count']),
            'max_people_count': max(day_data['people_count']) if day_data['people_count'] else 0,
            'total_detections': len(day_data['people_count']),
            'peak_hour': self._get_peak_hour(date),
            'zone_stats': {
                zone_id: {
                    'avg_occupancy': self._avg(counts),
                    'total_visits': len(counts)
                }
                for zone_id, counts in day_data['zone_occupancy'].items()
            }
        }
    
    def export_csv(self, filename: Optional[str] = None) -> str:
        """Export analytics data to CSV"""
        if filename is None:
            filename = f"analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
        
        filepath = os.path.join(self.data_dir, filename)
        
        with open(filepath, 'w', newline='') as f:
            writer = csv.writer(f)
            writer.writerow(['Hour', 'Avg People', 'Max People', 'Zone', 'Avg Occupancy'])
            
            for hour_key in sorted(self.hourly_data.keys()):
                data = self.hourly_data[hour_key]
                avg_people = self._avg(data['people_count'])
                max_people = max(data['people_count']) if data['people_count'] else 0
                
                for zone_id, counts in data['zone_occupancy'].items():
                    avg_occ = self._avg(counts)
                    writer.writerow([hour_key, avg_people, max_people, zone_id, avg_occ])
        
        return filepath
    
    def _avg(self, values: List[float]) -> float:
        """Calculate average"""
        return sum(values) / len(values) if values else 0.0
    
    def _get_peak_hour(self, date: str) -> int:
        """Find peak hour for a date"""
        max_count = 0
        peak_hour = 0
        
        for hour in range(24):
            hour_key = f"{date}_{hour:02d}"
            if hour_key in self.hourly_data:
                count = len(self.hourly_data[hour_key]['people_count'])
                if count > max_count:
                    max_count = count
                    peak_hour = hour
        
        return peak_hour
    
    def save(self):
        """Save analytics data to file"""
        filepath = os.path.join(self.data_dir, 'analytics.json')
        
        # Convert defaultdict to regular dict for JSON serialization
        data = {
            'hourly_data': dict(self.hourly_data),
            'saved_at': time.time()
        }
        
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    def load(self):
        """Load analytics data from file"""
        filepath = os.path.join(self.data_dir, 'analytics.json')
        
        if os.path.exists(filepath):
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            # Restore hourly data
            for hour_key, hour_data in data.get('hourly_data', {}).items():
                self.hourly_data[hour_key] = hour_data


if __name__ == "__main__":
    analytics = Analytics()
    print("Analytics module ready")
