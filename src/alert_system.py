"""
Alert System - Generate alerts for anomalies
"""
import time
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, field
from enum import Enum


class AlertType(Enum):
    FALL = "fall"
    FIGHT = "fight"
    LOITERING = "loitering"
    INTRUSION = "intrusion"
    CROWD = "crowd"
    OBJECT_LEFT = "object_left"


@dataclass
class Alert:
    id: str
    type: AlertType
    message: str
    severity: str  # low, medium, high, critical
    timestamp: float
    track_id: Optional[int] = None
    zone_id: Optional[str] = None
    image: Optional[bytes] = None
    acknowledged: bool = False
    resolved: bool = False


class AlertSystem:
    """Manage alerts for the people analytics system"""
    
    def __init__(self, cooldown_period: float = 30.0):
        self.cooldown_period = cooldown_period
        self.alerts: List[Alert] = []
        self.last_alert_time: Dict[str, float] = {}
        self.alert_handlers: List[Callable] = []
        self.alert_counter = 0
    
    def add_handler(self, handler: Callable):
        """Add an alert handler callback"""
        self.alert_handlers.append(handler)
    
    def trigger(self, alert_type: AlertType, message: str, 
                severity: str = "medium", track_id: Optional[int] = None,
                zone_id: Optional[str] = None, image: Optional[bytes] = None) -> Optional[Alert]:
        """
        Trigger a new alert
        
        Returns:
            Alert object if triggered, None if on cooldown
        """
        # Check cooldown
        alert_key = f"{alert_type.value}_{track_id}_{zone_id}"
        current_time = time.time()
        
        if alert_key in self.last_alert_time:
            if current_time - self.last_alert_time[alert_key] < self.cooldown_period:
                return None  # Still on cooldown
        
        # Create alert
        self.alert_counter += 1
        alert = Alert(
            id=f"alert_{self.alert_counter}_{int(current_time)}",
            type=alert_type,
            message=message,
            severity=severity,
            timestamp=current_time,
            track_id=track_id,
            zone_id=zone_id,
            image=image
        )
        
        self.alerts.append(alert)
        self.last_alert_time[alert_key] = current_time
        
        # Notify handlers
        for handler in self.alert_handlers:
            try:
                handler(alert)
            except Exception as e:
                print(f"Alert handler error: {e}")
        
        print(f"🚨 ALERT [{severity.upper()}]: {message}")
        
        return alert
    
    def check_fall(self, track_id: int, action: str, image: Optional[bytes] = None):
        """Check for fall detection"""
        if action == "falling":
            return self.trigger(
                AlertType.FALL,
                f"Person {track_id} detected falling!",
                severity="critical",
                track_id=track_id,
                image=image
            )
        return None
    
    def check_fight(self, fighter_ids: List[int], image: Optional[bytes] = None):
        """Check for fight detection"""
        if fighter_ids:
            return self.trigger(
                AlertType.FIGHT,
                f"Potential fight detected between persons: {fighter_ids}",
                severity="critical",
                image=image
            )
        return None
    
    def check_loitering(self, track_id: int, zone_id: str, 
                        dwell_time: float, threshold: float = 300,
                        image: Optional[bytes] = None):
        """Check for loitering"""
        if dwell_time > threshold:
            minutes = int(dwell_time / 60)
            return self.trigger(
                AlertType.LOITERING,
                f"Person {track_id} loitering in zone {zone_id} for {minutes} minutes",
                severity="medium",
                track_id=track_id,
                zone_id=zone_id,
                image=image
            )
        return None
    
    def check_crowd(self, people_count: int, threshold: int = 20,
                    image: Optional[bytes] = None):
        """Check for crowd formation"""
        if people_count > threshold:
            return self.trigger(
                AlertType.CROWD,
                f"Crowd detected: {people_count} people",
                severity="high",
                image=image
            )
        return None
    
    def check_intrusion(self, track_id: int, zone_id: str,
                        image: Optional[bytes] = None):
        """Check for zone intrusion"""
        return self.trigger(
            AlertType.INTRUSION,
            f"Person {track_id} entered restricted zone {zone_id}",
            severity="high",
            track_id=track_id,
            zone_id=zone_id,
            image=image
        )
    
    def get_active_alerts(self, severity: Optional[str] = None) -> List[Alert]:
        """Get all active (unresolved) alerts"""
        alerts = [a for a in self.alerts if not a.resolved]
        
        if severity:
            alerts = [a for a in alerts if a.severity == severity]
        
        return sorted(alerts, key=lambda x: x.timestamp, reverse=True)
    
    def acknowledge_alert(self, alert_id: str) -> bool:
        """Acknowledge an alert"""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.acknowledged = True
                return True
        return False
    
    def resolve_alert(self, alert_id: str) -> bool:
        """Mark alert as resolved"""
        for alert in self.alerts:
            if alert.id == alert_id:
                alert.resolved = True
                return True
        return False
    
    def get_alert_stats(self) -> Dict:
        """Get alert statistics"""
        stats = {
            'total': len(self.alerts),
            'active': len([a for a in self.alerts if not a.resolved]),
            'acknowledged': len([a for a in self.alerts if a.acknowledged]),
            'by_type': {},
            'by_severity': {}
        }
        
        for alert in self.alerts:
            alert_type = alert.type.value
            stats['by_type'][alert_type] = stats['by_type'].get(alert_type, 0) + 1
            
            severity = alert.severity
            stats['by_severity'][severity] = stats['by_severity'].get(severity, 0) + 1
        
        return stats
    
    def cleanup_old_alerts(self, max_age: float = 86400):
        """Remove alerts older than max_age seconds (default 24 hours)"""
        current_time = time.time()
        self.alerts = [
            a for a in self.alerts 
            if current_time - a.timestamp < max_age or not a.resolved
        ]


if __name__ == "__main__":
    alerts = AlertSystem()
    print("Alert system ready")
    
    # Test
    alert = alerts.check_fall(1, "falling")
    if alert:
        print(f"Created alert: {alert.message}")
