"""
TEMI Robot SDK Bridge
Handles integration with TEMI's Android SDK
"""
import json
import time
from typing import Callable, Optional


class TEMIBridge:
    """
    Bridge between VideoPeopleAnalytics and TEMI SDK
    Handles person detection callbacks, speech, and movement
    """
    
    def __init__(self):
        self.person_detected_callback: Optional[Callable] = None
        self.person_lost_callback: Optional[Callable] = None
        self.dwell_alert_callback: Optional[Callable] = None
        
        # TEMI state
        self.is_temi_ready = False
        self.detected_persons = set()
        self.last_detection_time = 0
        
    def register_callbacks(self, 
                          on_person_detected=None,
                          on_person_lost=None,
                          on_dwell_alert=None):
        """Register TEMI SDK callbacks"""
        self.person_detected_callback = on_person_detected
        self.person_lost_callback = on_person_lost
        self.dwell_alert_callback = on_dwell_alert
    
    def on_person_detected(self, track_id, bbox, confidence):
        """Called when a person is detected"""
        if track_id not in self.detected_persons:
            self.detected_persons.add(track_id)
            self.last_detection_time = time.time()
            
            if self.person_detected_callback:
                self.person_detected_callback({
                    'track_id': track_id,
                    'bbox': bbox,
                    'confidence': confidence,
                    'timestamp': time.time()
                })
    
    def on_person_lost(self, track_id):
        """Called when a person is no longer detected"""
        if track_id in self.detected_persons:
            self.detected_persons.remove(track_id)
            
            if self.person_lost_callback:
                self.person_lost_callback({
                    'track_id': track_id,
                    'timestamp': time.time()
                })
    
    def on_dwell_alert(self, zone_id, zone_name, track_id, dwell_time):
        """Called when someone dwells too long in a zone"""
        if self.dwell_alert_callback:
            self.dwell_alert_callback({
                'zone_id': zone_id,
                'zone_name': zone_name,
                'track_id': track_id,
                'dwell_time': dwell_time,
                'timestamp': time.time()
            })
    
    def get_temi_js_bridge(self):
        """
        Returns JavaScript code for TEMI WebView integration
        This is injected into the web page to communicate with TEMI SDK
        """
        return """
        // TEMI SDK Bridge
        window.TEMI = {
            // Check if running on TEMI
            isTemi: function() {
                return typeof Android !== 'undefined' && Android.speak;
            },
            
            // Speak text
            speak: function(text) {
                if (this.isTemi() && Android.speak) {
                    Android.speak(text);
                } else {
                    console.log('TEMI Speak:', text);
                }
            },
            
            // Move to location
            goTo: function(location) {
                if (this.isTemi() && Android.goTo) {
                    Android.goTo(location);
                } else {
                    console.log('TEMI GoTo:', location);
                }
            },
            
            // Turn by degrees
            turnBy: function(degrees) {
                if (this.isTemi() && Android.turnBy) {
                    Android.turnBy(degrees);
                } else {
                    console.log('TEMI TurnBy:', degrees);
                }
            },
            
            // Tilt head
            tilt: function(angle) {
                if (this.isTemi() && Android.tilt) {
                    Android.tilt(angle);
                } else {
                    console.log('TEMI Tilt:', angle);
                }
            },
            
            // Get battery level
            getBattery: function() {
                if (this.isTemi() && Android.getBattery) {
                    return Android.getBattery();
                }
                return null;
            },
            
            // Person detected callback
            onPersonDetected: function(data) {
                // Send to parent/Android
                if (this.isTemi() && Android.onPersonDetected) {
                    Android.onPersonDetected(JSON.stringify(data));
                }
                
                // Default greeting for new person
                if (data.track_id === 1) {
                    this.speak("Hello! Welcome!");
                }
            },
            
            // Dwell alert callback
            onDwellAlert: function(data) {
                if (this.isTemi() && Android.onDwellAlert) {
                    Android.onDwellAlert(JSON.stringify(data));
                }
                
                // Default alert
                this.speak("Please don't block the " + data.zone_name);
            }
        };
        
        // Expose to global scope for WebView
        window.temiBridge = window.TEMI;
        """
    
    def get_status(self):
        """Get current bridge status"""
        return {
            'is_temi_ready': self.is_temi_ready,
            'detected_persons': list(self.detected_persons),
            'person_count': len(self.detected_persons),
            'last_detection': self.last_detection_time
        }
