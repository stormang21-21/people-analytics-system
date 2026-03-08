"""
Settings Manager - Handle feature toggles and configuration
"""
import json
import os
from typing import Dict, Any


class SettingsManager:
    """Manage system settings and feature toggles"""
    
    DEFAULT_SETTINGS = {
        'animals': True,
        'objects': True,
        'pose': True,
        'actions': True,
        'faces': True,
        'fall': True,
        'fight': True,
        'loiter': True,
        'dwell': True,
        'analytics': True,
        'confidence_threshold': 0.3
    }
    
    def __init__(self, config_file: str = "config/user_settings.json"):
        self.config_file = config_file
        self.settings = self.DEFAULT_SETTINGS.copy()
        self._load_settings()
    
    def _load_settings(self):
        """Load settings from file"""
        if os.path.exists(self.config_file):
            try:
                with open(self.config_file, 'r') as f:
                    loaded = json.load(f)
                    self.settings.update(loaded)
            except Exception as e:
                print(f"Failed to load settings: {e}")
    
    def save_settings(self):
        """Save settings to file"""
        try:
            os.makedirs(os.path.dirname(self.config_file), exist_ok=True)
            with open(self.config_file, 'w') as f:
                json.dump(self.settings, f, indent=2)
            return True
        except Exception as e:
            print(f"Failed to save settings: {e}")
            return False
    
    def get(self, key: str, default=None) -> Any:
        """Get a setting value"""
        return self.settings.get(key, default)
    
    def set(self, key: str, value: Any):
        """Set a setting value"""
        self.settings[key] = value
    
    def update(self, new_settings: Dict):
        """Update multiple settings"""
        self.settings.update(new_settings)
        return self.save_settings()
    
    def get_all(self) -> Dict:
        """Get all settings"""
        return self.settings.copy()
    
    def is_enabled(self, feature: str) -> bool:
        """Check if a feature is enabled"""
        return self.settings.get(feature, True)
    
    def get_active_features(self) -> list:
        """Get list of enabled features"""
        return [k for k, v in self.settings.items() if v and k != 'confidence_threshold']
