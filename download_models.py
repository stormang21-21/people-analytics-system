"""
Download required models for People Analytics System
"""
import os
import sys
from pathlib import Path

def download_yolo_models():
    """Download YOLO models"""
    try:
        from ultralytics import YOLO
    except ImportError:
        print("Error: ultralytics not installed. Run: pip install ultralytics")
        sys.exit(1)
    
    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    
    models_to_download = [
        ("yolov8n.pt", "YOLOv8 Nano (detection)"),
        ("yolov8n-pose.pt", "YOLOv8 Nano Pose (pose estimation)"),
    ]
    
    print("Downloading models...")
    print("-" * 50)
    
    for model_name, description in models_to_download:
        print(f"\nDownloading {description}...")
        try:
            model = YOLO(model_name)
            print(f"✓ {model_name} ready")
        except Exception as e:
            print(f"✗ Failed to download {model_name}: {e}")
    
    print("\n" + "-" * 50)
    print("Model download complete!")
    print(f"Models saved to: {models_dir.absolute()}")

if __name__ == "__main__":
    download_yolo_models()
