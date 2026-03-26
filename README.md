# VideoPeopleAnalytics_Temi

Lightweight people analytics optimized for TEMI robot. Runs in TEMI WebView with minimal resource usage.

## TEMI-Specific Optimizations

- **Lightweight YOLO**: YOLOv8n (nano) instead of YOLO26
- **Simplified Tracking**: Basic IOU tracker instead of ByteTrack
- **No Pose Estimation**: Removed to save compute
- **No Action Recognition**: Removed to save compute
- **WebRTC Camera**: Uses TEMI's getUserMedia()
- **Touch-Optimized UI**: Large buttons for TEMI tablet
- **TEMI SDK Integration**: Movement, speech, person detection callbacks

## Architecture

```
VideoPeopleAnalytics_Temi/
├── src/
│   ├── detector.py          # YOLOv8n person detection only
│   ├── tracker.py           # Simple IOU tracker
│   ├── dwell_tracker.py     # Time-in-zone tracking
│   ├── analytics.py         # Basic analytics
│   └── temi_bridge.py       # TEMI SDK integration
├── web/
│   ├── app.py               # Flask server
│   ├── templates/
│   │   └── index.html       # TEMI-optimized UI
│   └── static/
├── temi_sdk/
│   └── temi_wrapper.py      # Android/TEMI SDK wrapper
└── config/
    └── temi_config.yaml     # TEMI-specific settings
```

## Quick Start

```bash
pip install -r requirements-temi.txt
python web/app.py
```

## TEMI Deployment

1. Host on server accessible to TEMI
2. Configure TEMI WebView to load URL
3. Grant camera permissions
4. TEMI SDK handles movement/speech

## Requirements

- Python 3.8+
- 2GB RAM minimum
- No GPU required (CPU-only mode)
- TEMI robot with WebView support
