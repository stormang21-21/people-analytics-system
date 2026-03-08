# People Analytics System

Comprehensive video analytics with person detection, tracking, action recognition, and dwell time analysis.

## Features

- **Real-time Detection**: YOLO26 for person + 80 object classes
- **Multi-Object Tracking**: ByteTrack for persistent IDs
- **Pose Estimation**: 17 keypoints for posture analysis
- **Action Recognition**: Complex actions (texting, fighting, falling)
- **Object Interaction**: Detect what people are holding/using
- **Dwell Time**: Track time spent in zones
- **Web UI**: Configure IP cameras, view live feed, analytics dashboard
- **Alerts**: Anomaly detection (fighting, falls, loitering)

## Architecture

```
people-analytics-system/
├── src/
│   ├── detector.py      # YOLO26 detection wrapper
│   ├── tracker.py       # ByteTrack multi-object tracking
│   ├── pose_estimator.py # Pose keypoints extraction
│   ├── action_classifier.py # Action recognition
│   ├── interaction_detector.py # Object-person interaction
│   ├── dwell_tracker.py # Time-in-zone tracking
│   ├── analytics.py     # Data aggregation & reporting
│   └── alert_system.py  # Anomaly alerts
├── web/
│   ├── app.py          # Flask/FastAPI web server
│   ├── camera_manager.py # IP camera configuration
│   ├── routes.py       # API endpoints
│   └── websocket.py    # Real-time updates
├── static/             # CSS, JS, assets
├── templates/          # HTML templates
├── config/             # Configuration files
└── models/             # Downloaded model weights
```

## Quick Start

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Download models:
   ```bash
   python download_models.py
   ```

3. Configure cameras in `config/cameras.yaml`

4. Start the system:
   ```bash
   python web/app.py
   ```

5. Open browser: `http://localhost:5000`

## Web UI Features

- **Dashboard**: Live view, statistics, alerts
- **Camera Config**: Add/edit IP cameras (RTSP/HTTP)
- **Zone Editor**: Draw dwell time zones
- **Analytics**: Historical data, heatmaps, reports
- **Settings**: Detection thresholds, alert rules

## Requirements

- Python 3.10+
- CUDA-capable GPU (recommended)
- 8-16GB RAM
- IP cameras with RTSP/HTTP streams
