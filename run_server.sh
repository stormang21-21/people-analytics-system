#!/bin/bash
# Run People Analytics Server with Webcam

cd ~/people-analytics-system

# Activate virtual environment
source venv/bin/activate

# Set port
export PA_PORT=5051

# Run server
echo "Starting People Analytics Server..."
echo "Open http://localhost:5051 in your browser"
echo ""
echo "If a camera permission dialog appears, click 'Allow'"
echo ""
python3 web/app.py
