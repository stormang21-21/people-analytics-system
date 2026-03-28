#!/bin/bash
# Run Python with camera entitlements

cd ~/people-analytics-system

# Export camera permission environment
export OPENCV_AVFOUNDATION_SKIP_AUTH=0

# Run the server
PA_PORT=5050 python3 web/app.py
