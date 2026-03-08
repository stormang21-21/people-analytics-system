#!/bin/bash
# Start People Analytics System with remote access

# Configuration
PORT=${PA_PORT:-5000}
HOST=${PA_HOST:-0.0.0.0}
USERNAME=${PA_USERNAME:-admin}
PASSWORD=${PA_PASSWORD:-admin123}

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  People Analytics System${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Get IP addresses
echo -e "${YELLOW}Network Information:${NC}"
echo "----------------------------------------"

# Get local IP
LOCAL_IP=$(ifconfig | grep "inet " | grep -v 127.0.0.1 | awk '{print $2}' | head -1)
if [ -z "$LOCAL_IP" ]; then
    LOCAL_IP=$(hostname -I | awk '{print $1}')
fi

echo "Local IP: $LOCAL_IP"
echo "Port: $PORT"
echo ""

echo -e "${YELLOW}Access URLs:${NC}"
echo "----------------------------------------"
echo -e "${GREEN}Local:${NC}     http://localhost:$PORT"
echo -e "${GREEN}Network:${NC}   http://$LOCAL_IP:$PORT"
echo ""

echo -e "${YELLOW}Authentication:${NC}"
echo "----------------------------------------"
echo "Username: $USERNAME"
echo "Password: $PASSWORD"
echo ""

echo -e "${YELLOW}Remote Access Options:${NC}"
echo "----------------------------------------"
echo "1. Same WiFi: Use Network URL above"
echo "2. Tailscale: Install + run 'tailscale up'"
echo "3. ngrok: Run 'ngrok http $PORT'"
echo ""

cd "$(dirname "$0")"

# Check if virtual environment exists
if [ -d "venv" ]; then
    source venv/bin/activate
fi

echo -e "${BLUE}Starting server...${NC}"
echo "Press Ctrl+C to stop"
echo ""

# Export auth credentials
export PA_USERNAME=$USERNAME
export PA_PASSWORD=$PASSWORD

python web/app.py
