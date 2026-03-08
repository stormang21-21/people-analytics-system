#!/bin/bash
# Start People Analytics System with ngrok tunnel

PORT=${PA_PORT:-5001}
USERNAME=${PA_USERNAME:-admin}
PASSWORD=${PA_PASSWORD:-admin123}

# Colors
GREEN='\033[0;32m'
BLUE='\033[0;34m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m'

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}  People Analytics + ngrok Tunnel${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Check if ngrok is installed
if ! command -v ngrok &> /dev/null; then
    echo -e "${RED}ngrok not found!${NC}"
    echo "Install with: brew install ngrok"
    echo "Then setup: ngrok config add-authtoken YOUR_TOKEN"
    exit 1
fi

# Check if authtoken is set
if ! ngrok config check &> /dev/null; then
    echo -e "${RED}ngrok not configured!${NC}"
    echo "Get token from: https://dashboard.ngrok.com/get-started/your-authtoken"
    echo "Then run: ngrok config add-authtoken YOUR_TOKEN"
    exit 1
fi

cd "$(dirname "$0")"

# Export auth credentials
export PA_USERNAME=$USERNAME
export PA_PASSWORD=$PASSWORD

echo -e "${YELLOW}Starting People Analytics Server...${NC}"
python3 web/app.py &
SERVER_PID=$!

# Wait for server to start
sleep 3

echo ""
echo -e "${YELLOW}Starting ngrok tunnel...${NC}"
echo ""

# Function to cleanup on exit
cleanup() {
    echo ""
    echo -e "${YELLOW}Shutting down...${NC}"
    kill $SERVER_PID 2>/dev/null
    pkill -f "ngrok http $PORT" 2>/dev/null
    exit 0
}

trap cleanup INT TERM

# Start ngrok and capture URL
echo -e "${GREEN}Public URL will appear below:${NC}"
echo "----------------------------------------"
ngrok http $PORT --log=stdout &
NGROK_PID=$!

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${YELLOW}Access Information:${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "Local:    ${GREEN}http://localhost:$PORT${NC}"
echo -e "Public:   ${GREEN}https://xxxx.ngrok.io${NC} (check above)"
echo ""
echo -e "Username: ${YELLOW}$USERNAME${NC}"
echo -e "Password: ${YELLOW}$PASSWORD${NC}"
echo ""
echo -e "${BLUE}========================================${NC}"
echo "Press Ctrl+C to stop"
echo ""

# Keep script running
wait
