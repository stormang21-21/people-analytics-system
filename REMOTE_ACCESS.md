# Remote Access Configuration for People Analytics System

## Current Setup

The server already binds to `0.0.0.0:5000`, making it accessible from any device on your network.

## Access URLs

| Location | URL |
|----------|-----|
| Local machine | `http://localhost:5000` |
| Same network | `http://YOUR_IP:5000` |
| Remote (with tunnel) | `https://your-domain.com` |

## Security Options

### Option 1: Basic Authentication (Simple)

Add login to the web app.

### Option 2: Tailscale (Recommended)

Zero-config VPN — access from anywhere securely.

```bash
# Install Tailscale
brew install tailscale  # macOS
# or
curl -fsSL https://tailscale.com/install.sh | sh  # Linux

# Start Tailscale
tailscale up

# Get your Tailscale IP
tailscale ip -4
# Example: 100.x.x.x
```

Then access via: `http://100.x.x.x:5000` from any device logged into your Tailscale network.

### Option 3: Cloudflare Tunnel (Public access)

Expose to internet with HTTPS:

```bash
# Install cloudflared
brew install cloudflared

# Create tunnel
cloudflared tunnel create people-analytics

# Route tunnel
cd ~/.openclaw/workspace/people-analytics-system
cloudflared tunnel route dns people-analytics analytics.yourdomain.com

# Run tunnel
cloudflared tunnel run people-analytics
```

### Option 4: ngrok (Quick temporary)

```bash
# Install ngrok
brew install ngrok

# Start tunnel
ngrok http 5000

# Get public URL: https://xxxx.ngrok.io
```

## Security Checklist

- [ ] Enable authentication (see below)
- [ ] Use HTTPS for remote access
- [ ] Set strong password
- [ ] Restrict camera access if needed
- [ ] Consider firewall rules

## Enable Authentication

Edit `web/app.py` and add before routes:

```python
from functools import wraps
from flask import request, Response

# Simple auth
USERNAME = 'admin'
PASSWORD = 'your-secure-password'

def check_auth(username, password):
    return username == USERNAME and password == PASSWORD

def authenticate():
    return Response('Login required', 401, 
                   {'WWW-Authenticate': 'Basic realm="People Analytics"'})

def requires_auth(f):
    @wraps(f)
    def decorated(*args, **kwargs):
        auth = request.authorization
        if not auth or not check_auth(auth.username, auth.password):
            return authenticate()
        return f(*args, **kwargs)
    return decorated

# Add @requires_auth to routes
```

## Find Your IP

```bash
# macOS
ifconfig | grep "inet " | grep -v 127.0.0.1

# Or
ipconfig getifaddr en0  # WiFi
ipconfig getifaddr en1  # Ethernet
```

## Test Remote Access

1. Start the server: `python web/app.py`
2. Find your IP: `192.168.1.xxx`
3. From another device on same network: `http://192.168.1.xxx:5000`

## Troubleshooting

| Issue | Fix |
|-------|-----|
| Connection refused | Check firewall: `sudo ufw allow 5000` |
| Slow video | Reduce resolution in detector.py |
| No camera feed | Check camera URL is accessible from server |

## Production Deployment

For 24/7 remote access, consider:

1. **Docker container** with auto-restart
2. **Reverse proxy** (nginx/caddy) with SSL
3. **Systemd service** for auto-start on boot

Want me to set up any of these options?
