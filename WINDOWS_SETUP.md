# Windows Setup Guide

## Quick Start

1. **Install Python 3.10+**
   - Download from: https://www.python.org/downloads/
   - ⚠️ **Important**: Check "Add Python to PATH" during installation

2. **Download this folder** to your Windows PC

3. **Double-click `start.bat`**
   - First run will install dependencies (takes a few minutes)
   - Subsequent runs are instant

4. **Open browser** to: `http://localhost:5000`

## Login
- Username: `admin`
- Password: `admin123`

## For Remote Access (ngrok)

1. **Download ngrok**: https://ngrok.com/download
2. **Extract** `ngrok.exe` to the same folder
3. **Get authtoken** from https://dashboard.ngrok.com/get-started/your-authtoken
4. **Run once**: `ngrok config add-authtoken YOUR_TOKEN`
5. **Double-click `start-ngrok.bat`**
6. **Copy the HTTPS URL** that appears and share it

## Troubleshooting

| Problem | Solution |
|---------|----------|
| "Python not found" | Reinstall Python, check "Add to PATH" |
| "pip not found" | Run: `python -m ensurepip` |
| Port 5000 in use | Edit `start.bat`, change `PA_PORT=5000` to `PA_PORT=5001` |
| Camera not connecting | Check RTSP URL format for your camera brand |
| Slow performance | Close other apps, reduce camera resolution |

## File Structure

```
people-analytics-system/
├── start.bat              ← Double-click to run
├── start-ngrok.bat        ← For remote access
├── requirements.txt       ← Python dependencies
├── web/
│   ├── app.py            ← Main server
│   └── templates/        ← Web UI files
└── src/                  ← Detection & tracking code
```

## System Requirements

- Windows 10/11
- 8GB RAM minimum (16GB recommended)
- Webcam or IP camera
- Internet connection (for ngrok remote access)

## Updating

To update to latest code:
1. Stop the server (Ctrl+C or close window)
2. Replace the folder with new version
3. Run `start.bat` again

## Support

For issues, check:
- Server logs in the black command window
- Camera compatibility list in README.md
- ngrok status at http://localhost:4040 (when using ngrok)
