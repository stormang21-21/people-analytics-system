import cv2
import time

print("Testing camera access...")
cap = cv2.VideoCapture(0)

if cap.isOpened():
    print("SUCCESS: Camera is accessible!")
    ret, frame = cap.read()
    if ret:
        print(f"Frame captured: {frame.shape}")
    else:
        print("Could not read frame")
    cap.release()
else:
    print("FAILED: Camera not accessible")

print("Done!")
input("Press Enter to exit...")

