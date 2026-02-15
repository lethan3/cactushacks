#!/usr/bin/env python3
"""
Minimal camera capture script — tries all available methods to grab one frame.
Saves to workspace as camera_test_frame.png.

Usage:
    python3 camera_test_capture.py          # auto-detect
    python3 camera_test_capture.py 0        # force index 0
    python3 camera_test_capture.py 1        # force index 1
"""
import sys
import os
import cv2
from pathlib import Path
from datetime import datetime

OUT_DIR = Path(__file__).parent
OUT_FILE = OUT_DIR / "camera_test_frame.png"


def try_index(idx: int) -> bool:
    """Try to capture from a specific V4L2 camera index."""
    print(f"  Trying cv2.VideoCapture({idx}) ...")
    cap = cv2.VideoCapture(idx)
    if not cap.isOpened():
        print(f"    Could not open index {idx}")
        cap.release()
        return False

    # Set MJPG for better throughput
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    # Warm up (auto-exposure / auto-focus)
    for _ in range(5):
        cap.read()

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        print(f"    Opened index {idx} but read() failed")
        return False

    cv2.imwrite(str(OUT_FILE), frame)
    h, w = frame.shape[:2]
    print(f"    SUCCESS: {w}x{h} frame saved to {OUT_FILE}")
    return True


def probe_all() -> bool:
    """Scan /dev/video* and try each, then fall back to brute-force 0-9."""
    # Check device nodes first
    dev_videos = sorted(Path("/dev").glob("video*"))
    if dev_videos:
        print(f"Found device nodes: {[str(d) for d in dev_videos]}")
        for dev in dev_videos:
            idx = int(dev.name.replace("video", ""))
            if try_index(idx):
                return True
    else:
        print("No /dev/video* nodes found")

    # Brute-force indices 0-4
    print("Brute-force scanning indices 0-4 ...")
    for idx in range(5):
        if try_index(idx):
            return True

    return False


def main():
    print(f"Camera test capture — {datetime.now().isoformat()}")
    print(f"OpenCV {cv2.__version__}")
    print()

    if len(sys.argv) > 1:
        idx = int(sys.argv[1])
        print(f"Forced camera index: {idx}")
        if not try_index(idx):
            print("\nFAILED — camera not available at that index")
            sys.exit(1)
    else:
        print("Auto-detecting camera ...")
        if not probe_all():
            print("\nFAILED — no camera detected")
            print("  Check: ls /dev/video*")
            print("  Check: lsusb (look for webcam)")
            print("  Is the USB camera plugged in?")
            sys.exit(1)

    print(f"\nDone. Output: {OUT_FILE} ({os.path.getsize(OUT_FILE)} bytes)")


if __name__ == "__main__":
    main()
