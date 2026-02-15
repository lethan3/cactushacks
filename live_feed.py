#!/usr/bin/env python3
"""Live camera feed on the left half of the touchscreen. Press 'q' or ESC to quit."""
import cv2
import os
import glob
import subprocess

os.environ.setdefault('DISPLAY', ':0')
os.environ.setdefault('DBUS_SESSION_BUS_ADDRESS', 'unix:path=/run/user/1000/bus')

DOCK_KEYS = {
    'autohide': None,
    'dock-fixed': None,
    'intellihide': None,
}
SCHEMA = 'org.gnome.shell.extensions.dash-to-dock'


def gsettings(action, key, value=None):
    cmd = ['gsettings', action, SCHEMA, key]
    if value is not None:
        cmd.append(value)
    return subprocess.run(cmd, capture_output=True, text=True).stdout.strip()


def hide_dock():
    """Save current dock state and hide it."""
    for key in DOCK_KEYS:
        DOCK_KEYS[key] = gsettings('get', key)
    gsettings('set', 'autohide', 'false')
    gsettings('set', 'dock-fixed', 'false')
    gsettings('set', 'intellihide', 'false')
    print("Dock hidden")


def restore_dock():
    """Restore dock to its original state."""
    for key, val in DOCK_KEYS.items():
        if val is not None:
            gsettings('set', key, val)
    print("Dock restored")


def find_camera():
    # Extract index from /dev/videoN and open by index (V4L2 needs int)
    for dev in sorted(glob.glob('/dev/video*')):
        idx = int(dev.replace('/dev/video', ''))
        cap = cv2.VideoCapture(idx)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                print(f"Found camera at {dev} (index {idx})")
                return cap
            cap.release()
    raise RuntimeError("No camera found")


# 1. Hide dock first
hide_dock()

# 2. Open camera
cap = find_camera()
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# 3. Create window on left half (1024x600 screen → 512x600 left half)
WIN = "Live Feed"
cv2.namedWindow(WIN, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN, 512, 600)
cv2.moveWindow(WIN, 0, 0)

print("Streaming to left half — press q or ESC to quit")

try:
    while True:
        ret, frame = cap.read()
        if not ret:
            continue
        cv2.imshow(WIN, frame)
        key = cv2.waitKey(1) & 0xFF
        if key in (ord('q'), 27):
            break
finally:
    # 4. Always restore dock on exit
    cap.release()
    cv2.destroyAllWindows()
    restore_dock()
