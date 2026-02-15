# Camera Audit — 2026-02-15

## Goal

Capture a single live frame from the camera feed as a PNG.

## System

- **Board**: Jetson Orin Nano Super (Tegra234), R36 rev 4.4, JetPack 6.x
- **Kernel**: 5.15.148-tegra aarch64
- **OpenCV**: 4.13.0 (FFMPEG: YES, V4L2: YES, **GStreamer: NO**)
- **GStreamer**: 1.20.3 installed system-wide but NOT compiled into OpenCV

## Finding: no camera currently connected

### Evidence

| Check | Result |
|-------|--------|
| `ls /dev/video*` | No such file or directory — zero V4L2 video device nodes |
| `lsusb` | Hub, Bluetooth, Touchscreen — no USB webcam |
| `dmesg \| grep -i video/uvc/camera` | Empty — no camera driver messages in kernel log |
| `cv2.VideoCapture(i)` for i in 0–9 | All fail: "device_list->nb_devices is 0" |
| GStreamer nvarguscamerasrc pipeline via OpenCV | Fails (OpenCV compiled without GStreamer) |
| `/sys/class/video4linux/` | Directory missing or empty |
| `/proc/device-tree/cam*` | Debug nodes only (dbg, echo, diag) — no CSI sensor |

### USB devices present

```
Bus 002: 4-Port USB 3.0 Hub (Realtek 0bda:0489)
Bus 001: Bluetooth Radio (IMC Networks 13d3:3549)
Bus 001: ILITEK-TOUCH (ILI Technology 222a:0001)
Bus 001: 4-Port USB 2.0 Hub (Realtek 0bda:5489)
```

No webcam.

## Previous successful captures

A camera WAS working earlier today and has since been disconnected.

| File | Timestamp | Resolution | Size |
|------|-----------|------------|------|
| `webcam_photo.jpg` | 04:09 | 640×480 | 62 KB |
| `webcam_photo_jetson.jpg` | 04:14 | 640×480 | 63 KB |
| `20260215_051023.jpg` | 06:35 | 4080×2296 | 2.9 MB |

- 640×480 images match `capture.py` and `ActualCamera` settings (index 1, MJPG codec)
- 4080×2296 image suggests either a higher-res USB camera or different capture settings
- Pixel inspection confirms real photos (varied colors, not blank)

## Existing capture code

| File | Camera index | Resolution | Notes |
|------|-------------|------------|-------|
| `cactushacks/capture.py` | 1 | default (no explicit set) | Standalone script, saves `webcam_photo_jetson.jpg` |
| `cactushacks/motion/camera.py` `ActualCamera` | 1 | 640×480 forced | MJPG codec, 5-frame warmup, returns PIL Image |

Both use `cv2.VideoCapture(1)`.

## Conclusion (initial audit ~07:26)

Camera was disconnected at time of first audit. No `/dev/video*` nodes, no USB webcam in `lsusb`.

---

## Live capture — SUCCESS (07:28)

Camera reconnected by user. Full audit and capture completed.

### USB discovery

```
Bus 001 Device 005: ID 046d:08e5 Logitech, Inc. HD Pro Webcam C920
```

Connected via USB 2.0 hub at port 1-2.3.

### V4L2 device nodes

| Node | Name | Backend | Status |
|------|------|---------|--------|
| `/dev/video0` | HD Pro Webcam C920 | V4L2 | **Primary capture** — works |
| `/dev/video1` | HD Pro Webcam C920 | (metadata) | Not a capture source |

### Permissions

- Device: `crw-rw----` root:video
- User `cactushacks` is in `video` group — access OK

### Capture details

- **Index 0** (not 1 — existing code uses 1, which is wrong for current setup)
- Requested 1920×1080 MJPG, negotiated successfully at 30fps
- 10-frame warmup for auto-exposure
- Output: `/home/cactushacks/Documents/camera_live_frame.png` (1920×1080, 2.4 MB)
- Verified: 90 unique colors in 20×20 center patch — confirmed real photo
- Content: green seedlings in soil with pansy, inside clear container

### IMPORTANT: camera index mismatch

| File | Hardcoded index | Correct index now |
|------|----------------|-------------------|
| `cactushacks/capture.py` | `VideoCapture(1)` | Should be `0` |
| `cactushacks/motion/camera.py` `ActualCamera` | `CAMERA_INDEX = 1` | Should be `0` |

The C920 registers as `/dev/video0` + `/dev/video1` (metadata). The capture node is **video0 = index 0**. The old code used index 1, which may have worked when a different device order existed or a different camera was attached.

### System details

| Component | Value |
|-----------|-------|
| Camera | Logitech HD Pro Webcam C920 (046d:08e5) |
| USB path | Bus 001, port 1-2.3 (via Realtek 4-port USB 2.0 hub) |
| V4L2 nodes | video0 (capture), video1 (metadata) |
| OpenCV | 4.13.0 (V4L2: YES, FFMPEG: YES, GStreamer: NO) |
| Negotiated | 1920×1080 @ 30fps MJPG |
| Kernel | 5.15.148-tegra, R36 rev 4.4 |
