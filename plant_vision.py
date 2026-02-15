#!/usr/bin/env python3
"""
Plant vision DSL — pure function that captures a camera frame and returns
a text description of plant attributes from a cloud VLM.

DSL function:
    analyze_camera() -> str      # capture + analyze, returns text
    analyze_image(path) -> str   # analyze an existing image, returns text

The return value is always a plain-text string (never an image, never raw JSON)
suitable for feeding directly into a local language model as tool-call output.

Test in isolation:
    python3 plant_vision.py                        # capture from camera
    python3 plant_vision.py /path/to/image.png     # use existing image
"""

import base64
import json
import time
import tempfile
from pathlib import Path

import cv2
from together import Together

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
MODEL = "Qwen/Qwen3-VL-8B-Instruct"
API_KEY = "tgp_v1_-dP1hu9uR4IPgLfsOo5oXY0hcwwDQa9szUz9QrQ0bvI"

PROMPT = """You are a botanist analyzing a photo from a live plant-monitoring camera.
Identify every plant visible and return a JSON array. For each plant, include:

{
  "species_guess": "best guess at species or cultivar",
  "common_name": "common name",
  "leaf_color": "describe the color(s)",
  "health_score": 0-100,
  "health_notes": "brief description of visible condition",
  "needs_water": true/false,
  "care_recommendation": "one actionable sentence"
}

Return ONLY the JSON array, no markdown fences, no explanation."""


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------
def _capture_frame() -> Path:
    """Grab one frame from the USB camera, save to a temp file, return path."""
    cap = cv2.VideoCapture(1)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    if not cap.isOpened():
        # fallback: try index 0
        cap = cv2.VideoCapture(0)
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*"MJPG"))

    if not cap.isOpened():
        raise RuntimeError("No camera available (tried indices 1, 0)")

    # warm-up (auto-exposure / auto-focus)
    for _ in range(5):
        cap.read()

    ret, frame = cap.read()
    cap.release()

    if not ret or frame is None:
        raise RuntimeError("Camera opened but read() returned no frame")

    tmp = Path(tempfile.gettempdir()) / "plant_vision_frame.jpg"
    cv2.imwrite(str(tmp), frame)
    return tmp


def _encode_image(path: Path) -> tuple[str, str]:
    """Return (base64_string, mime_type) for an image file."""
    b64 = base64.b64encode(path.read_bytes()).decode("utf-8")
    mime = "image/png" if path.suffix.lower() == ".png" else "image/jpeg"
    return b64, mime


def _call_vlm(b64: str, mime: str) -> tuple[list[dict], float]:
    """Send base64 image to Together AI, return (parsed_plants, elapsed_sec)."""
    client = Together(api_key=API_KEY)
    t0 = time.perf_counter()

    response = client.chat.completions.create(
        model=MODEL,
        max_tokens=512,
        temperature=0.2,
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": PROMPT},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{mime};base64,{b64}"},
                    },
                ],
            }
        ],
    )

    elapsed = time.perf_counter() - t0
    raw = response.choices[0].message.content.strip()

    # strip markdown fences if model included them
    if raw.startswith("```"):
        lines = raw.split("\n")
        raw = "\n".join(lines[1:-1])

    plants = json.loads(raw)
    if isinstance(plants, dict):
        plants = [plants]

    return plants, elapsed


def _format_text(plants: list[dict], elapsed: float) -> str:
    """Turn the structured plant list into a human-readable text block."""
    lines = [f"Plant analysis ({elapsed:.1f}s):"]
    for i, p in enumerate(plants, 1):
        lines.append(f"  Plant {i}: {p.get('common_name', '?')} ({p.get('species_guess', '?')})")
        lines.append(f"    Leaf color: {p.get('leaf_color', '?')}")
        lines.append(f"    Health: {p.get('health_score', '?')}/100 — {p.get('health_notes', '?')}")
        lines.append(f"    Needs water: {'yes' if p.get('needs_water') else 'no'}")
        lines.append(f"    Recommendation: {p.get('care_recommendation', '?')}")
    return "\n".join(lines)


# ---------------------------------------------------------------------------
# DSL public functions
# ---------------------------------------------------------------------------
def analyze_image(image_path: str | Path) -> str:
    """Analyze an existing image file. Returns a plain-text report string."""
    path = Path(image_path)
    if not path.exists():
        return f"error: image not found: {path}"
    b64, mime = _encode_image(path)
    plants, elapsed = _call_vlm(b64, mime)
    return _format_text(plants, elapsed)


def analyze_camera() -> str:
    """Capture one frame from the camera, analyze it, return plain-text report."""
    frame_path = _capture_frame()
    return analyze_image(frame_path)


# ---------------------------------------------------------------------------
# Standalone test harness
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys

    if len(sys.argv) > 1:
        path = sys.argv[1]
        print(f"[test] analyzing image: {path}")
        result = analyze_image(path)
    else:
        print("[test] capturing from camera...")
        try:
            result = analyze_camera()
        except RuntimeError as e:
            print(f"[test] camera capture failed: {e}")
            # fallback: try the default live frame
            default = Path(__file__).parent.parent / "camera_live_frame.png"
            if default.exists():
                print(f"[test] falling back to {default}")
                result = analyze_image(default)
            else:
                print("[test] no fallback image available")
                sys.exit(1)

    print()
    print(result)
