import cv2

def take_picture():
    # Initialize the camera (0 is the default USB camera index)
    cap = cv2.VideoCapture(1)
    
    # Optional: Set resolution to 1080p (1920x1080)
    # Without this, it often defaults to 640x480
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
    
    # Optional: Set codec to MJPG for better framerate at high res
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))

    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return

    # Warm up the camera (skip first few frames for auto-focus/exposure adjustment)
    for _ in range(5):
        cap.read()

    # Capture a single frame
    ret, frame = cap.read()

    if ret:
        # Save the image
        filename = "webcam_photo_jetson.jpg"
        cv2.imwrite(filename, frame)
        print(f"Success: Image saved as {filename}")
    else:
        print("Error: Failed to capture image.")

    # Release the camera
    cap.release()

if __name__ == "__main__":
    take_picture()
