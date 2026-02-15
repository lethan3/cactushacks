#!/usr/bin/env python3
"""
Camera tester for simulating camera movement and capturing subimages.
Maintains current position (cx, cy) and allows movement in all directions.
"""
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional
from abc import ABC, abstractmethod


class Camera(ABC):
    """Base camera class with movement and picture-taking capabilities."""
    
    # Default image dimensions
    DEFAULT_WIDTH = 640
    DEFAULT_HEIGHT = 480
    
    # Inch per pixel of width constant
    # Calculated as: (10/4 * 0.957) inches / 1185 pixels = 2.3925 / 1185
    INCH_PER_PIXEL_OF_WIDTH = (10 / 4 * 0.957) / 1185
    
    # Velocity constant: 2 inches per second
    VELOCITY_INCHES_PER_SEC = 2.0
    
    @abstractmethod
    def move_right(self, pixels: int = 1) -> Tuple[int, int]:
        """
        Move camera right (positive) or left (negative).
        
        Args:
            pixels: Number of pixels to move (positive = right, negative = left)
            
        Returns:
            New (x, y) position
        """
        pass
    
    @abstractmethod
    def move_up(self, pixels: int = 1) -> Tuple[int, int]:
        """
        Move camera up (positive) or down (negative).
        
        Args:
            pixels: Number of pixels to move (positive = up, negative = down)
            
        Returns:
            New (x, y) position
        """
        pass
    
    @abstractmethod
    def take_picture(self) -> Image.Image:
        """
        Take a picture with the camera.
        
        Returns:
            PIL Image captured by the camera
        """
        pass


class CameraTester(Camera):
    """Simulates a camera that can move and capture subimages from test_borders.png."""
    
    def __init__(self, image_path: str = "test_borders.png", 
                 initial_x: int = 0, initial_y: int = 0):
        """
        Initialize the camera tester.
        
        Args:
            image_path: Path to the test image (test_borders.png)
            initial_x: Initial x coordinate (default: 0)
            initial_y: Initial y coordinate (default: 0)
        """
        self.image_path = Path(image_path)
        if not self.image_path.exists():
            raise FileNotFoundError(f"Image not found: {self.image_path}")
        
        self.image = Image.open(self.image_path)
        self.image_width, self.image_height = self.image.size
        
        # Subimages scaled down by 4x (was 200x200, now 50x50)
        self.subimage_width = 50
        self.subimage_height = 50
        
        # Current camera position (upper-left corner of viewport)
        self.cx = initial_x
        self.cy = initial_y
        
        # Ensure initial position is within bounds
        self.cx = max(0, min(self.cx, self.image_width - 1))
        self.cy = max(0, min(self.cy, self.image_height - 1))
    
    def move_right(self, pixels: int = 1) -> Tuple[int, int]:
        """
        Move camera right (positive) or left (negative).
        
        Args:
            pixels: Number of pixels to move (positive = right, negative = left)
            
        Returns:
            New (cx, cy) position
        """
        self.cx += pixels
        self.cx = max(0, min(self.cx, self.image_width - 1))
        return (self.cx, self.cy)
    
    def move_up(self, pixels: int = 1) -> Tuple[int, int]:
        """
        Move camera up (positive) or down (negative).
        
        Args:
            pixels: Number of pixels to move (positive = up, negative = down)
            
        Returns:
            New (cx, cy) position
        """
        self.cy -= pixels  # Y increases downward, so up is negative
        self.cy = max(0, min(self.cy, self.image_height - 1))
        return (self.cx, self.cy)
    
    def take_picture(self) -> Image.Image:
        """
        Take a picture with the camera (returns subimage at current position).
        
        Returns:
            PIL Image of the subimage
        """
        return self.get_subimage()
    
    def move_left(self, pixels: int = 1) -> Tuple[int, int]:
        """Move camera left. Convenience method for move_right(-pixels)."""
        return self.move_right(-pixels)
    
    def move_down(self, pixels: int = 1) -> Tuple[int, int]:
        """Move camera down. Convenience method for move_up(-pixels)."""
        return self.move_up(-pixels)
    
    def move(self, dx: int = 0, dy: int = 0) -> Tuple[int, int]:
        """
        Move camera by delta x and delta y.
        
        Args:
            dx: Change in x (positive = right, negative = left)
            dy: Change in y (positive = down, negative = up)
            
        Returns:
            New (cx, cy) position
        """
        self.cx += dx
        self.cy -= dy  # Y increases downward, so up is negative
        self.cx = max(0, min(self.cx, self.image_width - 1))
        self.cy = max(0, min(self.cy, self.image_height - 1))
        return (self.cx, self.cy)
    
    def set_position(self, x: int, y: int) -> Tuple[int, int]:
        """
        Set camera position directly.
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            New (cx, cy) position (clamped to image bounds)
        """
        self.cx = max(0, min(x, self.image_width - 1))
        self.cy = max(0, min(y, self.image_height - 1))
        return (self.cx, self.cy)
    
    def get_position(self) -> Tuple[int, int]:
        """Get current camera position."""
        return (self.cx, self.cy)
    
    def get_subimage(self) -> Image.Image:
        """
        Get a subimage using self.subimage_width and self.subimage_height with upper-left corner at (cx, cy).
        
        Returns:
            PIL Image of the subimage (may be smaller if near image edges)
        """
        # Calculate bounds
        x1 = self.cx
        y1 = self.cy
        x2 = min(self.cx + self.subimage_width, self.image_width)
        y2 = min(self.cy + self.subimage_height, self.image_height)
        
        # Ensure we have valid bounds
        if x2 <= x1 or y2 <= y1:
            # Return a blank image if invalid
            return Image.new('RGB', (self.subimage_width, self.subimage_height), color='black')
        
        # Crop the image
        box = (x1, y1, x2, y2)
        subimage = self.image.crop(box)
        
        # If the subimage is smaller than requested (near edges), pad it
        if subimage.size[0] < self.subimage_width or subimage.size[1] < self.subimage_height:
            padded = Image.new(self.image.mode, (self.subimage_width, self.subimage_height), color='black')
            padded.paste(subimage, (0, 0))
            return padded
        
        return subimage
    
    def get_image_size(self) -> Tuple[int, int]:
        """Get the size of the source image."""
        return (self.image_width, self.image_height)
    
    def __repr__(self) -> str:
        return f"CameraTester(cx={self.cx}, cy={self.cy}, image_size={self.image_width}x{self.image_height})"


class ActualCamera(Camera):
    """Actual camera implementation (placeholder)."""
    
    # Camera device index (only one camera used for entire harness)
    CAMERA_INDEX = 1
    
    def __init__(self):
        """
        Initialize actual camera with default dimensions.
        Only one camera is used for the entire harness.
        """
        self.width = self.DEFAULT_WIDTH
        self.height = self.DEFAULT_HEIGHT
        self.cap = None
        
        # Set subimage dimensions to 640x480
        self.subimage_width = 640
        self.subimage_height = 480
        
        # Track current position (for compatibility with CameraTester interface)
        self.cx = 0
        self.cy = 0
    
    def run_motor_horiz(self, direction: int, duration: float):
        """
        Run horizontal motor.
        
        Args:
            direction: 1 for right, -1 for left
            duration: Duration in seconds
        """
        # TODO: Implement actual motor control
        pass
    
    def run_motor_vert(self, direction: int, duration: float):
        """
        Run vertical motor.
        
        Args:
            direction: 1 for up, -1 for down
            duration: Duration in seconds
        """
        # TODO: Implement actual motor control
        pass
    
    def move_right(self, pixels: int = 1) -> Tuple[int, int]:
        """
        Move camera right (positive) or left (negative).
        
        Args:
            pixels: Number of pixels to move (positive = right, negative = left)
            
        Returns:
            New (x, y) position
        """
        # Convert pixels to inches using inch per pixel of width
        inches = abs(pixels) * self.INCH_PER_PIXEL_OF_WIDTH
        
        # Calculate duration based on velocity (distance / velocity = time)
        duration = inches / self.VELOCITY_INCHES_PER_SEC
        
        # Determine direction: 1 for right (positive pixels), -1 for left (negative pixels)
        direction = 1 if pixels >= 0 else -1
        
        # Run the horizontal motor
        self.run_motor_horiz(direction, duration)
        
        # Update position tracking
        self.cx += pixels
        return (self.cx, self.cy)
    
    def move_up(self, pixels: int = 1) -> Tuple[int, int]:
        """
        Move camera up (positive) or down (negative).
        
        Args:
            pixels: Number of pixels to move (positive = up, negative = down)
            
        Returns:
            New (x, y) position
        """
        # Convert pixels to inches using inch per pixel of width
        inches = abs(pixels) * self.INCH_PER_PIXEL_OF_WIDTH
        
        # Calculate duration based on velocity (distance / velocity = time)
        duration = inches / self.VELOCITY_INCHES_PER_SEC
        
        # Determine direction: 1 for up (positive pixels), -1 for down (negative pixels)
        direction = 1 if pixels >= 0 else -1
        
        # Run the vertical motor
        self.run_motor_vert(direction, duration)
        
        # Update position tracking (Y increases downward, so up is negative)
        self.cy -= pixels
        return (self.cx, self.cy)
    
    def move_left(self, pixels: int = 1) -> Tuple[int, int]:
        """Move camera left. Convenience method for move_right(-pixels)."""
        return self.move_right(-pixels)
    
    def move_down(self, pixels: int = 1) -> Tuple[int, int]:
        """Move camera down. Convenience method for move_up(-pixels)."""
        return self.move_up(-pixels)
    
    def move(self, dx: int = 0, dy: int = 0) -> Tuple[int, int]:
        """
        Move camera by delta x and delta y.
        
        Args:
            dx: Change in x (positive = right, negative = left)
            dy: Change in y (positive = down, negative = up)
                  Note: For ActualCamera, positive dy moves down (increases cy)
            
        Returns:
            New (cx, cy) position
        """
        if dx != 0:
            self.move_right(dx)
        if dy != 0:
            # move_up expects: positive = up (decreases cy), negative = down (increases cy)
            # dy: positive = down (should increase cy), negative = up (should decrease cy)
            # So we need to negate dy: move_up(-dy)
            self.move_up(-dy)
        return (self.cx, self.cy)
    
    def get_position(self) -> Tuple[int, int]:
        """Get current camera position."""
        return (self.cx, self.cy)
    
    def get_subimage(self) -> Image.Image:
        """
        Get a subimage from the captured image.
        For ActualCamera, this just returns the full captured image (640x480).
        
        Returns:
            PIL Image of the subimage
        """
        # For actual camera, we always return the full captured image
        return self.take_picture()
    
    def set_position(self, x: int, y: int) -> Tuple[int, int]:
        """
        Set camera position directly.
        
        Args:
            x: X coordinate
            y: Y coordinate
            
        Returns:
            New (cx, cy) position
        """
        self.cx = x
        self.cy = y
        return (self.cx, self.cy)
    
    def take_picture(self) -> Image.Image:
        """
        Take a picture with the camera and return as PIL Image.
        
        Returns:
            PIL Image of size 640x480
        """
        import cv2
        
        # Initialize camera if not already done
        if self.cap is None:
            self.cap = cv2.VideoCapture(self.CAMERA_INDEX)
            if not self.cap.isOpened():
                raise RuntimeError(f"Could not open camera at index {self.CAMERA_INDEX}")
            
            # Set resolution to 640x480
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.width)
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.height)
            
            # Set codec to MJPG for better framerate
            self.cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
        
        # Warm up the camera (skip first few frames for auto-focus/exposure adjustment)
        for _ in range(5):
            self.cap.read()
        
        # Capture a single frame
        ret, frame = self.cap.read()
        
        if not ret:
            raise RuntimeError("Failed to capture image from camera")
        
        # Resize to 640x480 if needed
        if frame.shape[1] != self.width or frame.shape[0] != self.height:
            frame = cv2.resize(frame, (self.width, self.height))
        
        # Convert BGR to RGB for PIL
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        return Image.fromarray(frame_rgb)
    
    def release(self):
        """Release the camera resource."""
        if self.cap is not None:
            self.cap.release()
            self.cap = None


if __name__ == "__main__":
    # Example usage
    print("Camera Tester Example")
    print("=" * 50)
    
    # Initialize camera at position (0, 0)
    camera = CameraTester("test_borders.png", initial_x=0, initial_y=0)
    print(f"Initialized: {camera}")
    print(f"Image size: {camera.get_image_size()}")
    print()
    
    # Get a subimage
    print("Getting subimage at (0, 0)...")
    subimage = camera.get_subimage()
    print(f"Subimage size: {subimage.size}")
    subimage.save("test_subimage_initial.png")
    print("Saved to test_subimage_initial.png")
    print()
    
    # Move right
    print("Moving right by 100 pixels...")
    camera.move_right(100)
    print(f"New position: {camera.get_position()}")
    subimage = camera.get_subimage()
    subimage.save("test_subimage_right.png")
    print("Saved to test_subimage_right.png")
    print()
    
    # Move up
    print("Moving up by 50 pixels...")
    camera.move_up(50)
    print(f"New position: {camera.get_position()}")
    subimage = camera.get_subimage()
    subimage.save("test_subimage_up.png")
    print("Saved to test_subimage_up.png")
    print()
    
    # Move left (negative)
    print("Moving left by 50 pixels...")
    camera.move_left(50)
    print(f"New position: {camera.get_position()}")
    print()
    
    # Move down (negative)
    print("Moving down by 25 pixels...")
    camera.move_down(25)
    print(f"New position: {camera.get_position()}")
    print()
    
    # Test near edge
    print("Moving to near bottom-right corner...")
    camera.set_position(1500, 800)
    print(f"Position: {camera.get_position()}")
    subimage = camera.get_subimage()
    print(f"Subimage size (may be smaller near edge): {subimage.size}")
    subimage.save("test_subimage_edge.png")
    print("Saved to test_subimage_edge.png")
    print()
    
    print("Done! Check the generated test images.")
