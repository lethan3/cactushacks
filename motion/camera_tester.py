#!/usr/bin/env python3
"""
Camera tester for simulating camera movement and capturing subimages.
Maintains current position (cx, cy) and allows movement in all directions.
"""
from PIL import Image
from pathlib import Path
from typing import Tuple, Optional


class CameraTester:
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
