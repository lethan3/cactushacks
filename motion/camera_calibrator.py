#!/usr/bin/env python3
"""
Camera calibrator that finds the green region boundary (Qx, Qy) using binary search.
The image is green for all x <= Qx and y <= Qy.
"""
import numpy as np
from PIL import Image, ImageDraw
from camera_tester import CameraTester
from pathlib import Path


class CameraCalibrator:
    """Calibrates camera position to find green region boundary."""
    
    def __init__(self, image_path: str = "test_borders.png", 
                 initial_x: int = 0, initial_y: int = 0,
                 kernel_size: int = 13, test_jump: int = 20):
        """
        Initialize the calibrator.
        
        Args:
            image_path: Path to test image
            initial_x: Initial x position
            initial_y: Initial y position
            kernel_size: Size of edge detection kernel (k x k) - should be 13 for the new kernel
            test_jump: Number of pixels to jump when searching for initial bounds
        """
        self.camera = CameraTester(image_path, initial_x, initial_y)
        self.test_jump = test_jump
        
        # Kernel pattern: scaled down by 4x (19x19 kernel, was 76x76)
        # Pattern: 7 values of -3, then -2, -1, 0, 1, 2, then 7 values of 3 (19 elements total)
        # So kernel_size is 19
        self.kernel_size = 19
        
        # Create vertical edge detection kernel (detects horizontal transitions/left-right edges)
        # Row vector: 7x[-3] + [-2, -1, 0, 1, 2] + 7x[3]
        row_vector = np.concatenate([
            np.full(7, -3),  # 7 values of -3
            np.array([-2, -1, 0, 1, 2]),  # Transition values
            np.full(7, 3)  # 7 values of 3
        ])
        # Repeat row vector for each row (19 x 19 kernel)
        self.vertical_kernel = np.tile(row_vector, (self.kernel_size, 1))
        
        # Create horizontal edge detection kernel (detects vertical transitions/up-down edges)
        # Column vector: same pattern
        column_vector = np.concatenate([
            np.full(7, -3),  # 7 values of -3
            np.array([-2, -1, 0, 1, 2]),  # Transition values
            np.full(7, 3)  # 7 values of 3
        ])
        # Repeat column vector for each column (19 x 19 kernel)
        self.horizontal_kernel = np.tile(column_vector.reshape(-1, 1), (1, self.kernel_size))
        
        self.debug_dir = Path("calibration_debug")
        self.debug_dir.mkdir(exist_ok=True)
        self.step_count = 0
        # Subimages scaled down by 4x (was 200x200, now 50x50)
        self.subimage_width = 50
        self.subimage_height = 50
    
    def check_green(self, subimage: Image.Image, binsearch_value: bool = False, return_activations: bool = False, check_direction: str = "vertical") -> int | dict | tuple:
        """
        Check green region status with detailed detection information.
        
        Args:
            subimage: PIL Image to check
            binsearch_value: If True, returns binary search value (0, 1, or 2).
                           If False, returns detailed dictionary.
            return_activations: If True, returns results dict, activations, and binsearch_result
            check_direction: "vertical" (default) for left-right edges, "horizontal" or "up_down" for up-down edges
        
        Returns:
            If binsearch_value=True:
                1 if edge detected (direction depends on check_direction)
                2 if too much green (>= 80% green, still in green region)
                0 if no green (not enough green)
            If binsearch_value=False:
                Dictionary with detection results: left_right_edge, up_down_edge, 
                green_threshold, lr_value, ud_value, green_ratio
        """
        # Convert to numpy array and extract channels
        img_array = np.array(subimage, dtype=np.float32)
        height, width = img_array.shape[:2]
        
        # Extract red and blue channels and compute |red| - |blue|
        if len(img_array.shape) == 3:
            red_channel = img_array[:, :, 0] / 255.0  # Normalize to 0-1
            blue_channel = img_array[:, :, 2] / 255.0  # Normalize to 0-1
            # Compute |red| - |blue| for kernel convolution
            kernel_channel = np.abs(red_channel) - np.abs(blue_channel)
        else:
            # Grayscale: use as-is
            kernel_channel = img_array / 255.0
        
        # Edge detection using convolution
        pad_size = self.kernel_size // 2
        padded = np.pad(kernel_channel, pad_size, mode='edge')
        
        # Convolve vertical kernel over |red| - |blue| channel (detects left-right edges/horizontal transitions)
        try:
            from scipy import signal
            vertical_result = signal.convolve2d(kernel_channel, self.vertical_kernel, mode='same')
        except ImportError:
            # Manual convolution
            vertical_result = np.zeros_like(kernel_channel)
            for i in range(height):
                for j in range(width):
                    region = padded[i:i+self.kernel_size, j:j+self.kernel_size]
                    vertical_result[i, j] = np.sum(region * self.vertical_kernel)
        
        # Convolve horizontal kernel over |red| - |blue| channel (detects up-down edges/vertical transitions)
        try:
            from scipy import signal
            horizontal_result = signal.convolve2d(kernel_channel, self.horizontal_kernel, mode='same')
        except ImportError:
            # Manual convolution
            pad_size = self.kernel_size // 2
            padded = np.pad(kernel_channel, pad_size, mode='edge')
            horizontal_result = np.zeros_like(kernel_channel)
            for i in range(height):
                for j in range(width):
                    region = padded[i:i+self.kernel_size, j:j+self.kernel_size]
                    horizontal_result[i, j] = np.sum(region * self.horizontal_kernel)
        
        # Calculate absolute intensities for edge detection
        vertical_intensities = np.abs(vertical_result)
        horizontal_intensities = np.abs(horizontal_result)
        
        # Edge detection: if more than 5% of pixels have intensity >= 100
        intensity_threshold = 100.0
        percentage_threshold = 0.05  # 5%
        total_pixels = height * width
        
        vertical_high_intensity_pixels = np.sum(vertical_intensities >= intensity_threshold)
        vertical_high_intensity_ratio = vertical_high_intensity_pixels / total_pixels
        vertical_edge_detected = vertical_high_intensity_ratio >= percentage_threshold
        
        horizontal_high_intensity_pixels = np.sum(horizontal_intensities >= intensity_threshold)
        horizontal_high_intensity_ratio = horizontal_high_intensity_pixels / total_pixels
        horizontal_edge_detected = horizontal_high_intensity_ratio >= percentage_threshold
        
        # For backward compatibility, calculate sums and thresholds
        vertical_edge_sum = np.sum(vertical_intensities)
        horizontal_edge_sum = np.sum(horizontal_intensities)
        # Use a threshold based on the image size for display purposes
        vertical_threshold = total_pixels * intensity_threshold * percentage_threshold
        horizontal_threshold = total_pixels * intensity_threshold * percentage_threshold
        
        # Green threshold check: ratio of green to (red + blue) >= 0.75
        # Count pixel as green if: green / (red + blue) >= 0.75
        green_threshold = 0.8  # 95% threshold for "too much green"
        if len(img_array.shape) == 3:
            red_channel = img_array[:, :, 0] / 255.0  # Normalize to 0-1
            green_channel = img_array[:, :, 1] / 255.0  # Normalize to 0-1
            blue_channel = img_array[:, :, 2] / 255.0  # Normalize to 0-1
            # Compute red + blue for each pixel
            red_blue_sum = red_channel + blue_channel
            # Avoid division by zero: if red+blue is 0, consider it green if green > 0
            # Otherwise, check if green / (red + blue) >= 0.75
            green_ratio_mask = np.where(red_blue_sum > 0,
                                       green_channel / red_blue_sum >= 0.75,
                                       green_channel > 0)
            green_pixels = np.sum(green_ratio_mask)
            green_pixel_threshold = 0.75
        else:
            # Grayscale: use original logic
            green_channel = img_array / 255.0
            green_pixels = np.sum(green_channel >= 0.2)
            green_pixel_threshold = 0.2
        green_ratio = green_pixels / (height * width)
        print(f'Green ratio: {green_ratio:.4f}, Green threshold: {green_threshold:.2f} (using green/(red+blue) >= {green_pixel_threshold})')
        
        # Build results dictionary
        results = {
            'left_right_edge': vertical_edge_detected,
            'up_down_edge': horizontal_edge_detected,
            'green_threshold': green_ratio >= green_threshold,
            'lr_value': float(vertical_high_intensity_ratio) / percentage_threshold,  # Normalized to threshold
            'ud_value': float(horizontal_high_intensity_ratio) / percentage_threshold,  # Normalized to threshold
            'green_ratio': float(green_ratio)
        }
        
        # Calculate binary search result if needed
        # Priority: edge detection first, then green content check only if no edge
        # Use the appropriate edge detection based on check_direction
        binsearch_result = None
        if binsearch_value:
            # Determine which edge to check based on direction
            if check_direction == "horizontal" or check_direction == "up_down":
                edge_detected = horizontal_edge_detected
            else:  # default to vertical/left_right
                edge_detected = vertical_edge_detected
            
            if edge_detected:
                # Edge detected - return 1 immediately
                binsearch_result = 1
            else:
                # No edge detected - check green content
                if green_ratio >= green_threshold:
                    binsearch_result = 2  # Too much green (still in green region)
                else:
                    binsearch_result = 0  # No green (not enough green)
        
        # Return results based on what was requested
        if return_activations:
            return results, vertical_result, horizontal_result, binsearch_result
        elif binsearch_value:
            return binsearch_result
        else:
            return results
    
    def save_debug_image(self, subimage: Image.Image, label: str):
        """
        Save full image with red box around subimage area and annotations inside.
        
        Args:
            subimage: The subimage that was captured
            label: Label for the debug image filename
        """
        filename = f"step_{self.step_count:03d}_{label}.png"
        filepath = self.debug_dir / filename
        
        # Load the full image
        full_image = Image.open(self.camera.image_path).copy()
        draw = ImageDraw.Draw(full_image)
        
        # Draw red rectangle around the subimage area
        x1 = self.camera.cx
        y1 = self.camera.cy
        x2 = self.camera.cx + self.subimage_width
        y2 = self.camera.cy + self.subimage_height
        draw.rectangle([x1, y1, x2, y2], outline="red", width=3)
        
        # Get detailed detection results and kernel activations
        detections, vertical_activations, horizontal_activations, check_result = self.check_green(
            subimage, binsearch_value=False, return_activations=True
        )
        
        # Save kernel activation visualizations
        self._save_activation_images(vertical_activations, horizontal_activations, label)
        
        # Annotate inside the box
        annotations = []
        
        # Add check_green result
        if check_result == 1:
            annotations.append("Edge detected (1)")
        elif check_result == 2:
            annotations.append("Too much green (2)")
        elif check_result == 0:
            annotations.append("No green (0)")
        
        # Add detailed detections with values (as percentages of threshold)
        lr_percent = detections['lr_value'] * 100
        ud_percent = detections['ud_value'] * 100
        
        if detections['left_right_edge']:
            annotations.append(f"L/R edge: {lr_percent:.1f}%")
        else:
            annotations.append(f"L/R: {lr_percent:.1f}%")
        
        if detections['up_down_edge']:
            annotations.append(f"U/D edge: {ud_percent:.1f}%")
        else:
            annotations.append(f"U/D: {ud_percent:.1f}%")
        
        annotations.append(f"Green: {detections['green_ratio']:.2%}")
        
        # Draw annotations inside the box (top-left corner)
        text_y = y1 + 5
        for annotation in annotations:
            bbox = draw.textbbox((x1 + 5, text_y), annotation)
            draw.rectangle([x1 + 3, text_y - 2, bbox[2] + 2, bbox[3] + 2], fill="black")
            draw.text((x1 + 5, text_y), annotation, fill="yellow")
            text_y += 15
        
        # Save the annotated full image
        full_image.save(filepath)
        print(f"  Saved debug image: {filepath} (red box at x={x1}, y={y1})")
        self.step_count += 1
    
    def _save_activation_images(self, vertical_activations: np.ndarray, 
                                horizontal_activations: np.ndarray, label: str):
        """
        Save visualization images of kernel activations.
        
        Args:
            vertical_activations: 2D array of vertical kernel convolution results
            horizontal_activations: 2D array of horizontal kernel convolution results
            label: Label for the debug image filename
        """
        # Normalize activations to 0-255 range for visualization
        def normalize_activations(activations):
            # Use absolute value for visualization
            abs_activations = np.abs(activations)
            if abs_activations.max() > 0:
                normalized = (abs_activations / abs_activations.max() * 255).astype(np.uint8)
            else:
                normalized = np.zeros_like(activations, dtype=np.uint8)
            return normalized
        
        # Create activation images
        vertical_img = Image.fromarray(normalize_activations(vertical_activations), mode='L')
        horizontal_img = Image.fromarray(normalize_activations(horizontal_activations), mode='L')
        
        # Convert to RGB for better visualization (apply colormap)
        vertical_rgb = vertical_img.convert('RGB')
        horizontal_rgb = horizontal_img.convert('RGB')
        
        # Apply a colormap (hot colormap: black -> red -> yellow -> white)
        vertical_array = np.array(vertical_rgb)
        horizontal_array = np.array(horizontal_rgb)
        
        normalized_v = normalize_activations(vertical_activations) / 255.0
        normalized_h = normalize_activations(horizontal_activations) / 255.0
        
        # Hot colormap: intensity -> RGB (with values capped at 255)
        for i in range(vertical_array.shape[0]):
            for j in range(vertical_array.shape[1]):
                intensity = normalized_v[i, j]
                if intensity < 0.33:
                    # Black to red
                    r = min(255, int(intensity * 3 * 255))
                    vertical_array[i, j] = [r, 0, 0]
                elif intensity < 0.66:
                    # Red to yellow
                    g = min(255, int((intensity - 0.33) * 3 * 255))
                    vertical_array[i, j] = [255, g, 0]
                else:
                    # Yellow to white
                    t = (intensity - 0.66) * 3
                    b = min(255, int(t * 255))
                    vertical_array[i, j] = [255, 255, b]
        
        for i in range(horizontal_array.shape[0]):
            for j in range(horizontal_array.shape[1]):
                intensity = normalized_h[i, j]
                if intensity < 0.33:
                    # Black to red
                    r = min(255, int(intensity * 3 * 255))
                    horizontal_array[i, j] = [r, 0, 0]
                elif intensity < 0.66:
                    # Red to yellow
                    g = min(255, int((intensity - 0.33) * 3 * 255))
                    horizontal_array[i, j] = [255, g, 0]
                else:
                    # Yellow to white
                    t = (intensity - 0.66) * 3
                    b = min(255, int(t * 255))
                    horizontal_array[i, j] = [255, 255, b]
        
        # Ensure arrays are uint8 and values are capped at 255
        vertical_array = np.clip(vertical_array, 0, 255).astype(np.uint8)
        horizontal_array = np.clip(horizontal_array, 0, 255).astype(np.uint8)
        
        # Save activation images
        vertical_filename = f"step_{self.step_count:03d}_{label}_vertical_activations.png"
        horizontal_filename = f"step_{self.step_count:03d}_{label}_horizontal_activations.png"
        
        Image.fromarray(vertical_array).save(self.debug_dir / vertical_filename)
        Image.fromarray(horizontal_array).save(self.debug_dir / horizontal_filename)
        
        print(f"  Saved activation images: {vertical_filename}, {horizontal_filename}")
    
    def find_qx(self) -> int:
        """
        Find Qx using binary search.
        
        Returns:
            Qx coordinate
        """
        print("\n" + "="*60)
        print("Finding Qx (x-coordinate boundary)")
        print("="*60)
        
        # Find initial bounds for x
        print("\nStep 1: Finding initial bounds for x...")
        image_width, _ = self.camera.get_image_size()
        lo = float('-inf')
        hi = image_width
        
        # Check current position first
        subimage = self.camera.get_subimage(self.subimage_width, self.subimage_height)
        result = self.check_green(subimage, binsearch_value=True)
        self.save_debug_image(subimage, f"find_qx_init_x{self.camera.cx}_y{self.camera.cy}")
        
        # If we're already in green region, use current position as lo
        if result == 1 or result == 2:
            lo = self.camera.cx
            print(f"  Already in green region at x={self.camera.cx}, using as left bound")
        else:
            # Move left until we find the left bound (or hit boundary)
            while lo == float('-inf'):
                subimage = self.camera.get_subimage(self.subimage_width, self.subimage_height)
                result = self.check_green(subimage, binsearch_value=True)
                self.save_debug_image(subimage, f"find_qx_init_x{self.camera.cx}_y{self.camera.cy}")
                
                # result == 1 (edge) or result == 2 (too much green) means we're in green region
                if result == 1 or result == 2:
                    lo = self.camera.cx
                    print(f"  Found left bound at x={lo} (result={result})")
                else:
                    old_x = self.camera.cx
                    if old_x == 0:
                        # Already at left boundary and still no green - this shouldn't happen
                        # if we properly exited green region, but handle it anyway
                        print(f"  Warning: At x=0 with no green detected. This suggests the entire image has no green at this y position.")
                        print(f"  Setting lo=0 as fallback, but binary search may not work correctly.")
                        lo = 0
                        break
                    self.camera.move_left(self.test_jump)
                    # Check if we actually moved (might be clamped at boundary)
                    if self.camera.cx == old_x:
                        # We're stuck at the left boundary, start from 0
                        lo = 0
                        print(f"  Hit left boundary at x=0, starting search from x=0")
                    else:
                        print(f"  Moving left by {self.test_jump} pixels, current x={self.camera.cx} (result={result})")
        
        # Binary search for Qx
        print("\nStep 2: Binary search for Qx...")
        for iteration in range(10):
            # Check if we've converged
            if hi - lo <= 1:
                print(f"  Converged at iteration {iteration + 1}: lo={lo}, hi={hi}")
                break
            
            mid = int((lo + hi) / 2)
            self.camera.set_position(mid, self.camera.cy)
            
            subimage = self.camera.get_subimage(self.subimage_width, self.subimage_height)
            result = self.check_green(subimage, binsearch_value=True)
            self.save_debug_image(subimage, f"find_qx_iter{iteration}_x{mid}_y{self.camera.cy}")
            
            print(f"  Iteration {iteration + 1}/10: x={mid}, result={result}, lo={lo}, hi={hi}")
            
            # result == 1 (edge) or result == 2 (too much green) means too far left
            if result == 1 or result == 2:
                lo = mid + 1  # Too far left, move right (ensure progress)
            else:
                hi = mid - 1  # Too far right (no green), move left (ensure progress)
        
        qx = int((lo + hi) / 2)
        print(f"\nQx found: {qx}")
        return qx
    
    def find_qy(self) -> int:
        """
        Find Qy using binary search.
        
        Returns:
            Qy coordinate
        """
        print("\n" + "="*60)
        print("Finding Qy (y-coordinate boundary)")
        print("="*60)
        
        # Find initial bounds for y
        print("\nStep 1: Finding initial bounds for y...")
        _, image_height = self.camera.get_image_size()
        lo = float('-inf')
        hi = image_height
        
        # Check current position first (use horizontal/up_down edge detection)
        subimage = self.camera.get_subimage(self.subimage_width, self.subimage_height)
        result = self.check_green(subimage, binsearch_value=True, check_direction="horizontal")
        self.save_debug_image(subimage, f"find_qy_init_x{self.camera.cx}_y{self.camera.cy}")
        
        # If we're already in green region, use current position as lo
        if result == 1 or result == 2:
            lo = self.camera.cy
            print(f"  Already in green region at y={self.camera.cy}, using as upper bound")
        else:
            # Move up until we find the upper bound (or hit boundary)
            while lo == float('-inf'):
                subimage = self.camera.get_subimage(self.subimage_width, self.subimage_height)
                result = self.check_green(subimage, binsearch_value=True, check_direction="horizontal")
                self.save_debug_image(subimage, f"find_qy_init_x{self.camera.cx}_y{self.camera.cy}")
                
                # result == 1 (edge) or result == 2 (too much green) means we're in green region
                if result == 1 or result == 2:
                    lo = self.camera.cy
                    print(f"  Found upper bound at y={lo} (result={result})")
                else:
                    old_y = self.camera.cy
                    if old_y == 0:
                        # Already at upper boundary and still no green - this shouldn't happen
                        # if we properly exited green region, but handle it anyway
                        print(f"  Warning: At y=0 with no green detected. This suggests the entire image has no green at this x position.")
                        print(f"  Setting lo=0 as fallback, but binary search may not work correctly.")
                        lo = 0
                        break
                    self.camera.move_up(self.test_jump)
                    # Check if we actually moved (might be clamped at boundary)
                    if self.camera.cy == old_y:
                        # We're stuck at the upper boundary, start from 0
                        lo = 0
                        print(f"  Hit upper boundary at y=0, starting search from y=0")
                    else:
                        print(f"  Moving up by {self.test_jump} pixels, current y={self.camera.cy} (result={result})")
        
        # Binary search for Qy
        print("\nStep 2: Binary search for Qy...")
        for iteration in range(10):
            # Check if we've converged
            if hi - lo <= 1:
                print(f"  Converged at iteration {iteration + 1}: lo={lo}, hi={hi}")
                break
            
            mid = int((lo + hi) / 2)
            self.camera.set_position(self.camera.cx, mid)
            
            subimage = self.camera.get_subimage(self.subimage_width, self.subimage_height)
            result = self.check_green(subimage, binsearch_value=True, check_direction="horizontal")
            self.save_debug_image(subimage, f"find_qy_iter{iteration}_x{self.camera.cx}_y{mid}")
            
            print(f"  Iteration {iteration + 1}/10: y={mid}, result={result}, lo={lo}, hi={hi}")
            
            # result == 1 (edge) or result == 2 (too much green) means too far up
            if result == 1 or result == 2:
                lo = mid + 1  # Too far up, move down (ensure progress)
            else:
                hi = mid - 1  # Too far down (no green), move up (ensure progress)
        
        qy = int((lo + hi) / 2)
        print(f"\nQy found: {qy}")
        return qy
    
    def calibrate(self) -> tuple[int, int]:
        """
        Perform full calibration to find Qx and Qy.
        
        Returns:
            (Qx, Qy) tuple
        """
        print("Starting camera calibration...")
        print(f"Image size: {self.camera.get_image_size()}")
        print(f"Initial position: {self.camera.get_position()}")
        
        # Reset to starting position
        self.camera.set_position(0, 0)
        
        # First, check if we're already outside the green region
        subimage = self.camera.get_subimage(self.subimage_width, self.subimage_height)
        initial_result = self.check_green(subimage, binsearch_value=True)
        print(f"Initial position (0,0) check result: {initial_result}")
        
        # Move down and right until we get 0 from check_green (no green)
        print("\n" + "="*60)
        print("Step 0: Moving down and right until we exit green region")
        print("="*60)
        
        # If we're already outside green region, we need to find where green starts
        if initial_result == 0:
            print("  Starting outside green region, moving right and down to find green...")
            # Move right and down until we enter green region, then continue until we exit
            max_iterations = 1000
            iteration = 0
            found_green = False
            while iteration < max_iterations:
                subimage = self.camera.get_subimage(self.subimage_width, self.subimage_height)
                result = self.check_green(subimage, binsearch_value=True)
                self.save_debug_image(subimage, f"initial_move_find_green_x{self.camera.cx}_y{self.camera.cy}")
                
                if result != 0 and not found_green:
                    found_green = True
                    print(f"  Entered green region at x={self.camera.cx}, y={self.camera.cy}")
                
                if found_green and result == 0:
                    print(f"  Exited green region at x={self.camera.cx}, y={self.camera.cy}")
                    break
                
                old_pos = (self.camera.cx, self.camera.cy)
                self.camera.move_right(self.test_jump)
                self.camera.move_down(self.test_jump)
                new_pos = (self.camera.cx, self.camera.cy)
                
                if new_pos == old_pos:
                    print(f"  Cannot move further (at boundary), current position: x={self.camera.cx}, y={self.camera.cy}")
                    break
                
                iteration += 1
        else:
            # We're in green region, move down and right until we exit
            max_iterations = 1000
            iteration = 0
            while iteration < max_iterations:
                subimage = self.camera.get_subimage(self.subimage_width, self.subimage_height)
                result = self.check_green(subimage, binsearch_value=True)
                self.save_debug_image(subimage, f"initial_move_x{self.camera.cx}_y{self.camera.cy}")
                
                if result == 0:
                    print(f"  Exited green region at x={self.camera.cx}, y={self.camera.cy}")
                    break
                
                old_pos = (self.camera.cx, self.camera.cy)
                self.camera.move_down(self.test_jump)
                self.camera.move_right(self.test_jump)
                new_pos = (self.camera.cx, self.camera.cy)
                
                if new_pos == old_pos:
                    print(f"  Cannot move further (at boundary), current position: x={self.camera.cx}, y={self.camera.cy}")
                    break
                
                print(f"  Still in green region at x={self.camera.cx}, y={self.camera.cy}, moving down and right... (result={result})")
                iteration += 1
        
        # Find Qx (keeping current y position)
        qx = self.find_qx()
        
        # Find Qy (keeping current x position from Qx search)
        qy = self.find_qy()
        
        print("\n" + "="*60)
        print("CALIBRATION COMPLETE")
        print("="*60)
        print(f"Qx = {qx}")
        print(f"Qy = {qy}")
        print(f"Green region: x <= {qx} AND y <= {qy}")
        print(f"\nDebug images saved in: {self.debug_dir}")
        
        return (qx, qy)
    
    def sweep_image(self, step_size: int = 50, save_individual: bool = False) -> None:
        """
        Sweep across the entire image and debug check_green results at each position.
        
        Args:
            step_size: Number of pixels to step between positions
            save_individual: If True, save individual debug images for each position
        """
        print("\n" + "="*60)
        print("SWEEPING ENTIRE IMAGE")
        print("="*60)
        
        image_width, image_height = self.camera.get_image_size()
        print(f"Image size: {image_width} x {image_height}")
        print(f"Step size: {step_size}")
        print(f"Will check approximately {(image_width // step_size + 1) * (image_height // step_size + 1)} positions")
        
        # Create results grid
        results_grid = []
        positions = []
        
        # Sweep across image
        y_positions = list(range(0, image_height, step_size))
        if y_positions[-1] != image_height - 1:
            y_positions.append(image_height - 1)
        
        x_positions = list(range(0, image_width, step_size))
        if x_positions[-1] != image_width - 1:
            x_positions.append(image_width - 1)
        
        total_positions = len(x_positions) * len(y_positions)
        current_position = 0
        
        for y in y_positions:
            row_results = []
            row_positions = []
            for x in x_positions:
                self.camera.set_position(x, y)
                subimage = self.camera.get_subimage(self.subimage_width, self.subimage_height)
                
                # Get both binary and detailed results
                result_binary = self.check_green(subimage, binsearch_value=True)
                result_detailed = self.check_green(subimage, binsearch_value=False)
                
                row_results.append({
                    'binary': result_binary,
                    'detailed': result_detailed,
                    'x': x,
                    'y': y
                })
                row_positions.append((x, y))
                
                current_position += 1
                if current_position % 10 == 0:
                    print(f"  Progress: {current_position}/{total_positions} positions checked...")
                
                if save_individual:
                    self.save_debug_image(subimage, f"sweep_x{x}_y{y}")
            
            results_grid.append(row_results)
            positions.append(row_positions)
        
        print(f"\nCompleted sweep: {total_positions} positions checked")
        
        # Create summary visualization
        print("\nCreating summary visualization...")
        self._create_sweep_summary(results_grid, positions, image_width, image_height)
        
        # Print statistics
        self._print_sweep_stats(results_grid)
    
    def _create_sweep_summary(self, results_grid: list, positions: list, 
                              image_width: int, image_height: int) -> None:
        """Create a summary visualization of the sweep results."""
        from PIL import Image, ImageDraw, ImageFont
        
        # Create a colored map: 0=blue (no green), 1=red (edge), 2=yellow (too much green)
        summary_image = Image.new('RGB', (image_width, image_height), color='white')
        draw = ImageDraw.Draw(summary_image)
        
        # Draw grid cells
        for row_idx, row_results in enumerate(results_grid):
            for col_idx, result_data in enumerate(row_results):
                x = result_data['x']
                y = result_data['y']
                binary_result = result_data['binary']
                
                # Determine color based on binary result
                if binary_result == 0:
                    color = (0, 0, 255)  # Blue - no green
                elif binary_result == 1:
                    color = (255, 0, 0)  # Red - edge detected
                else:  # binary_result == 2
                    color = (255, 255, 0)  # Yellow - too much green
                
                # Draw a rectangle for this position (using step size to determine size)
                # Find next position to determine rectangle size
                if col_idx < len(row_results) - 1:
                    next_x = row_results[col_idx + 1]['x']
                    width = next_x - x
                else:
                    width = image_width - x
                
                if row_idx < len(results_grid) - 1:
                    next_y = results_grid[row_idx + 1][0]['y']
                    height = next_y - y
                else:
                    height = image_height - y
                
                draw.rectangle([x, y, x + width, y + height], fill=color, outline='black', width=1)
        
        # Save summary image
        summary_path = self.debug_dir / "sweep_summary.png"
        summary_image.save(summary_path)
        print(f"  Summary visualization saved to: {summary_path}")
        
        # Create detailed heatmap with L/R, U/D, and green ratio values
        self._create_detailed_heatmap(results_grid, positions, image_width, image_height)
    
    def _create_detailed_heatmap(self, results_grid: list, positions: list,
                                 image_width: int, image_height: int) -> None:
        """Create detailed heatmaps for L/R, U/D, and green ratio."""
        import numpy as np
        from PIL import Image
        
        # Create arrays for each metric
        lr_array = np.zeros((len(results_grid), len(results_grid[0])), dtype=np.float32)
        ud_array = np.zeros((len(results_grid), len(results_grid[0])), dtype=np.float32)
        green_array = np.zeros((len(results_grid), len(results_grid[0])), dtype=np.float32)
        
        for row_idx, row_results in enumerate(results_grid):
            for col_idx, result_data in enumerate(row_results):
                detailed = result_data['detailed']
                lr_array[row_idx, col_idx] = detailed['lr_value']
                ud_array[row_idx, col_idx] = detailed['ud_value']
                green_array[row_idx, col_idx] = detailed['green_ratio']
        
        # Normalize and create heatmaps
        def create_heatmap(data, title, colormap_name='viridis'):
            # Normalize to 0-255
            data_min, data_max = data.min(), data.max()
            if data_max > data_min:
                normalized = ((data - data_min) / (data_max - data_min) * 255).astype(np.uint8)
            else:
                normalized = np.zeros_like(data, dtype=np.uint8)
            
            # Resize to image dimensions
            from PIL import Image
            heatmap_img = Image.fromarray(normalized, mode='L')
            heatmap_img = heatmap_img.resize((image_width, image_height), Image.NEAREST)
            
            # Convert to RGB with colormap
            heatmap_rgb = heatmap_img.convert('RGB')
            
            # Apply colormap (simple implementation)
            if colormap_name == 'hot':
                # Red to yellow colormap
                heatmap_array = np.array(heatmap_rgb)
                normalized_array = normalized.repeat(len(results_grid[0])).reshape(len(results_grid), -1)
                normalized_array = np.repeat(normalized_array, image_height // len(results_grid) + 1, axis=0)[:image_height]
                normalized_array = np.repeat(normalized_array, image_width // len(results_grid[0]) + 1, axis=1)[:, :image_width]
                
                # Simple hot colormap: black -> red -> yellow -> white
                heatmap_array[:, :, 0] = np.clip(normalized_array * 2, 0, 255)  # Red channel
                heatmap_array[:, :, 1] = np.clip((normalized_array - 128) * 2, 0, 255)  # Green channel
                heatmap_array[:, :, 2] = np.clip((normalized_array - 192) * 4, 0, 255)  # Blue channel
                heatmap_rgb = Image.fromarray(heatmap_array)
            
            return heatmap_rgb
        
        # Create and save heatmaps
        lr_heatmap = create_heatmap(lr_array, "L/R Edge Value", 'hot')
        lr_path = self.debug_dir / "sweep_heatmap_lr.png"
        lr_heatmap.save(lr_path)
        print(f"  L/R edge heatmap saved to: {lr_path}")
        
        ud_heatmap = create_heatmap(ud_array, "U/D Edge Value", 'hot')
        ud_path = self.debug_dir / "sweep_heatmap_ud.png"
        ud_heatmap.save(ud_path)
        print(f"  U/D edge heatmap saved to: {ud_path}")
        
        green_heatmap = create_heatmap(green_array, "Green Ratio", 'hot')
        green_path = self.debug_dir / "sweep_heatmap_green.png"
        green_heatmap.save(green_path)
        print(f"  Green ratio heatmap saved to: {green_path}")
    
    def _print_sweep_stats(self, results_grid: list) -> None:
        """Print statistics about the sweep results."""
        print("\n" + "="*60)
        print("SWEEP STATISTICS")
        print("="*60)
        
        # Count results
        count_0 = 0  # No green
        count_1 = 0  # Edge detected
        count_2 = 0  # Too much green
        
        all_lr_values = []
        all_ud_values = []
        all_green_ratios = []
        
        for row_results in results_grid:
            for result_data in row_results:
                binary = result_data['binary']
                detailed = result_data['detailed']
                
                if binary == 0:
                    count_0 += 1
                elif binary == 1:
                    count_1 += 1
                else:
                    count_2 += 1
                
                all_lr_values.append(detailed['lr_value'])
                all_ud_values.append(detailed['ud_value'])
                all_green_ratios.append(detailed['green_ratio'])
        
        total = count_0 + count_1 + count_2
        
        print(f"\nBinary Results:")
        print(f"  0 (No green): {count_0} ({count_0/total*100:.1f}%)")
        print(f"  1 (Edge detected): {count_1} ({count_1/total*100:.1f}%)")
        print(f"  2 (Too much green): {count_2} ({count_2/total*100:.1f}%)")
        
        print(f"\nL/R Edge Values (as % of threshold):")
        print(f"  Min: {min(all_lr_values)*100:.1f}%")
        print(f"  Max: {max(all_lr_values)*100:.1f}%")
        print(f"  Mean: {sum(all_lr_values)/len(all_lr_values)*100:.1f}%")
        
        print(f"\nU/D Edge Values (as % of threshold):")
        print(f"  Min: {min(all_ud_values)*100:.1f}%")
        print(f"  Max: {max(all_ud_values)*100:.1f}%")
        print(f"  Mean: {sum(all_ud_values)/len(all_ud_values)*100:.1f}%")
        
        print(f"\nGreen Ratios:")
        print(f"  Min: {min(all_green_ratios):.3f}")
        print(f"  Max: {max(all_green_ratios):.3f}")
        print(f"  Mean: {sum(all_green_ratios)/len(all_green_ratios):.3f}")
        
        # Find positions with interesting results
        print(f"\nInteresting Positions:")
        print(f"  Positions with edge detected (result=1):")
        edge_positions = []
        for row_results in results_grid:
            for result_data in row_results:
                if result_data['binary'] == 1:
                    edge_positions.append((result_data['x'], result_data['y']))
        if edge_positions:
            for x, y in edge_positions[:10]:  # Show first 10
                print(f"    ({x}, {y})")
            if len(edge_positions) > 10:
                print(f"    ... and {len(edge_positions) - 10} more")
        else:
            print(f"    None found")
        
        print(f"\n  Positions with too much green (result=2):")
        green_positions = []
        for row_results in results_grid:
            for result_data in row_results:
                if result_data['binary'] == 2:
                    green_positions.append((result_data['x'], result_data['y']))
        if green_positions:
            for x, y in green_positions[:10]:  # Show first 10
                print(f"    ({x}, {y})")
            if len(green_positions) > 10:
                print(f"    ... and {len(green_positions) - 10} more")
        else:
            print(f"    None found")
    
    def run_kernel_on_full_image(self, output_prefix: str = "full_image"):
        """
        Run kernel convolution on the entire image and save the results.
        
        Args:
            output_prefix: Prefix for output filenames
        """
        print("Running kernel convolution on full image...")
        
        # Load the full image
        full_image = Image.open(self.camera.image_path)
        img_array = np.array(full_image, dtype=np.float32)
        height, width = img_array.shape[:2]
        
        print(f"Image size: {width} x {height}")
        
        # Extract red and blue channels and compute |red| - |blue|
        if len(img_array.shape) == 3:
            red_channel = img_array[:, :, 0] / 255.0
            blue_channel = img_array[:, :, 2] / 255.0
            kernel_channel = np.abs(red_channel) - np.abs(blue_channel)
        else:
            kernel_channel = img_array / 255.0
        
        # Run vertical kernel convolution
        print("Running vertical kernel convolution...")
        try:
            from scipy import signal
            vertical_result = signal.convolve2d(kernel_channel, self.vertical_kernel, mode='same')
        except ImportError:
            # Manual convolution
            pad_size = self.kernel_size // 2
            padded = np.pad(kernel_channel, pad_size, mode='edge')
            vertical_result = np.zeros_like(kernel_channel)
            for i in range(height):
                for j in range(width):
                    region = padded[i:i+self.kernel_size, j:j+self.kernel_size]
                    vertical_result[i, j] = np.sum(region * self.vertical_kernel)
        
        # Run horizontal kernel convolution
        print("Running horizontal kernel convolution...")
        try:
            from scipy import signal
            horizontal_result = signal.convolve2d(kernel_channel, self.horizontal_kernel, mode='same')
        except ImportError:
            # Manual convolution
            pad_size = self.kernel_size // 2
            padded = np.pad(kernel_channel, pad_size, mode='edge')
            horizontal_result = np.zeros_like(kernel_channel)
            for i in range(height):
                for j in range(width):
                    region = padded[i:i+self.kernel_size, j:j+self.kernel_size]
                    horizontal_result[i, j] = np.sum(region * self.horizontal_kernel)
        
        # Save activation images
        # Binary visualization: white if intensity >= 100, black otherwise
        intensity_threshold = 100.0
        
        # Calculate absolute intensities
        vertical_intensities = np.abs(vertical_result)
        horizontal_intensities = np.abs(horizontal_result)
        
        # Create binary images: white (255) if >= threshold, black (0) otherwise
        vertical_binary = (vertical_intensities >= intensity_threshold).astype(np.uint8) * 255
        horizontal_binary = (horizontal_intensities >= intensity_threshold).astype(np.uint8) * 255
        
        # Convert to RGB (all channels same for grayscale)
        vertical_array = np.stack([vertical_binary, vertical_binary, vertical_binary], axis=-1)
        horizontal_array = np.stack([horizontal_binary, horizontal_binary, horizontal_binary], axis=-1)
        
        # Ensure arrays are uint8
        vertical_array = vertical_array.astype(np.uint8)
        horizontal_array = horizontal_array.astype(np.uint8)
        
        # Save activation images
        vertical_filename = f"{output_prefix}_vertical_activations.png"
        horizontal_filename = f"{output_prefix}_horizontal_activations.png"
        
        Image.fromarray(vertical_array).save(self.debug_dir / vertical_filename)
        Image.fromarray(horizontal_array).save(self.debug_dir / horizontal_filename)
        
        print(f"Saved activation images:")
        print(f"  Vertical: {self.debug_dir}/{vertical_filename}")
        print(f"  Horizontal: {self.debug_dir}/{horizontal_filename}")
        
        # Create intensity distribution histograms
        self._create_intensity_histograms(vertical_result, horizontal_result, output_prefix)
        
        # Create green pixel detection visualization
        self._create_green_pixel_visualization(img_array, output_prefix)
    
    def _create_intensity_histograms(self, vertical_result: np.ndarray, 
                                     horizontal_result: np.ndarray, 
                                     output_prefix: str):
        """
        Create histogram visualizations showing intensity distribution.
        
        Args:
            vertical_result: Vertical kernel convolution results
            horizontal_result: Horizontal kernel convolution results
            output_prefix: Prefix for output filenames
        """
        try:
            import matplotlib
            matplotlib.use('Agg')  # Use non-interactive backend
            import matplotlib.pyplot as plt
        except ImportError:
            print("Warning: matplotlib not available, skipping histogram generation")
            return
        
        # Compute absolute values for intensity
        vertical_intensities = np.abs(vertical_result).flatten()
        horizontal_intensities = np.abs(horizontal_result).flatten()
        
        # Create figure with subplots
        fig, axes = plt.subplots(2, 2, figsize=(12, 10))
        
        # Vertical kernel histogram
        ax1 = axes[0, 0]
        ax1.hist(vertical_intensities, bins=100, color='blue', alpha=0.7, edgecolor='black')
        ax1.set_xlabel('Intensity (absolute value)')
        ax1.set_ylabel('Frequency (pixel count)')
        ax1.set_title('Vertical Kernel: Intensity Distribution')
        ax1.grid(True, alpha=0.3)
        ax1.set_yscale('log')  # Log scale for better visualization
        
        # Horizontal kernel histogram
        ax2 = axes[0, 1]
        ax2.hist(horizontal_intensities, bins=100, color='red', alpha=0.7, edgecolor='black')
        ax2.set_xlabel('Intensity (absolute value)')
        ax2.set_ylabel('Frequency (pixel count)')
        ax2.set_title('Horizontal Kernel: Intensity Distribution')
        ax2.grid(True, alpha=0.3)
        ax2.set_yscale('log')
        
        # Cumulative distribution functions
        ax3 = axes[1, 0]
        sorted_v = np.sort(vertical_intensities)
        cumulative_v = np.arange(1, len(sorted_v) + 1) / len(sorted_v)
        ax3.plot(sorted_v, cumulative_v, color='blue', linewidth=2)
        ax3.set_xlabel('Intensity (absolute value)')
        ax3.set_ylabel('Cumulative Probability')
        ax3.set_title('Vertical Kernel: Cumulative Distribution')
        ax3.grid(True, alpha=0.3)
        
        ax4 = axes[1, 1]
        sorted_h = np.sort(horizontal_intensities)
        cumulative_h = np.arange(1, len(sorted_h) + 1) / len(sorted_h)
        ax4.plot(sorted_h, cumulative_h, color='red', linewidth=2)
        ax4.set_xlabel('Intensity (absolute value)')
        ax4.set_ylabel('Cumulative Probability')
        ax4.set_title('Horizontal Kernel: Cumulative Distribution')
        ax4.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        # Save histogram
        histogram_filename = f"{output_prefix}_intensity_histograms.png"
        histogram_path = self.debug_dir / histogram_filename
        plt.savefig(histogram_path, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  Intensity histograms: {self.debug_dir}/{histogram_filename}")
        
        # Print statistics
        print(f"\nIntensity Statistics:")
        print(f"  Vertical kernel:")
        print(f"    Min: {vertical_intensities.min():.6f}")
        print(f"    Max: {vertical_intensities.max():.6f}")
        print(f"    Mean: {vertical_intensities.mean():.6f}")
        print(f"    Median: {np.median(vertical_intensities):.6f}")
        print(f"    Std: {vertical_intensities.std():.6f}")
        print(f"  Horizontal kernel:")
        print(f"    Min: {horizontal_intensities.min():.6f}")
        print(f"    Max: {horizontal_intensities.max():.6f}")
        print(f"    Mean: {horizontal_intensities.mean():.6f}")
        print(f"    Median: {np.median(horizontal_intensities):.6f}")
        print(f"    Std: {horizontal_intensities.std():.6f}")
    
    def _create_green_pixel_visualization(self, img_array: np.ndarray, output_prefix: str):
        """
        Create visualization showing which pixels are detected as green.
        
        Args:
            img_array: Original image array
            output_prefix: Prefix for output filenames
        """
        height, width = img_array.shape[:2]
        
        if len(img_array.shape) != 3:
            print("Warning: Cannot create green pixel visualization for grayscale image")
            return
        
        # Extract channels
        red_channel = img_array[:, :, 0] / 255.0  # Normalize to 0-1
        green_channel = img_array[:, :, 1] / 255.0  # Normalize to 0-1
        blue_channel = img_array[:, :, 2] / 255.0  # Normalize to 0-1
        
        # Compute red + blue for each pixel
        red_blue_sum = red_channel + blue_channel
        
        # Check if green / (red + blue) >= 0.75
        # Avoid division by zero: if red+blue is 0, consider it green if green > 0
        green_pixel_threshold = 0.75
        green_mask = np.where(red_blue_sum > 0,
                             green_channel / red_blue_sum >= green_pixel_threshold,
                             green_channel > 0)
        
        # Create binary image: white (255) for green pixels, black (0) for non-green
        green_binary = green_mask.astype(np.uint8) * 255
        
        # Convert to RGB (all channels same for grayscale)
        green_array = np.stack([green_binary, green_binary, green_binary], axis=-1)
        green_array = green_array.astype(np.uint8)
        
        # Save visualization
        green_filename = f"{output_prefix}_green_pixels.png"
        green_path = self.debug_dir / green_filename
        Image.fromarray(green_array).save(green_path)
        
        # Calculate statistics
        total_pixels = height * width
        green_pixels = np.sum(green_mask)
        green_ratio = green_pixels / total_pixels
        
        print(f"Saved green pixel detection visualization:")
        print(f"  {self.debug_dir}/{green_filename}")
        print(f"  Green pixels: {green_pixels}/{total_pixels} ({green_ratio*100:.2f}%)")
        print(f"  Threshold: green / (red + blue) >= {green_pixel_threshold}")


if __name__ == "__main__":
    import sys
    
    calibrator = CameraCalibrator(
        image_path="test_borders.png",
        initial_x=0,
        initial_y=0,
        kernel_size=13,  # Must be 13 for the new kernel pattern
        test_jump=50
    )
    
    # Check if user wants to run kernel on full image
    if len(sys.argv) > 1 and sys.argv[1] == "full":
        calibrator.run_kernel_on_full_image()
    # Check if user wants to sweep instead of calibrate
    elif len(sys.argv) > 1 and sys.argv[1] == "sweep":
        step_size = int(sys.argv[2]) if len(sys.argv) > 2 else 50
        save_individual = "--save-individual" in sys.argv
        calibrator.sweep_image(step_size=step_size, save_individual=save_individual)
    else:
        qx, qy = calibrator.calibrate()
        print(f"\nFinal result: Qx={qx}, Qy={qy}")
