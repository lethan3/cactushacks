#!/usr/bin/env python3
"""
Camera calibrator that finds the green region boundary (Qx, Qy) using binary search.
The image is green for all x <= Qx and y <= Qy.
"""
import numpy as np
from PIL import Image, ImageDraw
from camera import CameraTester
from pathlib import Path
from camera_calibrator_debug import CameraCalibratorDebug


class CameraCalibrator:
    """Calibrates camera position to find green region boundary."""
    
    def __init__(self, camera: CameraTester,
                 kernel_size: int = 13, test_jump: int = 20,
                 debug: bool = False):
        """
        Initialize the calibrator.
        
        Args:
            camera: CameraTester instance to use for camera operations
            kernel_size: Size of edge detection kernel (k x k) - should be 13 for the new kernel
            test_jump: Number of pixels to jump when searching for initial bounds
            debug: If True, save debug images and print detailed information
        """
        self.camera = camera
        self.test_jump = test_jump
        self.debug = debug
        
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
        self.step_count = 0
        
        # Create debug helper only if debug is enabled
        if self.debug:
            self.debug_dir.mkdir(exist_ok=True)
            self.debug_helper = CameraCalibratorDebug(self)
        else:
            self.debug_helper = None
    
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
        if self.debug:
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
    
    def _debug_print(self, *args, **kwargs):
        """Print only if debug mode is enabled."""
        if self.debug:
            print(*args, **kwargs)
    
    def find_qx(self, rel_x: int, rel_y: int, initial_cx: int, initial_cy: int) -> int:
        """
        Find Qx using binary search.
        
        Args:
            rel_x: Current relative x coordinate (will be updated)
            rel_y: Current relative y coordinate
            initial_cx: Initial camera x coordinate
            initial_cy: Initial camera y coordinate
        
        Returns:
            Qx coordinate (relative)
        """
        self._debug_print("\n" + "="*60)
        self._debug_print("Finding Qx (x-coordinate boundary)")
        self._debug_print("="*60)
        
        # Helper function to update rel_x from current camera position
        def update_rel_x():
            nonlocal rel_x
            cx, _ = self.camera.get_position()
            rel_x = cx - initial_cx
        
        # Find initial bounds for x
        self._debug_print("\nStep 1: Finding initial bounds for x...")
        image_width, _ = self.camera.get_image_size()
        lo = float('-inf')
        hi = image_width
        
        # Check current position first
        update_rel_x()
        subimage = self.camera.get_subimage()
        result = self.check_green(subimage, binsearch_value=True)
        if self.debug_helper:
            self.debug_helper.save_debug_image(subimage, f"find_qx_init_x{rel_x}_y{rel_y}")
        
        # If we're already in green region, use current position as lo
        if result == 1 or result == 2:
            lo = rel_x
            print(f"  Already in green region at x={rel_x}, using as left bound")
        else:
            # Move left until we find the left bound (or hit boundary)
            while lo == float('-inf'):
                update_rel_x()
                subimage = self.camera.get_subimage()
                result = self.check_green(subimage, binsearch_value=True)
                if self.debug_helper:
                    self.debug_helper.save_debug_image(subimage, f"find_qx_init_x{rel_x}_y{rel_y}")
                
                # result == 1 (edge) or result == 2 (too much green) means we're in green region
                if result == 1 or result == 2:
                    lo = rel_x
                    print(f"  Found left bound at x={lo} (result={result})")
                else:
                    old_rel_x = rel_x
                    if rel_x == 0:
                        # Already at left boundary and still no green - this shouldn't happen
                        # if we properly exited green region, but handle it anyway
                        print(f"  Warning: At x=0 with no green detected. This suggests the entire image has no green at this y position.")
                        print(f"  Setting lo=0 as fallback, but binary search may not work correctly.")
                        lo = 0
                        break
                    self.camera.move_left(self.test_jump)
                    update_rel_x()
                    # Check if we actually moved (might be clamped at boundary)
                    if rel_x == old_rel_x:
                        # We're stuck at the left boundary, start from 0
                        lo = 0
                        print(f"  Hit left boundary at x=0, starting search from x=0")
                    else:
                        print(f"  Moving left by {self.test_jump} pixels, current x={rel_x} (result={result})")
        
        # Binary search for Qx
        self._debug_print("\nStep 2: Binary search for Qx...")
        for iteration in range(10):
            # Check if we've converged
            if hi - lo <= 1:
                print(f"  Converged at iteration {iteration + 1}: lo={lo}, hi={hi}")
                break
            
            mid = int((lo + hi) / 2)
            # Move to target x position: dx = mid - rel_x (relative coordinates)
            dx = mid - rel_x
            self.camera.move(dx, 0)
            update_rel_x()
            
            subimage = self.camera.get_subimage()
            result = self.check_green(subimage, binsearch_value=True)
            if self.debug_helper:
                self.debug_helper.save_debug_image(subimage, f"find_qx_iter{iteration}_x{rel_x}_y{rel_y}")
            
            print(f"  Iteration {iteration + 1}/10: x={rel_x}, result={result}, lo={lo}, hi={hi}")
            
            # result == 1 (edge) or result == 2 (too much green) means too far left
            if result == 1 or result == 2:
                lo = mid + 1  # Too far left, move right (ensure progress)
            else:
                hi = mid - 1  # Too far right (no green), move left (ensure progress)
        
        qx = int((lo + hi) / 2)
        rel_x = qx
        self._debug_print(f"\nQx found: {qx}")
        return qx
    
    def find_qy(self, rel_x: int, rel_y: int, initial_cx: int, initial_cy: int) -> int:
        """
        Find Qy using binary search.
        
        Args:
            rel_x: Current relative x coordinate
            rel_y: Current relative y coordinate (will be updated)
            initial_cx: Initial camera x coordinate
            initial_cy: Initial camera y coordinate
        
        Returns:
            Qy coordinate (relative)
        """
        self._debug_print("\n" + "="*60)
        self._debug_print("Finding Qy (y-coordinate boundary)")
        self._debug_print("="*60)
        
        # Helper function to update rel_y from current camera position
        def update_rel_y():
            nonlocal rel_y
            _, cy = self.camera.get_position()
            rel_y = cy - initial_cy
        
        # Find initial bounds for y
        self._debug_print("\nStep 1: Finding initial bounds for y...")
        _, image_height = self.camera.get_image_size()
        lo = float('-inf')
        hi = image_height
        
        # Check current position first (use horizontal/up_down edge detection)
        update_rel_y()
        subimage = self.camera.get_subimage()
        result = self.check_green(subimage, binsearch_value=True, check_direction="horizontal")
        if self.debug_helper:
            self.debug_helper.save_debug_image(subimage, f"find_qy_init_x{rel_x}_y{rel_y}")
        
        # If we're already in green region, use current position as lo
        if result == 1 or result == 2:
            lo = rel_y
            print(f"  Already in green region at y={rel_y}, using as upper bound")
        else:
            # Move up until we find the upper bound (or hit boundary)
            while lo == float('-inf'):
                update_rel_y()
                subimage = self.camera.get_subimage()
                result = self.check_green(subimage, binsearch_value=True, check_direction="horizontal")
                if self.debug_helper:
                    self.debug_helper.save_debug_image(subimage, f"find_qy_init_x{rel_x}_y{rel_y}")
                
                # result == 1 (edge) or result == 2 (too much green) means we're in green region
                if result == 1 or result == 2:
                    lo = rel_y
                    print(f"  Found upper bound at y={lo} (result={result})")
                else:
                    old_rel_y = rel_y
                    if rel_y == 0:
                        # Already at upper boundary and still no green - this shouldn't happen
                        # if we properly exited green region, but handle it anyway
                        print(f"  Warning: At y=0 with no green detected. This suggests the entire image has no green at this x position.")
                        print(f"  Setting lo=0 as fallback, but binary search may not work correctly.")
                        lo = 0
                        break
                    self.camera.move_up(self.test_jump)
                    update_rel_y()
                    # Check if we actually moved (might be clamped at boundary)
                    if rel_y == old_rel_y:
                        # We're stuck at the upper boundary, start from 0
                        lo = 0
                        print(f"  Hit upper boundary at y=0, starting search from y=0")
                    else:
                        print(f"  Moving up by {self.test_jump} pixels, current y={rel_y} (result={result})")
        
        # Binary search for Qy
        self._debug_print("\nStep 2: Binary search for Qy...")
        for iteration in range(10):
            # Check if we've converged
            if hi - lo <= 1:
                print(f"  Converged at iteration {iteration + 1}: lo={lo}, hi={hi}")
                break
            
            mid = int((lo + hi) / 2)
            # Move to target y position: dy = rel_y - mid
            # Note: move() uses dy where positive = up (cy decreases), negative = down (cy increases)
            # We want rel_y = mid, so if mid > rel_y, we need to move down (negative dy)
            # If mid < rel_y, we need to move up (positive dy)
            dy = rel_y - mid
            self.camera.move(0, dy)
            update_rel_y()
            
            subimage = self.camera.get_subimage()
            result = self.check_green(subimage, binsearch_value=True, check_direction="horizontal")
            if self.debug_helper:
                self.debug_helper.save_debug_image(subimage, f"find_qy_iter{iteration}_x{rel_x}_y{rel_y}")
            
            print(f"  Iteration {iteration + 1}/10: y={rel_y}, result={result}, lo={lo}, hi={hi}")
            
            # result == 1 (edge) or result == 2 (too much green) means too far up
            if result == 1 or result == 2:
                lo = mid + 1  # Too far up, move down (ensure progress)
            else:
                hi = mid - 1  # Too far down (no green), move up (ensure progress)
        
        qy = int((lo + hi) / 2)
        rel_y = qy
        self._debug_print(f"\nQy found: {qy}")
        return qy
    
    def calibrate(self) -> tuple[int, int]:
        """
        Perform full calibration to find Qx and Qy.
        
        Returns:
            (Qx, Qy) tuple
        """
        self._debug_print("Starting camera calibration...")
        self._debug_print(f"Image size: {self.camera.get_image_size()}")
        self._debug_print(f"Initial position: {self.camera.get_position()}")
        
        # Initialize relative coordinates (0,0) at starting position
        initial_cx, initial_cy = self.camera.get_position()
        rel_x = 0
        rel_y = 0
        
        # Helper function to update rel_x, rel_y from current camera position
        def update_rel():
            nonlocal rel_x, rel_y
            cx, cy = self.camera.get_position()
            rel_x = cx - initial_cx
            rel_y = cy - initial_cy
        
        # First, check if we're already outside the green region
        subimage = self.camera.get_subimage()
        initial_result = self.check_green(subimage, binsearch_value=True)
        self._debug_print(f"Initial position (0,0) check result: {initial_result}")
        
        # Move down and right until we get 0 from check_green (no green)
        self._debug_print("\n" + "="*60)
        self._debug_print("Step 0: Moving down and right until we exit green region")
        self._debug_print("="*60)
        
        # If we're already outside green region, we need to find where green starts
        if initial_result == 0:
            print("  Starting outside green region, moving right and down to find green...")
            # Move right and down until we enter green region, then continue until we exit
            max_iterations = 1000
            iteration = 0
            found_green = False
            while iteration < max_iterations:
                update_rel()
                subimage = self.camera.get_subimage()
                result = self.check_green(subimage, binsearch_value=True)
                if self.debug_helper:
                    self.debug_helper.save_debug_image(subimage, f"initial_move_find_green_x{rel_x}_y{rel_y}")
                
                if result != 0 and not found_green:
                    found_green = True
                    print(f"  Entered green region at x={rel_x}, y={rel_y}")
                
                if found_green and result == 0:
                    print(f"  Exited green region at x={rel_x}, y={rel_y}")
                    break
                
                old_rel_x, old_rel_y = rel_x, rel_y
                self.camera.move_right(self.test_jump)
                self.camera.move_down(self.test_jump)
                update_rel()
                
                if rel_x == old_rel_x and rel_y == old_rel_y:
                    print(f"  Cannot move further (at boundary), current position: x={rel_x}, y={rel_y}")
                    break
                
                iteration += 1
        else:
            # We're in green region, move down and right until we exit
            max_iterations = 1000
            iteration = 0
            while iteration < max_iterations:
                update_rel()
                subimage = self.camera.get_subimage()
                result = self.check_green(subimage, binsearch_value=True)
                if self.debug_helper:
                    self.debug_helper.save_debug_image(subimage, f"initial_move_x{rel_x}_y{rel_y}")
                
                if result == 0:
                    print(f"  Exited green region at x={rel_x}, y={rel_y}")
                    break
                
                old_rel_x, old_rel_y = rel_x, rel_y
                self.camera.move_down(self.test_jump)
                self.camera.move_right(self.test_jump)
                update_rel()
                
                if rel_x == old_rel_x and rel_y == old_rel_y:
                    print(f"  Cannot move further (at boundary), current position: x={rel_x}, y={rel_y}")
                    break
                
                print(f"  Still in green region at x={rel_x}, y={rel_y}, moving down and right... (result={result})")
                iteration += 1
        
        # Find Qx (keeping current y position)
        qx = self.find_qx(rel_x, rel_y, initial_cx, initial_cy)
        
        # Find Qy (keeping current x position from Qx search)
        qy = self.find_qy(rel_x, rel_y, initial_cx, initial_cy)
        
        return (qx, qy)

if __name__ == "__main__":
    import sys
    
    camera = CameraTester(
        image_path="test_borders.png",
        initial_x=0,
        initial_y=0
    )
    calibrator = CameraCalibrator(
        camera=camera,
        kernel_size=13,  # Must be 13 for the new kernel pattern
        test_jump=50,
        debug=False
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
        print(f"Qx={qx}, Qy={qy}")
