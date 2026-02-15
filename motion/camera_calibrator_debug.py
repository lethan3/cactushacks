"""
Debug utilities for camera calibration.
These functions provide visualization and analysis tools for debugging calibration.
"""
import numpy as np
from PIL import Image, ImageDraw


class CameraCalibratorDebugMixin:
    """Mixin class providing debug utilities for camera calibration."""
    
    def save_debug_image(self, subimage: Image.Image, label: str):
        """
        Save full image with red box around subimage area and annotations inside.
        
        Args:
            subimage: The subimage that was captured
            label: Label for the debug image filename
        """
        if not self.debug:
            return
        
        filename = f"step_{self.step_count:03d}_{label}.png"
        filepath = self.debug_dir / filename
        
        # Load the full image
        full_image = Image.open(self.camera.image_path).copy()
        draw = ImageDraw.Draw(full_image)
        
        # Draw red rectangle around the subimage area
        x1 = self.camera.cx
        y1 = self.camera.cy
        x2 = self.camera.cx + self.camera.subimage_width
        y2 = self.camera.cy + self.camera.subimage_height
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
        if self.debug:
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
        if not self.debug:
            return
        
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
        
        if self.debug:
            print(f"  Saved activation images: {vertical_filename}, {horizontal_filename}")
    
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
                subimage = self.camera.get_subimage()
                
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
        from PIL import ImageDraw
        
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
