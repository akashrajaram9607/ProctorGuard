import cv2
import numpy as np
import os

class HeatmapGenerator:
    def __init__(self, width=1280, height=720, output_dir="logs"):
        self.width = width
        self.height = height
        self.output_dir = output_dir
        
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        
    def generate(self, gaze_points, filename="gaze_heatmap.jpg"):
        """
        Generates a heatmap from a list of (x, y) coordinates.
        Returns: (image_data, file_path)
        """
        # 1. Handle Empty Data
        if not gaze_points:
            # Return None, None so app.py doesn't crash
            return None, None

        # 2. Create Blank Image (Black Background)
        heatmap = np.zeros((self.height, self.width), dtype=np.float32)

        # 3. Add Heat Points
        for point in gaze_points:
            x, y = point
            # Ensure x and y are valid integers
            if 0 <= x < self.width and 0 <= y < self.height:
                # Create a temporary mask for one "look"
                temp_mask = np.zeros((self.height, self.width), dtype=np.float32)
                # Draw a white circle (intensity 1.0)
                cv2.circle(temp_mask, (int(x), int(y)), 40, (1.0), -1)
                # Add to main heatmap
                heatmap = cv2.add(heatmap, temp_mask)

        # 4. Smooth it out (Gaussian Blur)
        heatmap = cv2.GaussianBlur(heatmap, (99, 99), 0)

        # 5. Normalize (Scale values to 0-255 range)
        heatmap = cv2.normalize(heatmap, None, 0, 255, cv2.NORM_MINMAX)
        heatmap = np.uint8(heatmap)

        # 6. Colorize (Blue = Cold, Red = Hot)
        colored_heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)

        # 7. Save to File
        output_path = os.path.join(self.output_dir, filename)
        cv2.imwrite(output_path, colored_heatmap)
        
        # 8. RETURN EXACTLY 2 VALUES
        return colored_heatmap, output_path