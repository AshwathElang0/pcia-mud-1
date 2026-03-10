import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks

def extract_and_plot_colors(image_path, output_plot_path):
    image = cv2.imread(image_path)
    if image is None: return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5,5),np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    row_sums = np.sum(thresh, axis=1)
    col_sums = np.sum(thresh, axis=0)

    row_sums = np.convolve(row_sums, np.ones(20)/20, mode='same')
    col_sums = np.convolve(col_sums, np.ones(20)/20, mode='same')

    y_peaks, y_props = find_peaks(row_sums, distance=50, prominence=row_sums.max()*0.2)
    x_peaks, x_props = find_peaks(col_sums, distance=50, prominence=col_sums.max()*0.2)

    radius = 32 # Increased to match sample size accurately

    if len(y_peaks) >= 3 and len(x_peaks) >= 9:
        if len(y_peaks) > 3:
            idx = np.argsort(y_props['prominences'])[-3:]
            y_peaks = np.sort(y_peaks[idx])
        if len(x_peaks) > 9:
            idx = np.argsort(x_props['prominences'])[-9:]
            x_peaks = np.sort(x_peaks[idx])

        data = []
        out_img = image.copy()

        for r_idx, y in enumerate(y_peaks):
            for c_idx, x in enumerate(x_peaks):
                if 0 < c_idx < 8:
                    mask = np.zeros(gray.shape, dtype=np.uint8)
                    cv2.circle(mask, (x, y), radius, 255, -1)

                    rgb_pixels = image_rgb[mask == 255]
                    lab_pixels = image_lab[mask == 255]
                    hsv_pixels = image_hsv[mask == 255]

                    cv2.circle(out_img, (x, y), radius, (0, 255, 0), 4)

                    if len(rgb_pixels) > 0:
                        data.append({
                            'Row': r_idx,
                            'Column': c_idx,
                            'R_median': np.median(rgb_pixels[:, 0]),
                            'G_median': np.median(rgb_pixels[:, 1]),
                            'B_median': np.median(rgb_pixels[:, 2]),
                            'L_median': np.median(lab_pixels[:, 0]),
                            'A_median': np.median(lab_pixels[:, 1]),
                            'B_lab_median': np.median(lab_pixels[:, 2]),
                            'H_median': np.median(hsv_pixels[:, 0]),
                            'S_median': np.median(hsv_pixels[:, 1]),
                            'V_median': np.median(hsv_pixels[:, 2])
                        })

        cv2.imwrite('baseline_grid_detections.png', out_img)

        df = pd.DataFrame(data)
        col_avg = df.groupby('Column').mean().reset_index()

        fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)

        # RGB Medians
        axes[0].plot(col_avg['Column'], col_avg['R_median'], marker='o', color='red', linestyle='--', label='R (median)')
        axes[0].plot(col_avg['Column'], col_avg['G_median'], marker='o', color='green', linestyle='--', label='G (median)')
        axes[0].plot(col_avg['Column'], col_avg['B_median'], marker='o', color='blue', linestyle='--', label='B (median)')
        axes[0].set_title('RGB (Baseline Projection + Median Extraction)')
        axes[0].set_ylabel('Intensity')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)

        # LAB Medians
        axes[1].plot(col_avg['Column'], col_avg['L_median'], marker='o', color='black', linestyle='--', label='L* (median)')
        axes[1].plot(col_avg['Column'], col_avg['A_median'], marker='o', color='magenta', linestyle='--', label='a* (median)')
        axes[1].plot(col_avg['Column'], col_avg['B_lab_median'], marker='o', color='orange', linestyle='--', label='b* (median)')
        axes[1].set_title('CIELAB (Baseline Projection + Median Extraction)')
        axes[1].set_ylabel('Value')
        axes[1].legend()
        axes[1].grid(True, alpha=0.3)

        # HSV Medians
        axes[2].plot(col_avg['Column'], col_avg['H_median'], marker='o', color='cyan', linestyle='--', label='Hue (median)')
        axes[2].plot(col_avg['Column'], col_avg['S_median'], marker='o', color='gray', linestyle='--', label='Sat (median)')
        axes[2].plot(col_avg['Column'], col_avg['V_median'], marker='o', color='purple', linestyle='--', label='Val (median)')
        axes[2].set_title('HSV (Baseline Projection + Median Extraction)')
        axes[2].set_xlabel('Column Index (1 to 7)')
        axes[2].set_ylabel('Value')
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        axes[2].set_xticks(range(1, 8))

        plt.tight_layout()
        plt.savefig(output_plot_path)
        print(f"Saved comparison plot to {output_plot_path}")

extract_and_plot_colors('samples/25th_min.jpeg', 'baseline_color_analysis.png')
