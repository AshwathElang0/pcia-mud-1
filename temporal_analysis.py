import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os

def analyze_temporal_images(image_prefix="th_min.jpeg", timepoints=[0, 5, 10, 15, 20, 25]):
    all_data = []

    for t in timepoints:
        image_path = f"/home/ash/Desktop/acads/pcia/samples/{t}{image_prefix}"
        if not os.path.exists(image_path):
            print(f"Warning: Could not find image {image_path}. Skipping.")
            continue

        print(f"Processing timepoint: {t} mins")
        image = cv2.imread(image_path)
        
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
        
        # Heuristic to find the best 3 rows and 9 columns
        if len(y_peaks) >= 3 and len(x_peaks) >= 9:
            if len(y_peaks) > 3:
                idx = np.argsort(y_props['prominences'])[-3:]
                y_peaks = np.sort(y_peaks[idx])
            if len(x_peaks) > 9:
                idx = np.argsort(x_props['prominences'])[-9:]
                x_peaks = np.sort(x_peaks[idx])

            radius = 32

            for r_idx, y in enumerate(y_peaks):
                for c_idx, x in enumerate(x_peaks):
                    if 0 < c_idx < 8:
                        mask = np.zeros(gray.shape, dtype=np.uint8)
                        cv2.circle(mask, (x, y), radius, 255, -1)
                        
                        rgb_pixels = image_rgb[mask == 255]
                        lab_pixels = image_lab[mask == 255]
                        hsv_pixels = image_hsv[mask == 255]
                        
                        if len(rgb_pixels) > 0:
                            all_data.append({
                                'Time': t,
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
        else:
            print(f"Error: Grid tracking failed for {image_path}")

    # Aggregate by timepoint and column
    df = pd.DataFrame(all_data)
    if df.empty:
        print("No data extracted.")
        return

    # Averages across rows for each column at each timepoint
    grouped_df = df.groupby(['Time', 'Column']).mean().reset_index()

    # Plot temporal trends per column
    fig, axes = plt.subplots(3, 3, figsize=(15, 12), sharex=True)
    fig.suptitle('Temporal Colorimetric Trajectories by Column', fontsize=16)

    columns_to_plot = sorted(grouped_df['Column'].unique())
    colors = plt.cm.viridis(np.linspace(0, 1, len(columns_to_plot)))
    
    # Store standard labels 
    channels = [
        ('R_median', 'Red Intensity', axes[0, 0]),
        ('G_median', 'Green Intensity', axes[0, 1]),
        ('B_median', 'Blue Intensity', axes[0, 2]),
        ('L_median', 'L* (Lightness)', axes[1, 0]),
        ('A_median', 'a* (Green-Red)', axes[1, 1]),
        ('B_lab_median', 'b* (Blue-Yellow)', axes[1, 2]),
        ('H_median', 'Hue', axes[2, 0]),
        ('S_median', 'Saturation', axes[2, 1]),
        ('V_median', 'Value', axes[2, 2])
    ]

    for col, color in zip(columns_to_plot, colors):
        col_data = grouped_df[grouped_df['Column'] == col]
        
        for feature, title, ax in channels:
            ax.plot(col_data['Time'], col_data[feature], marker='o', color=color, label=f'Col {col}')
            ax.set_title(title)
            ax.grid(True, alpha=0.3)
            
            # Subplot aesthetic cleanup
            if ax in [axes[2, 0], axes[2, 1], axes[2, 2]]:
                ax.set_xlabel('Time (mins)')

    # Add legend to the last plot cleanly
    handles, labels = axes[0,0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower right', bbox_to_anchor=(0.95, 0.05), ncol=7)

    plt.tight_layout(rect=[0, 0.08, 1, 0.96]) # Leave room for suptitle and legend
    output_plot_path = '/home/ash/Desktop/acads/pcia/temporal_color_trends.png'
    plt.savefig(output_plot_path)
    print(f"Saved temporal trends to {output_plot_path}")

    # Optionally dump to CSV
    df.to_csv('/home/ash/Desktop/acads/pcia/temporal_data.csv', index=False)
    print("Saved temporal_data.csv")

if __name__ == "__main__":
    analyze_temporal_images()
