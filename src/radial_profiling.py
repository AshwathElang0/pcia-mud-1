"""
Radial Color Profiling
======================
This script analyzes the spatial gradient within each sample well by dividing
each disk into 5 concentric rings (annuli) and tracking the color shift 
independently in each ring over time.
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES_DIR = os.path.join(BASE_DIR, 'samples')
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'statistical')

def compute_radial_profiles(timepoints=[0, 5, 10, 15, 20, 25], image_prefix="th_min.jpeg"):
    all_radial_data = []

    for t in timepoints:
        image_path = os.path.join(SAMPLES_DIR, f"{t}{image_prefix}")
        if not os.path.exists(image_path):
            continue

        print(f"Processing timepoint: {t} mins for radial profiling")
        image = cv2.imread(image_path)
        image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Grid detection (Simplified baseline)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        y_peaks, _ = find_peaks(np.sum(thresh, axis=1), distance=50, prominence=5000)
        x_peaks, _ = find_peaks(np.sum(thresh, axis=0), distance=50, prominence=5000)
        
        if len(y_peaks) >= 3 and len(x_peaks) >= 9:
            y_peaks = np.sort(y_peaks[np.argsort(np.sum(thresh, axis=1)[y_peaks])[-3:]])
            x_peaks = np.sort(x_peaks[np.argsort(np.sum(thresh, axis=0)[x_peaks])[-9:]])
            
            radius = 32
            num_rings = 5
            ring_width = radius / num_rings

            for r_idx, y in enumerate(y_peaks):
                for c_idx, x in enumerate(x_peaks):
                    if 0 < c_idx < 8:
                        Y, X = np.ogrid[:image.shape[0], :image.shape[1]]
                        dist_sq = (X - x)**2 + (Y - y)**2
                        dist = np.sqrt(dist_sq)

                        for ring_idx in range(num_rings):
                            r_inner = ring_idx * ring_width
                            r_outer = (ring_idx + 1) * ring_width
                            
                            ring_mask = (dist >= r_inner) & (dist < r_outer)
                            ring_pixels = image_lab[ring_mask]
                            
                            if len(ring_pixels) > 0:
                                all_radial_data.append({
                                    'Time': t,
                                    'Row': r_idx,
                                    'Column': c_idx,
                                    'Ring': ring_idx,
                                    'A_median': np.median(ring_pixels[:, 1]),
                                    'L_median': np.median(ring_pixels[:, 0])
                                })

    df = pd.DataFrame(all_radial_data)
    
    # Visualization: Heatmap of Color Shift over Time for a specific Column
    # We'll pick Column 6 (the active one) to demonstrate the radial effect
    active_col = 6
    col_df = df[df['Column'] == active_col].groupby(['Time', 'Ring']).mean().reset_index()
    
    pivot_A = col_df.pivot(index='Ring', columns='Time', values='A_median')
    
    plt.figure(figsize=(10, 6))
    import seaborn as sns
    sns.heatmap(pivot_A, annot=True, fmt=".1f", cmap="magma")
    plt.title(f'Radial a* Profile Over Time (Column {active_col})')
    plt.xlabel('Time (min)')
    plt.ylabel('Ring Index (0=Center, 4=Periphery)')
    
    heatmap_path = os.path.join(RESULTS_DIR, 'radial_heatmaps.png')
    plt.savefig(heatmap_path, dpi=150)
    print(f"Saved radial heatmap to {heatmap_path}")

    # Plot Radial Profiles for all columns at final timepoint
    t_final = timepoints[-1]
    final_df = df[df['Time'] == t_final].groupby(['Column', 'Ring']).mean().reset_index()
    
    fig, ax = plt.subplots(figsize=(10, 6))
    for col in sorted(final_df['Column'].unique()):
        data = final_df[final_df['Column'] == col]
        ax.plot(data['Ring'], data['A_median'], marker='o', label=f'Col {col}')
    
    ax.set_title(f'Radial Gradient at t={t_final}')
    ax.set_xlabel('Ring (0=Center to 4=Periphery)')
    ax.set_ylabel('a* Median')
    ax.legend(title='Column', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    gradient_plot_path = os.path.join(RESULTS_DIR, 'radial_gradients_final.png')
    plt.savefig(gradient_plot_path, dpi=150)
    print(f"Saved radial gradient plot to {gradient_plot_path}")

    df.to_csv(os.path.join(RESULTS_DIR, 'radial_profiling_data.csv'), index=False)

if __name__ == "__main__":
    compute_radial_profiles()
