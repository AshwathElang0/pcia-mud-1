"""
Intra-Disk Region Detection
=============================
This script analyzes the spatial distribution of color *within* each sample disk
across the time series. It uses K-Means clustering on the pixels inside each 
well to segment the disk into "Reaction Patches" (pink/red) vs "Unreacted Patches" (blue).
"""

import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from sklearn.cluster import KMeans
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES_DIR = os.path.join(BASE_DIR, 'samples')
RESULTS_DIR_TEMP = os.path.join(BASE_DIR, 'results', 'temporal')
RESULTS_DIR_STAT = os.path.join(BASE_DIR, 'results', 'statistical')

def segment_intra_disk_regions(timepoints=[0, 5, 10, 15, 20, 25], image_prefix="th_min.jpeg"):
    all_region_data = []

    for t in timepoints:
        image_path = os.path.join(SAMPLES_DIR, f"{t}{image_prefix}")
        if not os.path.exists(image_path):
            print(f"Warning: Could not find image {image_path}. Skipping.")
            continue

        print(f"Processing timepoint: {t} mins for intra-disk regions")
        image = cv2.imread(image_path)
        
        # Color spaces
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
        
        # Grid detection
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
        
        if len(y_peaks) >= 3 and len(x_peaks) >= 9:
            if len(y_peaks) > 3:
                y_peaks = np.sort(y_peaks[np.argsort(y_props['prominences'])[-3:]])
            if len(x_peaks) > 9:
                x_peaks = np.sort(x_peaks[np.argsort(x_props['prominences'])[-9:]])

            radius = 32
            
            # Setup visualization for the final timepoint
            viz_img = image.copy() if t == timepoints[-1] else None

            for r_idx, y in enumerate(y_peaks):
                for c_idx, x in enumerate(x_peaks):
                    if 0 < c_idx < 8: # Active samples
                        
                        # Extract coordinates in the circular mask
                        Y, X = np.ogrid[:image.shape[0], :image.shape[1]]
                        dist_from_center = np.sqrt((X - x)**2 + (Y - y)**2)
                        mask = dist_from_center <= radius
                        
                        # Extract the LAB pixels for segmentation
                        # LAB is generally better for human-perceived color thresholding (e.g. pink vs blue)
                        lab_pixels = image_lab[mask].astype(float)
                        
                        if len(lab_pixels) < 10:
                            continue
                            
                        # Use just the a* (Green-Red) and b* (Blue-Yellow) channels for clustering
                        # We ignore L* (Lightness) to restrict shading/glare artifacts
                        ab_pixels = lab_pixels[:, 1:3]
                        
                        # Cluster into 2 regions: "Reaction Patches" (Pink/Red) vs "Unreacted" (Blue)
                        # We force n_init to suppress warnings
                        kmeans = KMeans(n_clusters=2, n_init=5, random_state=42)
                        labels = kmeans.fit_predict(ab_pixels)
                        
                        # Which cluster is the "reaction" cluster? 
                        # Higher a* value = more red/pink (reaction). Lower a* = more green/blue (unreacted).
                        cluster_centers = kmeans.cluster_centers_  # shape: (2, 2) where cols are [a*, b*]
                        reaction_cluster_idx = np.argmax(cluster_centers[:, 0]) # The center with higher a* is pinkish
                        unreacted_cluster_idx = 1 - reaction_cluster_idx
                        
                        # Count the number of pixels in each cluster
                        total_pixels = len(labels)
                        reaction_pixels = np.sum(labels == reaction_cluster_idx)
                        reaction_area_pct = (reaction_pixels / total_pixels) * 100
                        
                        # Calculate the mean RGB color of the reaction patches strictly for visualization/logging
                        rgb_pixels = image_rgb[mask]
                        reaction_rgb = np.median(rgb_pixels[labels == reaction_cluster_idx], axis=0) if reaction_pixels > 0 else [0,0,0]
                        
                        all_region_data.append({
                            'Time': t,
                            'Row': r_idx,
                            'Column': c_idx,
                            'Reaction_Area_Pct': reaction_area_pct,
                            'Reaction_a_star_intensity': cluster_centers[reaction_cluster_idx, 0]
                        })
                        
                        # If this is the final timepoint, annotate the visualization image
                        if viz_img is not None:
                            # Reconstruct the 2D label mask
                            full_label_mask = np.zeros(image.shape[:2], dtype=np.uint8)
                            full_label_mask[mask] = (labels == reaction_cluster_idx).astype(np.uint8) * 255
                            
                            # Create a vivid red overlay for the reaction patches
                            red_overlay = np.zeros_like(viz_img)
                            red_overlay[full_label_mask == 255] = [0, 0, 255] # BGR Red
                            
                            # Only blend the specific reaction pixels for THIS well
                            well_reaction_mask = full_label_mask == 255
                            viz_img[well_reaction_mask] = cv2.addWeighted(
                                viz_img[well_reaction_mask], 0.4, 
                                red_overlay[well_reaction_mask], 0.6, 0
                            )
                            
                            # Draw outer well boundary
                            cv2.circle(viz_img, (x, y), radius, (0, 255, 0), 2)
            
            # Save the final timepoint segmentation map
            if viz_img is not None:
                viz_path = os.path.join(RESULTS_DIR_STAT, 'intra_disk_segmentation_t25.png')
                cv2.imwrite(viz_path, viz_img)
                print(f"Saved intra-disk segmentation map to {viz_path}")
        else:
            print(f"Error: Grid tracking failed for {image_path}")

    # Process and Plot the Temporal Data
    df = pd.DataFrame(all_region_data)
    if df.empty:
        print("No region data extracted.")
        return

    # Average the Area % across the 3 replicate rows
    grouped_df = df.groupby(['Time', 'Column']).mean().reset_index()

    # Plot the Reaction Area Trajectories
    fig, ax = plt.subplots(figsize=(10, 6))
    plt.title('Intra-Disk Reaction Kinetics: Biological Pink Blob Area (%) Over Time', fontsize=14)
    
    columns_list = sorted(grouped_df['Column'].unique())
    colors = plt.cm.plasma(np.linspace(0, 0.9, len(columns_list)))

    for col, color in zip(columns_list, colors):
        col_data = grouped_df[grouped_df['Column'] == col]
        ax.plot(col_data['Time'], col_data['Reaction_Area_Pct'], marker='o', linewidth=2, color=color, label=f'Col {col}')
        
    ax.set_xlabel('Time (mins)', fontsize=12)
    ax.set_ylabel('Reaction Patch Area (%)', fontsize=12)
    ax.legend(title='Column', bbox_to_anchor=(1.05, 1), loc='upper left')
    ax.grid(True, alpha=0.3)
    ax.set_ylim(-5, 105) # Percentages 0-100
    
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR_STAT, 'spatial_reaction_kinetics.png')
    plt.savefig(plot_path, dpi=150)
    print(f"Saved spatial reaction trajectories to {plot_path}")

    csv_path = os.path.join(RESULTS_DIR_STAT, 'spatial_region_data.csv')
    df.to_csv(csv_path, index=False)
    print(f"Saved {csv_path}")

if __name__ == "__main__":
    segment_intra_disk_regions()
