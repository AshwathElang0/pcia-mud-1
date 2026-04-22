import os
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from glob import glob
import re

# Paths
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
SAMPLES_DIR = os.path.join(ROOT_DIR, "broth_samples")
MASKS_DIR = os.path.join(ROOT_DIR, "results", "broth")
ANALYSIS_DIR = os.path.join(ROOT_DIR, "results", "analysis")
os.makedirs(ANALYSIS_DIR, exist_ok=True)

# Concentrations from note (Col 1 to Col 10)
CONCENTRATIONS = [128, 64, 32, 16, 8, 4, 2, 1, 0.5, 0.25]

def extract_time(filename):
    """Extract minute from filenames like min15.png or min15_masks.npy"""
    match = re.search(r'min(\d+)', filename)
    return int(match.group(1)) if match else 0

def compute_median_lab(image_rgb, mask):
    """Calculate median Lab values within the mask."""
    # Convert RGB to Lab
    # In OpenCV, 8-bit Lab values are scaled to [0, 255]
    image_lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    pixels = image_lab[mask]
    if len(pixels) == 0:
        return np.array([np.nan, np.nan, np.nan])
    return np.median(pixels, axis=0)

def main():
    data = []
    
    # Find all mask files
    mask_files = sorted(glob(os.path.join(MASKS_DIR, "*_masks.npy")), key=extract_time)
    
    if not mask_files:
        print(f"No mask files found in {MASKS_DIR}. Run segmentation first.")
        return

    for mask_path in mask_files:
        time = extract_time(mask_path)
        img_path = os.path.join(SAMPLES_DIR, f"min{time}.png")
        
        if not os.path.exists(img_path):
            print(f"Skipping {mask_path}, image not found at {img_path}")
            continue
            
        print(f"Analyzing Time: {time} min...")
        image_bgr = cv2.imread(img_path)
        if image_bgr is None:
            continue
        image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        masks = np.load(mask_path)
        
        # 20 masks: Row 1 (0-9), Row 2 (10-19)
        for idx, mask in enumerate(masks):
            row = 1 if idx < 10 else 2
            col = (idx % 10) + 1
            conc = CONCENTRATIONS[col - 1]
            
            lab_vals = compute_median_lab(image_rgb, mask)
            
            data.append({
                "Time": time,
                "Row": f"Row {row}",
                "Column": col,
                "Concentration": conc,
                "L*": lab_vals[0],
                "a*": lab_vals[1],
                "b*": lab_vals[2],
                # a* represents pinkness/redness contrast in Lab space
                "Pinkness": lab_vals[1] 
            })
            
    df = pd.DataFrame(data)
    
    # Save CSV
    csv_path = os.path.join(ANALYSIS_DIR, "pinkness_stats.csv")
    df.to_csv(csv_path, index=False)
    print(f"Saved stats to {csv_path}")
    
    # --- Plotting ---
    sns.set_theme(style="whitegrid")
    
    # 1. Pinkness over Time (per Concentration)
    plt.figure(figsize=(12, 7))
    # Filter for Row 1 and Row 2 to plot as separate lines if needed, or grouped
    sns.lineplot(
        data=df, 
        x="Time", 
        y="Pinkness", 
        hue="Concentration", 
        style="Row", 
        palette="flare", 
        markers=True,
        dashes=False
    )
    plt.title("Pinkness (Lab a*) Over Time per Concentration")
    plt.ylabel("Pinkness (CIE a* channel value)")
    plt.xlabel("Time (minutes)")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Conc (ug/mL)")
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plot1_path = os.path.join(ANALYSIS_DIR, "pinkness_over_time.png")
    plt.savefig(plot1_path, dpi=200)
    plt.close()
    
    # 2. Pinkness vs Concentration (at each Time Point)
    plt.figure(figsize=(12, 7))
    sns.lineplot(
        data=df, 
        x="Concentration", 
        y="Pinkness", 
        hue="Time", 
        style="Row", 
        palette="viridis", 
        markers=True,
        dashes=False
    )
    plt.xscale("log")
    plt.title("Pinkness vs. Concentration Across Time Points")
    plt.ylabel("Pinkness (CIE a* channel value)")
    plt.xlabel("Concentration (ug/mL) - Log Scale")
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', title="Time (min)")
    plt.grid(True, which="both", ls="-", alpha=0.5)
    plt.tight_layout()
    plot2_path = os.path.join(ANALYSIS_DIR, "pinkness_concentration_profile.png")
    plt.savefig(plot2_path, dpi=200)
    plt.close()
    
    print(f"Saved plots to {ANALYSIS_DIR}")

if __name__ == "__main__":
    main()
