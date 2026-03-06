import cv2
import numpy as np
import torch
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
from transformers import SamModel, SamProcessor
from PIL import Image
import random

def extract_and_plot_sam(image_path, output_plot_path, viz_path):
    print("Loading image...")
    image = cv2.imread(image_path)
    if image is None: return

    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    pil_img = Image.fromarray(image_rgb)

    print("Detecting grid structure via projection...")
    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    kernel = np.ones((5,5),np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, kernel)

    row_sums = np.sum(thresh, axis=1)
    col_sums = np.sum(thresh, axis=0)

    row_sums = np.convolve(row_sums, np.ones(20)/20, mode='same')
    col_sums = np.convolve(col_sums, np.ones(20)/20, mode='same')

    y_peaks, y_props = find_peaks(row_sums, distance=50, prominence=row_sums.max()*0.2)
    x_peaks, x_props = find_peaks(col_sums, distance=50, prominence=col_sums.max()*0.2)

    if len(y_peaks) > 3:
        idx = np.argsort(y_props['prominences'])[-3:]
        y_peaks = np.sort(y_peaks[idx])
    if len(x_peaks) > 9:
        idx = np.argsort(x_props['prominences'])[-9:]
        x_peaks = np.sort(x_peaks[idx])

    if len(y_peaks) < 3 or len(x_peaks) < 9:
        print("Failed to find valid grid.")
        return

    print("Initializing SAM model...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)

    input_points = []
    metadata = []
    
    for r_idx, y in enumerate(y_peaks):
        for c_idx, x in enumerate(x_peaks):
            if 0 < c_idx < 8:
                input_points.append([int(x), int(y)])
                metadata.append({'Row': r_idx, 'Column': c_idx})

    print(f"Generating masks for {len(input_points)} valid sample locations...")
    
    data = []
    viz_img = image.copy()
    viz_mask_overlay = np.zeros_like(image, dtype=np.uint8)
    
    for pt, meta in zip(input_points, metadata):
        inputs = processor(pil_img, input_points=[[[pt]]], return_tensors="pt").to(device)
        with torch.no_grad():
            outputs = model(**inputs)
        
        masks = processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
        )
        scores = outputs.iou_scores.cpu().squeeze()
        
        if len(masks) > 0:
            # masks[0] shape: (1, 3, H, W)
            # We take the mask with the highest IOU score
            best_mask_idx = torch.argmax(scores).item()
            mask = masks[0][0][best_mask_idx].numpy().astype(bool)
            
            # Use a solid, extremely distinct color for all masks (e.g. vibrant magenta)
            color = (255, 0, 255) # BGR
            viz_mask_overlay[mask] = color
            cv2.circle(viz_img, (pt[0], pt[1]), 3, (0, 0, 0), -1) # Black center dot to distinguish from mask color

            rgb_pixels = image_rgb[mask]
            lab_pixels = image_lab[mask]
            hsv_pixels = image_hsv[mask]
            
            if len(rgb_pixels) > 0:
                data.append({
                    'Row': meta['Row'],
                    'Column': meta['Column'],
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
            print(f"Warning: No mask found for {meta}")

    print("Saving visualization...")
    cv2.addWeighted(viz_img, 0.7, viz_mask_overlay, 0.5, 0, viz_img)
    cv2.imwrite(viz_path, viz_img)

    print("Generating plots...")
    df = pd.DataFrame(data)
    col_avg = df.groupby('Column').mean().reset_index()

    fig, axes = plt.subplots(3, 1, figsize=(10, 15), sharex=True)
    
    axes[0].plot(col_avg['Column'], col_avg['R_median'], marker='o', color='red', linestyle='--', label='R (median)')
    axes[0].plot(col_avg['Column'], col_avg['G_median'], marker='o', color='green', linestyle='--', label='G (median)')
    axes[0].plot(col_avg['Column'], col_avg['B_median'], marker='o', color='blue', linestyle='--', label='B (median)')
    axes[0].set_title('RGB (SAM Median Extraction)')
    axes[0].set_ylabel('Intensity')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(col_avg['Column'], col_avg['L_median'], marker='o', color='black', linestyle='--', label='L* (median)')
    axes[1].plot(col_avg['Column'], col_avg['A_median'], marker='o', color='magenta', linestyle='--', label='a* (median)')
    axes[1].plot(col_avg['Column'], col_avg['B_lab_median'], marker='o', color='orange', linestyle='--', label='b* (median)')
    axes[1].set_title('CIELAB (SAM Median Extraction)')
    axes[1].set_ylabel('Value')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)

    axes[2].plot(col_avg['Column'], col_avg['H_median'], marker='o', color='cyan', linestyle='--', label='Hue (median)')
    axes[2].plot(col_avg['Column'], col_avg['S_median'], marker='o', color='gray', linestyle='--', label='Sat (median)')
    axes[2].plot(col_avg['Column'], col_avg['V_median'], marker='o', color='purple', linestyle='--', label='Val (median)')
    axes[2].set_title('HSV (SAM Median Extraction)')
    axes[2].set_xlabel('Column Index (1 to 7)')
    axes[2].set_ylabel('Value')
    axes[2].legend()
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xticks(range(1, 8))

    plt.tight_layout()
    plt.savefig(output_plot_path)
    print(f"Saved SAM plot to {output_plot_path}")

extract_and_plot_sam('/home/ash/Desktop/acads/pcia/samples/25th_min.jpeg', '/home/ash/Desktop/acads/pcia/sam_color_analysis.png', '/home/ash/Desktop/acads/pcia/sam_segmentation_viz.png')
