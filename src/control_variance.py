import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from glob import glob
import torch
from transformers import SamModel, SamProcessor
from PIL import Image

CONTROL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'samples', 'control')
RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'results', 'statistical')
os.makedirs(RESULTS_DIR, exist_ok=True)

image_files = sorted(glob(os.path.join(CONTROL_DIR, '*.png')))

N_ROWS = 2
N_COLS = 10

# Helper to extract time from filename
def extract_time(fname):
    try:
        return int(os.path.basename(fname).split('th')[0])
    except Exception:
        return -1

def detect_samples_sam(image_rgb):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
    pil_img = Image.fromarray(image_rgb)
    h, w = image_rgb.shape[:2]
    # Estimate grid points for 2x10 layout
    row_ys = np.linspace(h/6, h*5/6, N_ROWS)
    col_xs = np.linspace(w/20, w*19/20, N_COLS)
    input_points = [[int(x), int(y)] for y in row_ys for x in col_xs]
    input_points = np.array(input_points).reshape(N_ROWS, N_COLS, 2)
    masks = []
    for r in range(N_ROWS):
        row_masks = []
        for c in range(N_COLS):
            pt = input_points[r, c]
            inputs = processor(pil_img, input_points=[[[pt.tolist()]]], return_tensors="pt").to(device)
            with torch.no_grad():
                outputs = model(**inputs)
            mask = processor.image_processor.post_process_masks(
                outputs.pred_masks.cpu(), inputs["original_sizes"].cpu(), inputs["reshaped_input_sizes"].cpu()
            )[0][0][torch.argmax(outputs.iou_scores.cpu().squeeze()).item()].numpy().astype(bool)
            row_masks.append(mask)
        masks.append(row_masks)
    return np.array(masks)

# Store results
timepoints = []
row_variances_mean = {0: [], 1: []}
row_variances_median = {0: [], 1: []}

for img_path in image_files:
    img = cv2.imread(img_path)
    if img is None:
        continue
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    masks = detect_samples_sam(img_rgb)
    sample_means = np.zeros((N_ROWS, N_COLS, 3))
    sample_medians = np.zeros((N_ROWS, N_COLS, 3))
    for r in range(N_ROWS):
        for c in range(N_COLS):
            mask = masks[r, c]
            pixels = img_rgb[mask]
            if len(pixels) > 0:
                sample_means[r, c] = np.mean(pixels, axis=0)
                sample_medians[r, c] = np.median(pixels, axis=0)
            else:
                sample_means[r, c] = np.nan
                sample_medians[r, c] = np.nan
    for r in range(N_ROWS):
        row_variances_mean[r].append(np.nanvar(sample_means[r], axis=0).mean())
        row_variances_median[r].append(np.nanvar(sample_medians[r], axis=0).mean())
    t = extract_time(img_path)
    timepoints.append(t)

sorted_idx = np.argsort(timepoints)
timepoints = np.array(timepoints)[sorted_idx]
row_variances_mean[0] = np.array(row_variances_mean[0])[sorted_idx]
row_variances_mean[1] = np.array(row_variances_mean[1])[sorted_idx]
row_variances_median[0] = np.array(row_variances_median[0])[sorted_idx]
row_variances_median[1] = np.array(row_variances_median[1])[sorted_idx]

plt.figure(figsize=(10,6))
plt.plot(timepoints, row_variances_mean[0], marker='o', label='Dye row (mean)')
plt.plot(timepoints, row_variances_mean[1], marker='o', label='Bacteria row (mean)')
plt.plot(timepoints, row_variances_median[0], marker='s', linestyle='--', label='Dye row (median)')
plt.plot(timepoints, row_variances_median[1], marker='s', linestyle='--', label='Bacteria row (median)')
plt.xlabel('Time (min)')
plt.ylabel('Variance (mean/median RGB)')
plt.title('Variance across 10 identical samples per row (SAM detected)')
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'control_variance_over_time_sam.png'))
plt.close()

print('Variance plot (SAM) saved to', os.path.join(RESULTS_DIR, 'control_variance_over_time_sam.png'))
