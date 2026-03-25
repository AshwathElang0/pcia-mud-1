import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from glob import glob
import torch
from transformers import SamModel, SamProcessor
from PIL import Image
import seaborn as sns

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

def detect_disk_centers(image_rgb):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gray_blur = cv2.medianBlur(gray, 7)
    circles = cv2.HoughCircles(gray_blur, cv2.HOUGH_GRADIENT, dp=1.2, minDist=30,
                               param1=50, param2=30, minRadius=15, maxRadius=60)
    centers = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        # Sort by y, then x to get rows and columns
        circles = sorted(circles, key=lambda c: (c[1], c[0]))
        for c in circles:
            centers.append((c[0], c[1]))
    return centers

def detect_samples_sam(image_rgb):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)
    pil_img = Image.fromarray(image_rgb)
    centers = detect_disk_centers(image_rgb)
    # If not enough disks detected, fall back to grid
    if len(centers) != N_ROWS * N_COLS:
        h, w = image_rgb.shape[:2]
        row_ys = np.linspace(h/6, h*5/6, N_ROWS)
        col_xs = np.linspace(w/20, w*19/20, N_COLS)
        centers = [(int(x), int(y)) for y in row_ys for x in col_xs]
    centers = np.array(centers).reshape(N_ROWS, N_COLS, 2)
    masks = []
    for r in range(N_ROWS):
        row_masks = []
        for c in range(N_COLS):
            pt = centers[r, c]
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



# --- Store all medians for distribution analysis ---

# Store all medians for distribution analysis (R and a* channels)
row_medians_over_time = {0: [], 1: []}  # R channel
row_medians_a_over_time = {0: [], 1: []}  # a* channel


for img_path in image_files:
    img = cv2.imread(img_path)
    if img is None:
        continue
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    # --- Mask caching ---
    mask_path = os.path.splitext(img_path)[0] + '_sam_masks.npy'
    if os.path.exists(mask_path):
        masks = np.load(mask_path)
    else:
        masks = detect_samples_sam(img_rgb)
        np.save(mask_path, masks)
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
        row_medians_over_time[r].append(sample_medians[r,:,0].copy())  # R channel
        row_medians_a_over_time[r].append(sample_medians[r,:,1].copy())  # a* channel
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


# --- Plot KDEs for a* values of both rows (before normalization) ---
plt.figure(figsize=(12,6))
for r in range(N_ROWS):
    handles = []
    labels = []
    for i, t in enumerate(timepoints):
        color = f'C{i}'
        kde = sns.kdeplot(row_medians_a_over_time[r][i], label=f'Time {t} min', bw_adjust=0.5, alpha=0.5, fill=True, clip=(-128,128), color=color, lw=1)
        if kde.lines:
            handles.append(kde.lines[-1])
            labels.append(f'Time {t} min')
    plt.title(f'Row {r} a* Medians Over Time (Unnormalized)')
    plt.xlabel('a* value')
    plt.ylabel('Density')
    plt.legend(handles=handles, labels=labels, title='Timepoints', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(rect=[0,0,0.85,1])
    plt.savefig(os.path.join(RESULTS_DIR, f'row{r}_a_distribution_over_time.png'))
    plt.clf()


# --- Plot distribution evolution for each row ---
import seaborn as sns

row_medians_over_time[0] = np.array(row_medians_over_time[0])  # shape: (n_time, 10)
row_medians_over_time[1] = np.array(row_medians_over_time[1])
row_medians_a_over_time[0] = np.array(row_medians_a_over_time[0])
row_medians_a_over_time[1] = np.array(row_medians_a_over_time[1])

# --- Normalization for lighting correction using dye row (a* channel) --- #
dye_a_medians = np.nanmedian(np.array(row_medians_a_over_time[0]), axis=1)  # shape: (n_time,)
global_center = np.nanmean(dye_a_medians)
shifts = global_center - dye_a_medians  # shape: (n_time,)
norm_dye_a = []
norm_bacteria_a = []
for i, shift in enumerate(shifts):
    norm_dye_a.append(np.array(row_medians_a_over_time[0][i]) + shift)
    norm_bacteria_a.append(np.array(row_medians_a_over_time[1][i]) + shift)
norm_dye_a = np.array(norm_dye_a)
norm_bacteria_a = np.array(norm_bacteria_a)

# --- Plot normalized KDEs for both rows ---
plt.figure(figsize=(12,6))
for r, norm_data in zip([0,1], [norm_dye_a, norm_bacteria_a]):
    handles = []
    labels = []
    for i, t in enumerate(timepoints):
        color = f'C{i}'
        kde = sns.kdeplot(norm_data[i], label=f'Time {t} min', bw_adjust=0.5, alpha=0.5, fill=True, clip=(-128,128), color=color, lw=1)
        if kde.lines:
            handles.append(kde.lines[-1])
            labels.append(f'Time {t} min')
    plt.title(f'Row {r} a* Medians Over Time (Normalized)')
    plt.xlabel('Normalized a* value')
    plt.ylabel('Density')
    plt.legend(handles=handles, labels=labels, title='Timepoints', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(rect=[0,0,0.85,1])
    plt.savefig(os.path.join(RESULTS_DIR, f'row{r}_a_distribution_over_time_normalized.png'))
    plt.clf()


# --- Plot both rows' a* medians (unnormalized) on the same scatter plot ---
plt.figure(figsize=(10,6))
for r, label, marker in zip([0,1], ['Dye row', 'Bacteria row'], ['o', 's']):
    for i, t in enumerate(timepoints):
        plt.scatter([t]*len(row_medians_a_over_time[r][i]), row_medians_a_over_time[r][i], color=f'C{i}', marker=marker, label=f'{label} - {t} min' if i==0 else None, alpha=0.7)
plt.title('Unnormalized a* Medians (Both Rows)')
plt.xlabel('Time (min)')
plt.ylabel('a* value')
plt.legend(loc='best', fontsize='small')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'both_rows_a_scatter.png'))
plt.close()

# --- Plot both rows' a* medians (normalized) on the same scatter plot ---
plt.figure(figsize=(10,6))
for r, label, marker in zip([0,1], ['Dye row', 'Bacteria row'], ['o', 's']):
    norm_data = norm_dye_a if r == 0 else norm_bacteria_a
    for i, t in enumerate(timepoints):
        plt.scatter([t]*len(norm_data[i]), norm_data[i], color=f'C{i}', marker=marker, label=f'{label} - {t} min' if i==0 else None, alpha=0.7)
plt.title('Normalized a* Medians (Both Rows)')
plt.xlabel('Time (min)')
plt.ylabel('Normalized a* value')
plt.legend(loc='best', fontsize='small')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'both_rows_a_scatter_normalized.png'))
plt.close()

# --- Plot just the unnormalized a* values of the bacteria row (all timepoints as scatter) ---
plt.figure(figsize=(10,6))
for i, t in enumerate(timepoints):
    plt.scatter([t]*len(row_medians_a_over_time[1][i]), row_medians_a_over_time[1][i], color=f'C{i}', label=f'Time {t} min' if i==0 else None, alpha=0.7)
plt.title('Unnormalized Bacteria Row a* Medians (Scatter)')
plt.xlabel('Time (min)')
plt.ylabel('a* value')
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'bacteria_row_a_scatter.png'))
plt.close()

plt.figure(figsize=(12,6))

for r in range(N_ROWS):
    data = row_medians_over_time[r]
    handles = []
    labels = []
    for i, t in enumerate(timepoints):
        label = f'Time {t} min'
        color = f'C{i}'  # Use matplotlib color cycle for distinction
        kde = sns.kdeplot(data[i], label=label, bw_adjust=0.5, alpha=0.5, fill=True, clip=(0,255),
                         color=color, lw=1)
        # Only add to legend if a line was drawn
        if kde.lines:
            handles.append(kde.lines[-1])
            labels.append(label)
    plt.title(f'Distribution of Disk R-channel Medians (Row {r}, Dye=0, Bacteria=1)')
    plt.xlabel('Median R value')
    plt.ylabel('Density')
    plt.legend(handles=handles, labels=labels, title='Timepoints', bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.tight_layout(rect=[0,0,0.85,1])
    plt.savefig(os.path.join(RESULTS_DIR, f'row{r}_medianR_distribution_over_time.png'))
    plt.clf()

print('Variance plot (SAM) saved to', os.path.join(RESULTS_DIR, 'control_variance_over_time_sam.png'))
print('Distribution plots saved to', os.path.join(RESULTS_DIR, 'row0_medianR_distribution_over_time.png'), 'and row1_medianR_distribution_over_time.png')
