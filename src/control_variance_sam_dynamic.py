import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
from glob import glob
import torch
from transformers import SamModel, SamProcessor
from PIL import Image

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTROL_DIR = os.path.join(BASE_DIR, 'samples', 'well_plate')
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'statistical', 'well_plate')
os.makedirs(RESULTS_DIR, exist_ok=True)
DETECTED_CIRCLES_DIR = os.path.join(RESULTS_DIR, 'detected_circles')
os.makedirs(DETECTED_CIRCLES_DIR, exist_ok=True)

# Find image files
image_files = sorted(glob(os.path.join(CONTROL_DIR, '*.jpeg')))
if not image_files:
    image_files = sorted(glob(os.path.join(CONTROL_DIR, '*.png')))


def detect_disk_centers_hough(image_rgb):
    """Rough circle detection using Hough to get seed points for SAM."""
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gray_blur = cv2.medianBlur(gray, 7)
    circles = cv2.HoughCircles(
        gray_blur,
        cv2.HOUGH_GRADIENT,
        dp=1.3,
        minDist=42,
        param1=90,
        param2=42,
        minRadius=15,
        maxRadius=40,
    )
    centers = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for x, y, r in circles:
            centers.append((int(x), int(y)))
    return centers


def label_points_row_col(points):
    """Group detected points into rows and columns."""
    if not points:
        return []

    # Group by y-coordinate into rows
    points_sorted_y = sorted(points, key=lambda p: p[1])
    if len(points) == 1:
        row_groups = [points_sorted_y]
    else:
        # Use median spacing to determine row threshold
        y_diffs = [abs(points_sorted_y[i + 1][1] - points_sorted_y[i][1])
                   for i in range(len(points_sorted_y) - 1)]
        if y_diffs:
            threshold = np.median(y_diffs) * 0.7
        else:
            threshold = 20

        row_groups = [[points_sorted_y[0]]]
        for point in points_sorted_y[1:]:
            if abs(point[1] - row_groups[-1][-1][1]) > threshold:
                row_groups.append([point])
            else:
                row_groups[-1].append(point)

    # Sort row groups by y
    row_groups = sorted(row_groups, key=lambda grp: np.mean([p[1] for p in grp]))

    labeled = []
    for row_idx, row_group in enumerate(row_groups):
        row_group_sorted_x = sorted(row_group, key=lambda p: p[0])
        for col_idx, (x, y) in enumerate(row_group_sorted_x):
            labeled.append({
                'x': x,
                'y': y,
                'row': row_idx,
                'col': col_idx,
            })
    return labeled


def segment_disks_sam(image_rgb, points):
    """Segment disks using SAM given seed points."""
    device = "cuda" if torch.cuda.is_available() else "cpu"
    processor = SamProcessor.from_pretrained("facebook/sam-vit-base")
    model = SamModel.from_pretrained("facebook/sam-vit-base").to(device)

    pil_img = Image.fromarray(image_rgb)
    masks = []

    for i, point_dict in enumerate(points):
        x, y = point_dict['x'], point_dict['y']
        inputs = processor(
            pil_img,
            input_points=[[[x, y]]],
            return_tensors="pt"
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)

        # Get best mask based on IoU scores
        mask = processor.image_processor.post_process_masks(
            outputs.pred_masks.cpu(),
            inputs["original_sizes"].cpu(),
            inputs["reshaped_input_sizes"].cpu()
        )[0][0][torch.argmax(outputs.iou_scores.cpu().squeeze()).item()].numpy().astype(bool)

        point_dict['mask'] = mask
        masks.append(mask)
        print(f"Segmented disk {i + 1}/{len(points)} at ({x}, {y})")

    return points, np.array(masks)


all_data = []

for img_path in image_files:
    img = cv2.imread(img_path)
    if img is None:
        print(f"Missing: {img_path}")
        continue

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    detected_overlay = img_rgb.copy()

    # Get seed points from Hough
    seed_points = detect_disk_centers_hough(img_rgb)

    if not seed_points:
        print(f"Warning: no seed points detected for {os.path.basename(img_path)}")
        continue

    # Label points by row/col
    labeled_points = label_points_row_col(seed_points)

    # Segment with SAM
    labeled_points, masks_array = segment_disks_sam(img_rgb, labeled_points)

    print(f"Processing {os.path.basename(img_path)}: detected {len(labeled_points)} disks")

    # Draw overlays and extract color statistics
    color_map = {0: (255, 0, 0), 1: (203, 192, 255)}  # Red for row 0, pink for row 1

    for point_dict, mask in zip(labeled_points, masks_array):
        x, y = point_dict['x'], point_dict['y']
        row, col = point_dict['row'], point_dict['col']

        # Extract pixels in mask
        pixels = img_rgb[mask]
        if len(pixels) == 0:
            continue

        # Draw mask contours
        contours, _ = cv2.findContours(mask.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        circle_color = color_map.get(row, (128, 128, 128))
        cv2.drawContours(detected_overlay, contours, -1, circle_color, 2)

        # Label with (row, col)
        label = f"({row},{col})"
        cv2.putText(
            detected_overlay,
            label,
            (max(5, x - 20), max(15, y - 10)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

        # Extract color statistics
        r_med, g_med, b_med = np.median(pixels, axis=0)
        lab_med = cv2.cvtColor(
            np.array([[[r_med, g_med, b_med]]], dtype=np.uint8),
            cv2.COLOR_RGB2LAB,
        )[0, 0]
        a_star_med = float(lab_med[1])

        all_data.append({
            'Time': int(os.path.basename(img_path).split('th')[0]),
            'Row': row,
            'Col': col,
            'R_median': r_med,
            'G_median': g_med,
            'B_median': b_med,
            'A_star_median': a_star_med,
        })

    # Add detection count label
    cv2.putText(
        detected_overlay,
        f"Detected: {len(labeled_points)}",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )

    # Save overlay image
    base_name = os.path.splitext(os.path.basename(img_path))[0]
    overlay_path = os.path.join(DETECTED_CIRCLES_DIR, f"{base_name}_sam_overlay.png")
    overlay_bgr = cv2.cvtColor(detected_overlay, cv2.COLOR_RGB2BGR)
    cv2.imwrite(overlay_path, overlay_bgr)
    print(f"Saved overlay: {os.path.basename(overlay_path)}")

    # Save masks
    masks_path = os.path.join(DETECTED_CIRCLES_DIR, f"{base_name}_sam_masks.npy")
    np.save(masks_path, masks_array)
    print(f"Saved masks: {os.path.basename(masks_path)}")

# Save statistics
if all_data:
    control_df = pd.DataFrame(all_data)
    csv_path = os.path.join(RESULTS_DIR, 'control_samples_data_sam.csv')
    control_df.to_csv(csv_path, index=False)
    print(f"\nSaved statistics: {os.path.basename(csv_path)}")

    # Summary plot
    plt.figure(figsize=(10, 6))
    for row in sorted(control_df['Row'].unique()):
        subset = control_df[control_df['Row'] == row]
        plt.scatter(subset['Time'], subset['A_star_median'], s=40, alpha=0.44, label=f'Row {row}')
    plt.xlabel('Time (min)')
    plt.ylabel('Median a*')
    plt.title('Per-disk Median a* by Timepoint (SAM)')
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, 'control_median_a_star_scatter_sam.png'))
    plt.close()
    print(f"Saved plot: control_median_a_star_scatter_sam.png")
else:
    print("No data collected; no circles were detected.")
