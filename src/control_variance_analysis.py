import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTROL_DIR = os.path.join(BASE_DIR, 'samples', 'well_plate')
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'statistical', 'well_plate')
os.makedirs(RESULTS_DIR, exist_ok=True)
DETECTED_CIRCLES_DIR = os.path.join(RESULTS_DIR, 'detected_circles')
os.makedirs(DETECTED_CIRCLES_DIR, exist_ok=True)

# Timepoints and files
TIMEPOINTS = [0, 10, 20, 30, 40, 50, 60, 70]
IMG_FILES = [f"{t}th_min.jpeg" for t in TIMEPOINTS]

# Detection parameters (tuned for these images)
MIN_RADIUS = 15
MAX_RADIUS = 40


def detect_disks(image_rgb):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gray_blur = cv2.medianBlur(gray, 7)

    lab = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)
    l_chan = lab[:, :, 0]
    a_chan = lab[:, :, 1]

    # Blend luminance with red-green opponency to boost very light pink wells.
    l_norm = cv2.normalize(l_chan, None, 0, 255, cv2.NORM_MINMAX)
    a_norm = cv2.normalize(a_chan, None, 0, 255, cv2.NORM_MINMAX)
    pink_emphasis = cv2.addWeighted(l_norm, 0.05, a_norm, 0.80, 0)
    pink_emphasis_blur = cv2.GaussianBlur(pink_emphasis, (7, 7), 0)

    circles_pink = cv2.HoughCircles(
        pink_emphasis_blur,
        cv2.HOUGH_GRADIENT,
        dp=1.3,
        minDist=42,
        param1=90,
        param2=35,
        minRadius=MIN_RADIUS,
        maxRadius=MAX_RADIUS,
    )
    circles_gray = cv2.HoughCircles(
        gray_blur,
        cv2.HOUGH_GRADIENT,
        dp=1.3,
        minDist=42,
        param1=100,
        param2=42,
        minRadius=MIN_RADIUS,
        maxRadius=MAX_RADIUS,
    )

    candidate_circles = []
    for circles in (circles_pink, circles_gray):
        if circles is not None:
            candidate_circles.extend(np.round(circles[0, :]).astype("int"))

    disks = []
    for x, y, r in candidate_circles:
        is_duplicate = False
        for ex, ey, er in disks:
            if (x - ex) ** 2 + (y - ey) ** 2 <= (0.45 * (r + er)) ** 2:
                is_duplicate = True
                break
        if not is_duplicate:
            disks.append((int(x), int(y), int(r)))
    return disks


def label_disks_row_col(disks):
    if not disks:
        return []

    # Group disks into rows by y-gaps, then sort each row by x.
    disks_sorted_y = sorted(disks, key=lambda d: d[1])
    median_r = float(np.median([d[2] for d in disks_sorted_y]))
    row_split_threshold = max(8.0, 0.9 * median_r)

    row_groups = [[disks_sorted_y[0]]]
    for disk in disks_sorted_y[1:]:
        if abs(disk[1] - row_groups[-1][-1][1]) > row_split_threshold:
            row_groups.append([disk])
        else:
            row_groups[-1].append(disk)

    row_groups = sorted(row_groups, key=lambda grp: np.mean([d[1] for d in grp]))

    labeled = []
    for row_idx, row_group in enumerate(row_groups):
        row_group_sorted_x = sorted(row_group, key=lambda d: d[0])
        for col_idx, (x, y, r) in enumerate(row_group_sorted_x):
            labeled.append({
                'x': x,
                'y': y,
                'r': r,
                'row': row_idx,
                'col': col_idx,
            })
    return labeled

all_data = []

for t, fname in zip(TIMEPOINTS, IMG_FILES):
    img_path = os.path.join(CONTROL_DIR, fname)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Missing: {img_path}")
        continue
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    detected_overlay = img.copy()

    detected_disks = detect_disks(img_rgb)
    labeled_disks = label_disks_row_col(detected_disks)

    if not labeled_disks:
        print(f"Warning: no circles detected for t={t}")

    for disk in labeled_disks:
        x, y, r = disk['x'], disk['y'], disk['r']
        row, col = disk['row'], disk['col']

        Y, X = np.ogrid[:img.shape[0], :img.shape[1]]
        mask = (X - x)**2 + (Y - y)**2 <= r**2
        pixels = img_rgb[mask]
        if len(pixels) == 0:
            continue

        circle_color = (255, 0, 0) if row == 0 else (203, 192, 255)
        cv2.circle(detected_overlay, (x, y), r, circle_color, 2)
        label = f"({row},{col})"
        cv2.putText(
            detected_overlay,
            label,
            (x - r, max(15, y - r - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.45,
            (0, 255, 0),
            1,
            cv2.LINE_AA,
        )

        r_med, g_med, b_med = np.median(pixels, axis=0)
        # Convert median RGB to CIE Lab and keep a* in approximately [-128, 127].
        lab_med = cv2.cvtColor(
            np.array([[[r_med, g_med, b_med]]], dtype=np.uint8),
            cv2.COLOR_RGB2LAB,
        )[0, 0]
        a_star_med = float(lab_med[1])
        all_data.append({
            'Time': t,
            'Row': row,
            'Col': col,
            'Detected_radius': r,
            'R_median': r_med,
            'G_median': g_med,
            'B_median': b_med,
            'A_star_median': a_star_med,
        })

    cv2.putText(
        detected_overlay,
        f"Detected circles: {len(labeled_disks)}",
        (10, 24),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.65,
        (0, 255, 255),
        2,
        cv2.LINE_AA,
    )
    circles_out_path = os.path.join(DETECTED_CIRCLES_DIR, f"detected_circles_{t}min.png")
    cv2.imwrite(circles_out_path, detected_overlay)
    print(f"Saved: {os.path.basename(circles_out_path)}")

# DataFrame
control_df = pd.DataFrame(all_data)
control_df.to_csv(os.path.join(RESULTS_DIR, 'control_samples_data.csv'), index=False)

if control_df.empty:
    raise RuntimeError('No circles were detected in any image; no statistics to save.')

# Scatter plot of per-sample median a* values over time.
plt.figure(figsize=(10, 6))
for row in sorted(control_df['Row'].unique()):
    subset = control_df[control_df['Row'] == row]
    plt.scatter(subset['Time'], subset['A_star_median'], s=40, alpha=0.44, label=f'Row {row}')
plt.xlabel('Timesteps (min)')
plt.ylabel('Median a*')
plt.title('Per-sample Median a* by Timestep')
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'control_median_a_star_scatter.png'))
plt.close()
print('Saved: control_median_a_star_scatter.png')

# Compute variance across columns for each row and timepoint
var_df = (
    control_df
    .groupby(['Time', 'Row'])[['R_median', 'G_median', 'B_median']]
    .var()
    .reset_index()
    .melt(id_vars=['Time', 'Row'], var_name='Channel', value_name='Variance')
)

# Plot variance trends
groups = var_df.groupby(['Row', 'Channel'])
plt.figure(figsize=(10,6))
for (row, channel), group in groups:
    plt.plot(group['Time'], group['Variance'], marker='o', label=f"Row {row} - {channel}")
plt.xlabel('Time (min)')
plt.ylabel('Variance across 10 samples')
plt.title('Variance of Control Samples Over Time')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'control_variance_trends.png'))
print('Saved: control_variance_trends.png')
