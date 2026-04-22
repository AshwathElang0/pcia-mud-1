import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os

# Paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CONTROL_DIR = os.path.join(BASE_DIR, 'samples', 'control')
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'statistical')
os.makedirs(RESULTS_DIR, exist_ok=True)
DETECTED_CIRCLES_DIR = os.path.join(RESULTS_DIR, 'detected_circles')
os.makedirs(DETECTED_CIRCLES_DIR, exist_ok=True)

# Timepoints and files
TIMEPOINTS = [0, 5, 10, 20, 30]
IMG_FILES = [f"{t}th_min.png" for t in TIMEPOINTS]

# Tray layout: 2 rows x 10 columns
N_ROWS, N_COLS = 2, 10

# Extraction parameters (tuned for these images)
RADIUS = 22  # pixels, adjust if needed
ROW_OFFSETS = [60, 140]  # y-coordinates for row centers (estimate)
COL_OFFSETS = [40 + i*38 for i in range(N_COLS)]  # x-coordinates for col centers (estimate)


def detect_disk_centers(image_rgb):
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    gray_blur = cv2.medianBlur(gray, 7)
    circles = cv2.HoughCircles(
        gray_blur,
        cv2.HOUGH_GRADIENT,
        dp=1.2,
        minDist=30,
        param1=50,
        param2=30,
        minRadius=15,
        maxRadius=60,
    )
    centers = []
    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        # Sort by y, then x to map circles into row-major order.
        circles = sorted(circles, key=lambda c: (c[1], c[0]))
        for c in circles:
            centers.append((c[0], c[1]))
    return centers

all_data = []

for t, fname in zip(TIMEPOINTS, IMG_FILES):
    img_path = os.path.join(CONTROL_DIR, fname)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Missing: {img_path}")
        continue
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    detected_overlay = img.copy()

    detected_centers = detect_disk_centers(img_rgb)
    if len(detected_centers) == N_ROWS * N_COLS:
        centers = np.array(detected_centers).reshape(N_ROWS, N_COLS, 2)
    else:
        # Fallback to fixed grid if circle detection does not find all disks.
        centers = np.array(
            [[(COL_OFFSETS[col], ROW_OFFSETS[row]) for col in range(N_COLS)] for row in range(N_ROWS)]
        )
        print(f"Warning: detected {len(detected_centers)} circles for t={t}; using fallback grid")

    for row in range(N_ROWS):
        for col in range(N_COLS):
            x, y = centers[row, col]
            Y, X = np.ogrid[:img.shape[0], :img.shape[1]]
            mask = (X - x)**2 + (Y - y)**2 <= RADIUS**2
            pixels = img_rgb[mask]
            if len(pixels) == 0:
                continue
            circle_color = (255, 0, 0) if row == 0 else (203, 192, 255)
            cv2.circle(detected_overlay, (x, y), RADIUS, circle_color, 2)
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
                'R_median': r_med,
                'G_median': g_med,
                'B_median': b_med,
                'A_star_median': a_star_med,
            })
    circles_out_path = os.path.join(DETECTED_CIRCLES_DIR, f"detected_circles_{t}min.png")
    cv2.imwrite(circles_out_path, detected_overlay)
    print(f"Saved: {os.path.basename(circles_out_path)}")

# DataFrame
control_df = pd.DataFrame(all_data)
control_df.to_csv(os.path.join(RESULTS_DIR, 'control_samples_data.csv'), index=False)

# Scatter plot of per-sample median a* values over time (10 per row at each timepoint).
plt.figure(figsize=(10, 6))
row0 = control_df[control_df['Row'] == 0]
row1 = control_df[control_df['Row'] == 1]
plt.scatter(row0['Time'], row0['A_star_median'], color='blue', s=40, alpha=0.44, label='No bact (ref)')
plt.scatter(row1['Time'], row1['A_star_median'], color='pink', s=40, alpha=0.44, label='Bact')
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
var_data = []
for t in TIMEPOINTS:
    for row in range(N_ROWS):
        subset = control_df[(control_df['Time'] == t) & (control_df['Row'] == row)]
        for channel in ['R_median', 'G_median', 'B_median']:
            var = subset[channel].var()
            var_data.append({'Time': t, 'Row': row, 'Channel': channel, 'Variance': var})
var_df = pd.DataFrame(var_data)

# Plot variance trends
groups = var_df.groupby(['Row', 'Channel'])
plt.figure(figsize=(10,6))
for (row, channel), group in groups:
    plt.plot(group['Time'], group['Variance'], marker='o', label=f"Row {row} ({'dye' if row==0 else 'bacteria'}) - {channel}")
plt.xlabel('Time (min)')
plt.ylabel('Variance across 10 samples')
plt.title('Variance of Control Samples Over Time')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'control_variance_trends.png'))
print('Saved: control_variance_trends.png')
