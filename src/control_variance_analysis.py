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

# Timepoints and files
TIMEPOINTS = [0, 5, 10, 20, 30]
IMG_FILES = [f"{t}th_min.png" for t in TIMEPOINTS]

# Tray layout: 2 rows x 10 columns
N_ROWS, N_COLS = 2, 10

# Extraction parameters (tuned for these images)
RADIUS = 22  # pixels, adjust if needed
ROW_OFFSETS = [60, 140]  # y-coordinates for row centers (estimate)
COL_OFFSETS = [40 + i*38 for i in range(N_COLS)]  # x-coordinates for col centers (estimate)

all_data = []

for t, fname in zip(TIMEPOINTS, IMG_FILES):
    img_path = os.path.join(CONTROL_DIR, fname)
    img = cv2.imread(img_path)
    if img is None:
        print(f"Missing: {img_path}")
        continue
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    for row in range(N_ROWS):
        y = ROW_OFFSETS[row]
        for col in range(N_COLS):
            x = COL_OFFSETS[col]
            Y, X = np.ogrid[:img.shape[0], :img.shape[1]]
            mask = (X - x)**2 + (Y - y)**2 <= RADIUS**2
            pixels = img_rgb[mask]
            if len(pixels) == 0:
                continue
            r_med, g_med, b_med = np.median(pixels, axis=0)
            all_data.append({
                'Time': t,
                'Row': row,
                'Col': col,
                'R_median': r_med,
                'G_median': g_med,
                'B_median': b_med
            })

# DataFrame
control_df = pd.DataFrame(all_data)
control_df.to_csv(os.path.join(RESULTS_DIR, 'control_samples_data.csv'), index=False)

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
