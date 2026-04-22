import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'statistical', 'well_plate')
os.makedirs(RESULTS_DIR, exist_ok=True)

INPUT_CSV = os.path.join(RESULTS_DIR, 'control_samples_data.csv')
OUTPUT_CSV = os.path.join(RESULTS_DIR, 'wp_samples_data_layout.csv')
OUTPUT_PLOT = os.path.join(RESULTS_DIR, 'wp_median_a_star_scatter_layout.png')
SUMMARY_CSV = os.path.join(RESULTS_DIR, 'wp_reference_stats_layout.csv')

EXPECTED_ROW_COUNTS = {0: 10, 1: 10, 2: 10, 3: 6, 4: 6}
N_ROWS = 5
MAX_COLS = 10


if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(
        f'Input file not found: {INPUT_CSV}. Run control_variance_analysis.py first.'
    )


df = pd.read_csv(INPUT_CSV)
required_cols = {'Time', 'Row', 'Col', 'A_star_median'}
missing_cols = required_cols - set(df.columns)
if missing_cols:
    raise ValueError(f'Missing required columns in input CSV: {sorted(missing_cols)}')

valid_frames = []
summary_rows = []
skipped_times = []
all_time_records = []
contributor_records = []

for time_value, time_df in df.groupby('Time', sort=True):
    row_counts = time_df.groupby('Row').size().to_dict()

    # Require top rows to match 10/10/10. Lower rows may mismatch and still be used.
    if row_counts.get(0, 0) != 10 or row_counts.get(1, 0) != 10 or row_counts.get(2, 0) != 10:
        skipped_times.append((time_value, f'top rows mismatch (need 10/10/10): {row_counts}'))
        continue

    circles = np.full((N_ROWS, MAX_COLS), np.nan, dtype=float)
    layout_ok = True

    for row_index in range(N_ROWS):
        row_df = time_df[time_df['Row'] == row_index].sort_values('Col')
        expected_count = EXPECTED_ROW_COUNTS[row_index]
        if row_index < 3 and len(row_df) != expected_count:
            skipped_times.append((time_value, f'row {row_index} count mismatch: {len(row_df)} != {expected_count}'))
            layout_ok = False
            break

        for _, row in row_df.iterrows():
            col_index = int(row['Col'])
            if 0 <= col_index < MAX_COLS:
                circles[row_index, col_index] = float(row['A_star_median'])

    if not layout_ok:
        continue

    reference_block = circles[3:, 3:]
    reference_values = reference_block[np.isfinite(reference_block)]
    if reference_values.size == 0:
        reference_block = circles[3:, :3]
        reference_values = reference_block[np.isfinite(reference_block)]
        if reference_values.size == 0:
            skipped_times.append((time_value, 'reference block has no circles in circles[3:, :3]'))
            continue

    is_contributor = (row_counts == EXPECTED_ROW_COUNTS and reference_values.size == 6)

    record = {
        'Time': time_value,
        'reference_values': reference_values,
        'reference_mean': float(np.mean(reference_values)),
        'reference_std': float(np.std(reference_values, ddof=0)),
        'time_df': time_df.copy(),
        'is_contributor': is_contributor,
        'row_counts': row_counts,
    }
    all_time_records.append(record)
    if is_contributor:
        contributor_records.append(record)

if not all_time_records:
    raise RuntimeError('No images had usable top rows and reference circles, so nothing was normalized.')
if not contributor_records:
    raise RuntimeError('No images matched full 10/10/10/6/6 layout for global mean/std estimation.')

all_reference_values = np.concatenate([record['reference_values'] for record in contributor_records])
mean_all = float(np.mean(all_reference_values))
std_dev_all = float(np.mean([record['reference_std'] for record in contributor_records]))

for record in all_time_records:
    time_value = record['Time']
    reference_mean = record['reference_mean']
    reference_std = record['reference_std']

    # if reference_std == 0:
    #     scale_factor = 1.0
    # else:
    #     scale_factor = std_dev_all / reference_std
    # offset = mean_all - scale_factor * reference_mean
    scale_factor = 1.0
    offset = 0.0

    normalized_df = record['time_df'].copy()
    normalized_df['A_star_reference_mean'] = reference_mean
    normalized_df['A_star_reference_std'] = reference_std
    normalized_df['A_star_mean_all'] = mean_all
    normalized_df['A_star_std_dev_all'] = std_dev_all
    normalized_df['A_star_scale_factor'] = scale_factor
    normalized_df['A_star_offset'] = offset
    normalized_df['A_star_median_normalized'] = (
        scale_factor * normalized_df['A_star_median'] + offset
    )
    valid_frames.append(normalized_df)

    summary_rows.append({
        'Time': time_value,
        'Reference_mean': reference_mean,
        'Reference_std': reference_std,
        'Reference_count': int(record['reference_values'].size),
        'Contributes_to_global_stats': bool(record['is_contributor']),
        'Mean_all': mean_all,
        'Std_dev_all': std_dev_all,
        'Scale_factor': scale_factor,
        'Offset': offset,
    })

normalized_output = pd.concat(valid_frames, ignore_index=True).sort_values(['Time', 'Row', 'Col'])
normalized_output.to_csv(OUTPUT_CSV, index=False)

summary_df = pd.DataFrame(summary_rows).sort_values('Time')
summary_df.to_csv(SUMMARY_CSV, index=False)

# Build aggregated plotting values per timepoint from normalized circles.
def mean_or_nan(values):
    finite_values = values[np.isfinite(values)]
    if finite_values.size == 0:
        return np.nan
    return float(np.mean(finite_values))


plot_rows = []
for time_value, time_df in normalized_output.groupby('Time', sort=True):
    circles_norm = np.full((N_ROWS, MAX_COLS), np.nan, dtype=float)
    for _, row in time_df.iterrows():
        r = int(row['Row'])
        c = int(row['Col'])
        circles_norm[r, c] = float(row['A_star_median_normalized'])

    for col_index in range(10):
        col_mean = mean_or_nan(circles_norm[:3, col_index])
        plot_rows.append({
            'Time': time_value,
            'Group': f'{128.0*2**(-col_index)} μg/mL',
            'Value': col_mean,
        })

    ref_mean = mean_or_nan(circles_norm[3:, 3:])
    plot_rows.append({
        'Time': time_value,
        'Group': 'Sterility control (media only) (ref)',
        'Value': ref_mean,
    })

    # [edit] BRUTE FORCE, CHANGE LATER
    if np.isnan(ref_mean):
        ref_mean = mean_or_nan(circles_norm[3:, :3])
        plot_rows.append({
            'Time': time_value,
            'Group': 'Sterility control (media only) (ref)',
            'Value': ref_mean,
        })
    else:
        lower_left_mean = mean_or_nan(circles_norm[3:, :3])
        plot_rows.append({
            'Time': time_value,
            'Group': 'Growth control (bact only)',
            'Value': lower_left_mean,
        })

plot_df = pd.DataFrame(plot_rows)
plot_df = plot_df[plot_df['Value'].notna()]

plt.figure(figsize=(10, 6))
top_groups = [f'{128.0*2**(-i)} μg/mL' for i in range(10)]
top_cmap = plt.get_cmap('viridis')

for i, group_name in enumerate(top_groups):
    group_df = plot_df[plot_df['Group'] == group_name]
    if group_df.empty:
        continue
    plt.scatter(
        group_df['Time'],
        group_df['Value'],
        s=40,
        alpha=0.49,
        color=top_cmap(i / (len(top_groups) - 1)),
        label=group_name,
    )

bottom_left_df = plot_df[plot_df['Group'] == 'Growth control (bact only)']
if not bottom_left_df.empty:
    plt.scatter(
        bottom_left_df['Time'],
        bottom_left_df['Value'],
        s=58,
        alpha=0.9,
        color='#D95F02',
        label='Growth control (bact only)',
    )

ref_df = plot_df[plot_df['Group'] == 'Sterility control (media only) (ref)']
if not ref_df.empty:
    plt.scatter(
        ref_df['Time'],
        ref_df['Value'],
        s=58,
        alpha=0.9,
        color="#0000FF",
        label='Sterility control (media only) (ref)',
    )
plt.xlabel('Time (min)')
plt.ylabel('Mean a*')
plt.title('Aggregated Circle Means by Time')
plt.grid(True, alpha=0.3)
plt.legend(loc='best', fontsize='small')
plt.tight_layout()
plt.savefig(OUTPUT_PLOT)
plt.close()

print(f'Saved data: {OUTPUT_CSV}')
print(f'Saved summary: {SUMMARY_CSV}')
print(f'Saved scatter: {OUTPUT_PLOT}')
print(f'mean_all={mean_all:.6f}, std_dev_all={std_dev_all:.6f}')
if skipped_times:
    print('Skipped timepoints:')
    for time_value, reason in skipped_times:
        print(f'  t={time_value}: {reason}')
