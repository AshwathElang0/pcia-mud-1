import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import os

def analyze_pinkness_temporal(timepoints=[0, 5, 10, 15, 20, 25]):
    """
    Extract 'pinkest' pixel values from each ROI across all timepoints.
    Pinkness is measured using the a* channel (red-green axis) in LAB color space.
    Higher a* values = more pink/red.
    """
    all_data = []

    for t in timepoints:
        image_path = f"samples/{t}th_min.jpeg"
        if not os.path.exists(image_path):
            print(f"Warning: Could not find image {image_path}. Skipping.")
            continue

        print(f"Processing timepoint: {t} mins")
        image = cv2.imread(image_path)

        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)

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
                idx = np.argsort(y_props['prominences'])[-3:]
                y_peaks = np.sort(y_peaks[idx])
            if len(x_peaks) > 9:
                idx = np.argsort(x_props['prominences'])[-9:]
                x_peaks = np.sort(x_peaks[idx])

            radius = 32

            for r_idx, y in enumerate(y_peaks):
                for c_idx, x in enumerate(x_peaks):
                    # Only process columns 1-7 (skip first and last column)
                    if 0 < c_idx < 8:
                        mask = np.zeros(gray.shape, dtype=np.uint8)
                        cv2.circle(mask, (x, y), radius, 255, -1)

                        # Extract a* channel pixels (pinkness metric)
                        a_channel_pixels = image_lab[mask == 255, 1]  # a* channel

                        # Also get other color channels for comprehensive analysis
                        rgb_pixels = image_rgb[mask == 255]
                        lab_pixels = image_lab[mask == 255]

                        if len(a_channel_pixels) > 0:
                            # Calculate pinkness statistics
                            max_pinkness = np.max(a_channel_pixels)
                            p90_pinkness = np.percentile(a_channel_pixels, 90)
                            p80_pinkness = np.percentile(a_channel_pixels, 80)
                            median_pinkness = np.median(a_channel_pixels)
                            mean_pinkness = np.mean(a_channel_pixels)

                            # Additional metrics
                            r_channel_pixels = rgb_pixels[:, 0]
                            max_red = np.max(r_channel_pixels)
                            p90_red = np.percentile(r_channel_pixels, 90)

                            all_data.append({
                                'Time': t,
                                'Row': r_idx,
                                'Column': c_idx,
                                'A_max': max_pinkness,
                                'A_p90': p90_pinkness,
                                'A_p80': p80_pinkness,
                                'A_median': median_pinkness,
                                'A_mean': mean_pinkness,
                                'R_max': max_red,
                                'R_p90': p90_red,
                                'L_median': np.median(lab_pixels[:, 0])
                            })
        else:
            print(f"  Warning: Grid tracking failed for {image_path}")

    df = pd.DataFrame(all_data)
    if df.empty:
        print("No data extracted.")
        return None

    # Save raw data
    df.to_csv('pinkness_temporal_data.csv', index=False)
    print(f"\nSaved raw data to: pinkness_temporal_data.csv")

    return df


def plot_pinkness_trends(df):
    """
    Create comprehensive plots of pinkness trends across time.
    """
    # Average across replicate rows for each column and timepoint
    grouped_df = df.groupby(['Time', 'Column']).mean().reset_index()

    columns = sorted(grouped_df['Column'].unique())
    times = sorted(grouped_df['Time'].unique())

    # Create figure with multiple subplots
    fig = plt.figure(figsize=(18, 12))

    # Define color scheme for columns (concentration gradient)
    colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(columns)))

    # Plot 1: Max pinkness (a* channel) over time
    ax1 = plt.subplot(2, 3, 1)
    for idx, col in enumerate(columns):
        subset = grouped_df[grouped_df['Column'] == col]
        ax1.plot(subset['Time'], subset['A_max'], marker='o', color=colors[idx],
                linewidth=2, markersize=8, label=f'Col {col}')
    ax1.set_xlabel('Time (minutes)', fontsize=11)
    ax1.set_ylabel('Max a* (pinkness)', fontsize=11)
    ax1.set_title('Maximum Pinkness per Well', fontsize=12, fontweight='bold')
    ax1.legend(loc='best', fontsize=9)
    ax1.grid(True, alpha=0.3)

    # Plot 2: 90th percentile pinkness
    ax2 = plt.subplot(2, 3, 2)
    for idx, col in enumerate(columns):
        subset = grouped_df[grouped_df['Column'] == col]
        ax2.plot(subset['Time'], subset['A_p90'], marker='s', color=colors[idx],
                linewidth=2, markersize=8, label=f'Col {col}')
    ax2.set_xlabel('Time (minutes)', fontsize=11)
    ax2.set_ylabel('90th percentile a*', fontsize=11)
    ax2.set_title('90th Percentile Pinkness per Well', fontsize=12, fontweight='bold')
    ax2.legend(loc='best', fontsize=9)
    ax2.grid(True, alpha=0.3)

    # Plot 3: 80th percentile pinkness
    ax3 = plt.subplot(2, 3, 3)
    for idx, col in enumerate(columns):
        subset = grouped_df[grouped_df['Column'] == col]
        ax3.plot(subset['Time'], subset['A_p80'], marker='^', color=colors[idx],
                linewidth=2, markersize=8, label=f'Col {col}')
    ax3.set_xlabel('Time (minutes)', fontsize=11)
    ax3.set_ylabel('80th percentile a*', fontsize=11)
    ax3.set_title('80th Percentile Pinkness per Well', fontsize=12, fontweight='bold')
    ax3.legend(loc='best', fontsize=9)
    ax3.grid(True, alpha=0.3)

    # Plot 4: Comparison of metrics for a representative column (Column 6 - had growth)
    ax4 = plt.subplot(2, 3, 4)
    target_col = 6 if 6 in columns else columns[len(columns)//2]
    subset = grouped_df[grouped_df['Column'] == target_col]
    ax4.plot(subset['Time'], subset['A_max'], marker='o', linewidth=2,
            markersize=8, label='Max', color='darkred')
    ax4.plot(subset['Time'], subset['A_p90'], marker='s', linewidth=2,
            markersize=8, label='90th %ile', color='red')
    ax4.plot(subset['Time'], subset['A_p80'], marker='^', linewidth=2,
            markersize=8, label='80th %ile', color='orange')
    ax4.plot(subset['Time'], subset['A_median'], marker='d', linewidth=2,
            markersize=8, label='Median', color='gray')
    ax4.set_xlabel('Time (minutes)', fontsize=11)
    ax4.set_ylabel('a* value', fontsize=11)
    ax4.set_title(f'Column {target_col}: Pinkness Metric Comparison', fontsize=12, fontweight='bold')
    ax4.legend(loc='best', fontsize=9)
    ax4.grid(True, alpha=0.3)

    # Plot 5: Change from baseline (t=0) for max pinkness
    ax5 = plt.subplot(2, 3, 5)
    for idx, col in enumerate(columns):
        subset = grouped_df[grouped_df['Column'] == col].sort_values('Time')
        baseline = subset[subset['Time'] == subset['Time'].min()]['A_max'].values[0]
        changes = subset['A_max'] - baseline
        ax5.plot(subset['Time'], changes, marker='o', color=colors[idx],
                linewidth=2, markersize=8, label=f'Col {col}')
    ax5.axhline(y=0, color='black', linestyle='--', alpha=0.3)
    ax5.set_xlabel('Time (minutes)', fontsize=11)
    ax5.set_ylabel('Δ Max a* from baseline', fontsize=11)
    ax5.set_title('Change in Maximum Pinkness', fontsize=12, fontweight='bold')
    ax5.legend(loc='best', fontsize=9)
    ax5.grid(True, alpha=0.3)

    # Plot 6: Heatmap-style visualization (90th percentile)
    ax6 = plt.subplot(2, 3, 6)
    pivot_data = grouped_df.pivot(index='Time', columns='Column', values='A_p90')
    im = ax6.imshow(pivot_data.values, cmap='RdPu', aspect='auto', interpolation='nearest')
    ax6.set_xticks(range(len(pivot_data.columns)))
    ax6.set_xticklabels(pivot_data.columns)
    ax6.set_yticks(range(len(pivot_data.index)))
    ax6.set_yticklabels(pivot_data.index)
    ax6.set_xlabel('Column', fontsize=11)
    ax6.set_ylabel('Time (minutes)', fontsize=11)
    ax6.set_title('90th Percentile Pinkness Heatmap', fontsize=12, fontweight='bold')
    plt.colorbar(im, ax=ax6, label='a* value')

    # Add annotations to heatmap
    for i in range(len(pivot_data.index)):
        for j in range(len(pivot_data.columns)):
            text = ax6.text(j, i, f"{pivot_data.values[i, j]:.0f}",
                          ha="center", va="center", color="black", fontsize=8)

    plt.suptitle('Temporal Pinkness Analysis: Max and Percentile Metrics',
                fontsize=16, fontweight='bold', y=0.995)
    plt.tight_layout()
    plt.savefig('pinkness_trends.png', dpi=150, bbox_inches='tight')
    print("Saved plot to: pinkness_trends.png")

    # Print summary statistics
    print("\n" + "="*70)
    print("PINKNESS TREND SUMMARY")
    print("="*70)

    for col in columns:
        subset = grouped_df[grouped_df['Column'] == col].sort_values('Time')
        t0_max = subset[subset['Time'] == subset['Time'].min()]['A_max'].values[0]
        t_final_max = subset[subset['Time'] == subset['Time'].max()]['A_max'].values[0]
        delta_max = t_final_max - t0_max

        t0_p90 = subset[subset['Time'] == subset['Time'].min()]['A_p90'].values[0]
        t_final_p90 = subset[subset['Time'] == subset['Time'].max()]['A_p90'].values[0]
        delta_p90 = t_final_p90 - t0_p90

        print(f"\nColumn {col}:")
        print(f"  Max a*:      {t0_max:.1f} → {t_final_max:.1f} (Δ = {delta_max:+.1f})")
        print(f"  90th p'tile: {t0_p90:.1f} → {t_final_p90:.1f} (Δ = {delta_p90:+.1f})")


def create_individual_column_plots(df):
    """
    Create detailed plots for each column showing all percentile metrics.
    """
    grouped_df = df.groupby(['Time', 'Column']).mean().reset_index()
    columns = sorted(grouped_df['Column'].unique())

    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    axes = axes.flatten()

    for idx, col in enumerate(columns):
        if idx >= len(axes):
            break

        ax = axes[idx]
        subset = grouped_df[grouped_df['Column'] == col]

        ax.plot(subset['Time'], subset['A_max'], marker='o', linewidth=2,
               markersize=6, label='Max', color='darkred')
        ax.plot(subset['Time'], subset['A_p90'], marker='s', linewidth=2,
               markersize=6, label='90th', color='red')
        ax.plot(subset['Time'], subset['A_p80'], marker='^', linewidth=2,
               markersize=6, label='80th', color='orange')
        ax.plot(subset['Time'], subset['A_median'], marker='d', linewidth=2,
               markersize=6, label='Median', color='gray', alpha=0.7)

        ax.set_xlabel('Time (min)', fontsize=10)
        ax.set_ylabel('a* value', fontsize=10)
        ax.set_title(f'Column {col}', fontsize=11, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)

    # Hide unused subplots
    for idx in range(len(columns), len(axes)):
        axes[idx].axis('off')

    plt.suptitle('Individual Column Pinkness Trajectories', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('pinkness_by_column.png', dpi=150, bbox_inches='tight')
    print("Saved plot to: pinkness_by_column.png")


if __name__ == "__main__":
    print("="*70)
    print("PINKNESS PERCENTILE ANALYSIS")
    print("="*70)
    print("\nExtracting max, 90th, and 80th percentile 'pinkest' pixels from each ROI...")
    print("Pinkness measured using a* channel in LAB color space.\n")

    # Extract data
    df = analyze_pinkness_temporal()

    if df is not None:
        # Create plots
        print("\nGenerating plots...")
        plot_pinkness_trends(df)
        create_individual_column_plots(df)

        print("\n" + "="*70)
        print("ANALYSIS COMPLETE")
        print("="*70)
        print("\nGenerated files:")
        print("  - pinkness_temporal_data.csv (raw data)")
        print("  - pinkness_trends.png (main analysis)")
        print("  - pinkness_by_column.png (detailed per-column plots)")
