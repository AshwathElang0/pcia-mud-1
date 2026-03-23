"""
Smart Replicate Selection (Refined Trends)
==========================================
This script calculates the trajectory distance between the three replicates 
(Row 0, 1, 2) in each column. It identifies and discards "heterogeneous" 
outliers to produce more stable and representative trends.
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, 'results', 'temporal', 'temporal_data.csv')
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'statistical')

def refined_trend_analysis():
    if not os.path.exists(CSV_PATH):
        print(f"Error: {CSV_PATH} not found. Run temporal_analysis.py first.")
        return

    df = pd.read_csv(CSV_PATH)
    columns = sorted(df['Column'].unique())
    timepoints = sorted(df['Time'].unique())
    
    refined_data = []
    dropped_count = 0

    print("--- Starting Smart Replicate Selection ---")
    
    for col in columns:
        col_df = df[df['Column'] == col]
        rows = sorted(col_df['Row'].unique())
        
        # Calculate full a* trajectories for each row
        trajectories = {}
        for r in rows:
            traj = col_df[col_df['Row'] == r].sort_values('Time')['A_median'].values
            if len(traj) == len(timepoints):
                trajectories[r] = traj
        
        valid_rows = list(trajectories.keys())
        if len(valid_rows) < 2:
            print(f"Column {col}: Insufficient data for comparison.")
            refined_data.append(col_df)
            continue

        # Calculate pairwise Euclidean distances between trajectories
        distances = {}
        for r1 in valid_rows:
            dist_sum = 0
            for r2 in valid_rows:
                if r1 != r2:
                    dist_sum += np.linalg.norm(trajectories[r1] - trajectories[r2])
            distances[r1] = dist_sum / (len(valid_rows) - 1)

        # Outlier Detection:
        # If one row's average distance to others is > 2x the minimum distance, flag it.
        min_dist = min(distances.values())
        selected_rows = []
        
        if len(valid_rows) == 3:
            # Check if one is a clear outlier
            for r, d in distances.items():
                if d > 1.8 * min_dist and d > 5.0: # threshold of 5.0 in a* space
                    print(f"Column {col}: Row {r} identified as outlier (dist {d:.2f} vs min {min_dist:.2f}). Dropping.")
                    dropped_count += 1
                else:
                    selected_rows.append(r)
        else:
            selected_rows = valid_rows

        # If we dropped all or didn't find specific ones, use all (fallback)
        if not selected_rows:
            selected_rows = valid_rows

        refined_col_df = col_df[col_df['Row'].isin(selected_rows)]
        refined_data.append(refined_col_df)

    final_df = pd.concat(refined_data)
    
    # Generate Plots
    grouped_orig = df.groupby(['Time', 'Column']).mean().reset_index()
    grouped_refined = final_df.groupby(['Time', 'Column']).mean().reset_index()

    fig, axes = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    
    colors = plt.cm.viridis(np.linspace(0, 1, len(columns)))

    # Original Plot
    for col, color in zip(columns, colors):
        d = grouped_orig[grouped_orig['Column'] == col]
        axes[0].plot(d['Time'], d['A_median'], marker='o', color=color, alpha=0.5, linestyle='--', label=f'Col {col}')
    axes[0].set_title('Original Trends (All Rows Averaged)')
    axes[0].set_xlabel('Time (min)')
    axes[0].set_ylabel('a* Median Value')
    axes[0].grid(True, alpha=0.3)

    # Refined Plot
    for col, color in zip(columns, colors):
        d = grouped_refined[grouped_refined['Column'] == col]
        axes[1].plot(d['Time'], d['A_median'], marker='o', color=color, linewidth=2, label=f'Col {col}')
    axes[1].set_title(f'Refined Trends (Consensus Selection, {dropped_count} rows dropped)')
    axes[1].set_xlabel('Time (min)')
    axes[1].grid(True, alpha=0.3)
    
    axes[1].legend(title='Column', bbox_to_anchor=(1.05, 1), loc='upper left')
    
    plt.tight_layout()
    plot_path = os.path.join(RESULTS_DIR, 'refined_trends_comparison.png')
    plt.savefig(plot_path, dpi=150)
    print(f"--- Refinement Complete ---")
    print(f"Dropped {dropped_count} outlier trajectories.")
    print(f"Saved comparison plot to {plot_path}")

    # Save refined data
    final_df.to_csv(os.path.join(RESULTS_DIR, 'refined_temporal_data.csv'), index=False)

if __name__ == "__main__":
    refined_trend_analysis()
