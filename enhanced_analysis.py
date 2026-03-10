import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt

"""
Enhanced analysis to extract more insight from subtle colorimetric changes.
"""

def enhanced_temporal_analysis(csv_path='temporal_data.csv'):
    df = pd.read_csv(csv_path)
    
    # Average across rows
    col_avg = df.groupby(['Time', 'Column']).mean().reset_index()
    
    print("=" * 60)
    print("ENHANCED TEMPORAL ANALYSIS")
    print("=" * 60)
    
    # 1. Statistical Significance Testing
    print("\n1. STATISTICAL SIGNIFICANCE (t-test: t0 vs t25)")
    print("-" * 60)
    
    columns = sorted(col_avg['Column'].unique())
    for col in columns:
        col_data = col_avg[col_avg['Column'] == col]
        
        t0_data = df[(df['Time'] == 0) & (df['Column'] == col)]['A_median']
        t25_data = df[(df['Time'] == 25) & (df['Column'] == col)]['A_median']
        
        if len(t0_data) > 1 and len(t25_data) > 1:
            t_stat, p_value = stats.ttest_ind(t0_data, t25_data)
            significant = "✓ SIG" if p_value < 0.05 else "✗ NS"
            print(f"Col {col}: Δa* = {t25_data.mean() - t0_data.mean():+.2f}, p={p_value:.4f} {significant}")
    
    # 2. Coefficient of Variation (noise-to-signal)
    print("\n2. COEFFICIENT OF VARIATION (within-well variability)")
    print("-" * 60)
    
    for time in [0, 25]:
        print(f"\nTime = {time} min:")
        for col in columns:
            subset = df[(df['Time'] == time) & (df['Column'] == col)]['A_median']
            cv = (subset.std() / subset.mean()) * 100 if subset.mean() != 0 else 0
            print(f"  Col {col}: CV = {cv:.1f}%")
    
    # 3. Alternative Metrics
    print("\n3. ALTERNATIVE COLOR METRICS")
    print("-" * 60)
    
    # Red/Blue ratio
    df['R_B_ratio'] = df['R_median'] / (df['B_median'] + 1)
    
    # Chromatic components
    df['Chroma'] = np.sqrt(df['A_median']**2 + df['B_lab_median']**2)
    
    # Saturation-normalized hue
    df['Normalized_H'] = df['H_median'] * df['S_median'] / 255.0
    
    col_avg_enhanced = df.groupby(['Time', 'Column']).mean().reset_index()
    
    for col in columns:
        subset = col_avg_enhanced[col_avg_enhanced['Column'] == col]
        delta_ratio = subset.iloc[-1]['R_B_ratio'] - subset.iloc[0]['R_B_ratio']
        delta_chroma = subset.iloc[-1]['Chroma'] - subset.iloc[0]['Chroma']
        print(f"Col {col}: ΔR/B ratio = {delta_ratio:+.3f}, ΔChroma = {delta_chroma:+.2f}")
    
    # 4. Visualization: Heatmap of changes
    print("\n4. Generating enhanced visualizations...")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Heatmap: A* over time
    pivot_A = col_avg.pivot(index='Time', columns='Column', values='A_median')
    im1 = axes[0, 0].imshow(pivot_A.values, cmap='RdYlGn_r', aspect='auto')
    axes[0, 0].set_xticks(range(len(pivot_A.columns)))
    axes[0, 0].set_xticklabels(pivot_A.columns)
    axes[0, 0].set_yticks(range(len(pivot_A.index)))
    axes[0, 0].set_yticklabels(pivot_A.index)
    axes[0, 0].set_title('a* Channel (Redness) Over Time')
    axes[0, 0].set_ylabel('Time (min)')
    axes[0, 0].set_xlabel('Column')
    plt.colorbar(im1, ax=axes[0, 0], label='a* value')
    
    # Heatmap: R channel over time
    pivot_R = col_avg.pivot(index='Time', columns='Column', values='R_median')
    im2 = axes[0, 1].imshow(pivot_R.values, cmap='Reds', aspect='auto')
    axes[0, 1].set_xticks(range(len(pivot_R.columns)))
    axes[0, 1].set_xticklabels(pivot_R.columns)
    axes[0, 1].set_yticks(range(len(pivot_R.index)))
    axes[0, 1].set_yticklabels(pivot_R.index)
    axes[0, 1].set_title('Red Channel Intensity Over Time')
    axes[0, 1].set_ylabel('Time (min)')
    axes[0, 1].set_xlabel('Column')
    plt.colorbar(im2, ax=axes[0, 1], label='R intensity')
    
    # Line plot: A* trajectories
    for col in columns:
        subset = col_avg[col_avg['Column'] == col]
        axes[1, 0].plot(subset['Time'], subset['A_median'], marker='o', label=f'Col {col}')
    axes[1, 0].set_xlabel('Time (min)')
    axes[1, 0].set_ylabel('a* median')
    axes[1, 0].set_title('a* Temporal Trajectories')
    axes[1, 0].legend()
    axes[1, 0].grid(True, alpha=0.3)
    
    # Variability plot: error bars
    for col in columns:
        subset = df[df['Column'] == col].groupby('Time').agg({
            'A_median': ['mean', 'std']
        }).reset_index()
        subset.columns = ['Time', 'mean', 'std']
        axes[1, 1].errorbar(subset['Time'], subset['mean'], yerr=subset['std'], 
                           marker='o', label=f'Col {col}', capsize=5)
    axes[1, 1].set_xlabel('Time (min)')
    axes[1, 1].set_ylabel('a* median ± SD')
    axes[1, 1].set_title('a* with Variability (across replicates)')
    axes[1, 1].legend()
    axes[1, 1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('enhanced_analysis.png', dpi=150)
    print("Saved: enhanced_analysis.png")
    
    # 5. Change Magnitude Summary
    print("\n5. CHANGE MAGNITUDE SUMMARY (t0 → t25)")
    print("-" * 60)
    
    for col in columns:
        subset = col_avg[col_avg['Column'] == col]
        changes = {
            'A*': subset.iloc[-1]['A_median'] - subset.iloc[0]['A_median'],
            'L*': subset.iloc[-1]['L_median'] - subset.iloc[0]['L_median'],
            'H': subset.iloc[-1]['H_median'] - subset.iloc[0]['H_median'],
            'R': subset.iloc[-1]['R_median'] - subset.iloc[0]['R_median'],
        }
        print(f"Col {col}: A*={changes['A*']:+.1f}, L*={changes['L*']:+.1f}, " +
              f"H={changes['H']:+.1f}, R={changes['R']:+.1f}")

if __name__ == "__main__":
    enhanced_temporal_analysis()
