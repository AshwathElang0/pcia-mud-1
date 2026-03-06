import pandas as pd
import numpy as np
from scipy import stats
import argparse
import sys

def identify_mic(csv_path, threshold=0.1):
    try:
        df = pd.read_csv(csv_path)
    except FileNotFoundError:
        print(f"Error: {csv_path} not found.")
        sys.exit(1)

    # We want to perform linear regression on the a* channel (Red-Green) for each column
    # over time to detect biological growth (shift towards pink/red)

    # First, average the 3 replicate rows for each timepoint and each column
    col_avg = df.groupby(['Time', 'Column']).mean().reset_index()

    print("--- Temporal Growth Kinetics Analysis ---")
    print(f"Using a* slope threshold for growth: {threshold}\n")

    results = []
    
    # Analyze each column
    columns = sorted(col_avg['Column'].unique())
    for col in columns:
        col_data = col_avg[col_avg['Column'] == col]
        
        # Linear regression: independent variable is Time, dependent is A_median
        x = col_data['Time'].values
        y = col_data['A_median'].values
        
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        
        # Tag as growth if the regression slope exceeds our designated threshold
        growth = slope > threshold
        
        results.append({
            'Column': col,
            'A_Slope': round(slope, 3),
            'Growth_Detected': growth,
            'R_Squared': round(r_value**2, 3)
        })

    results_df = pd.DataFrame(results)
    print(results_df.to_string(index=False))
    print("-" * 40)

    # Identify MIC (Minimum Inhibitory Concentration)
    # The MIC is the lowest concentration (highest column index) that PREVENTS growth.
    # We look for the transition point where growth stops.
    mic_column = None
    
    # Scan from high conc (col 1) to low conc (col 7)
    # the first column going left-to-right that shows growth means the PREVIOUS column was the MIC.
    for i, res in enumerate(results):
        if res['Growth_Detected']:
            if i == 0:
                print("⚠️ Biological growth detected in ALL columns. MIC is higher than the max concentration tested (or assay failed).")
                mic_column = "Above Column 1"
            else:
                mic_column_idx = results[i-1]['Column']
                print(f"✅ MIC Identified: Column {mic_column_idx}")
                mic_column = mic_column_idx
            break
    
    if mic_column is None:
        print("⚠️ No growth detected in ANY column. MIC is lower than the minimum concentration tested (or bacteria failed to grow).")
        mic_column = "Below Column 7"

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Analyze temporal colorimetric data for MIC identification.")
    parser.add_argument("--csv", default="/home/ash/Desktop/acads/pcia/temporal_data.csv", help="Path to the temporal_data.csv file")
    parser.add_argument("--threshold", type=float, default=0.1, help="Positive linear slope threshold in the a* channel to classify as bacterial growth")
    args = parser.parse_args()

    identify_mic(args.csv, args.threshold)
