import pandas as pd

df = pd.read_csv('temporal_data.csv')

# Ratios cancel out illumination changes
df['Red_Green_ratio'] = df['R_median'] / (df['G_median'] + 1)
df['A_L_ratio'] = df['A_median'] / (df['L_median'] + 1)  # Normalize by brightness

col_avg = df.groupby(['Time', 'Column']).mean().reset_index()

for col in sorted(col_avg['Column'].unique()):
    subset = col_avg[col_avg['Column'] == col]
    delta_rg = subset.iloc[-1]['Red_Green_ratio'] - subset.iloc[0]['Red_Green_ratio']
    print(f"Column {col}: ΔR/G = {delta_rg:+.4f}")
