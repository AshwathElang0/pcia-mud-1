import pandas as pd

df = pd.read_csv('temporal_data.csv')

# Let's average across the 3 rows
col_avg = df.groupby(['Time', 'Column']).mean().reset_index()

pivot_A = col_avg.pivot(index='Time', columns='Column', values='A_median')
pivot_H = col_avg.pivot(index='Time', columns='Column', values='H_median')
pivot_R = col_avg.pivot(index='Time', columns='Column', values='R_median')

print("=== A* Channel (Redness) ===")
print((pivot_A.iloc[-1] - pivot_A.iloc[0]).round(2))

print("\n=== Hue Channel ===")
print((pivot_H.iloc[-1] - pivot_H.iloc[0]).round(2))

print("\n=== R Channel ===")
print((pivot_R.iloc[-1] - pivot_R.iloc[0]).round(2))

print("\n=== Max A* at t=25 ===")
print(pivot_A.iloc[-1].round(2))
