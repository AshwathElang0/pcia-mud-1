import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RESULTS_DIR = os.path.join(BASE_DIR, "results", "statistical")
os.makedirs(RESULTS_DIR, exist_ok=True)

INPUT_CSV = os.path.join(RESULTS_DIR, "control_samples_data.csv")
OUTPUT_CSV = os.path.join(RESULTS_DIR, "control_samples_data_normalized_a_star.csv")
OUTPUT_PLOT = os.path.join(RESULTS_DIR, "control_normalized_median_a_star_scatter.png")


if not os.path.exists(INPUT_CSV):
    raise FileNotFoundError(
        f"Input file not found: {INPUT_CSV}. Run control_variance_analysis.py first."
    )


df = pd.read_csv(INPUT_CSV)
required_cols = {"Time", "Row", "A_star_median"}
missing_cols = required_cols - set(df.columns)
if missing_cols:
    raise ValueError(f"Missing required columns in input CSV: {sorted(missing_cols)}")

row0 = df[df["Row"] == 0].copy()
row1 = df[df["Row"] == 1].copy()

if row0.empty or row1.empty:
    raise ValueError("Both row 0 and row 1 data are required for normalization.")

# Mean a* for row 0 at each timestep.
row0_mean_by_time = row0.groupby("Time")["A_star_median"].mean().sort_index()

# Requested global stats: mean and std dev of those timestep-wise means.
target_mean = float(row0_mean_by_time.mean())
target_std = float(row0_mean_by_time.std(ddof=0))
# Results after running: target_mean = 139.240000, target_std = 1.194320

# Per-timestep row-0 source stats used for normalization of both rows.
row0_stats_by_time = (
    row0.groupby("Time")["A_star_median"]
    .agg(source_mean="mean", source_std=lambda s: float(np.std(s.to_numpy(), ddof=0)))
    .sort_index()
)

# Guard against zero std to avoid division by zero.
row0_stats_by_time["safe_source_std"] = row0_stats_by_time["source_std"].replace(0.0, 1.0)


def normalize_with_row0_stats(frame: pd.DataFrame) -> pd.DataFrame:
    merged = frame.merge(row0_stats_by_time, left_on="Time", right_index=True, how="left")
    z = (merged["A_star_median"] - merged["source_mean"]) / merged["safe_source_std"]
    merged["A_star_median_normalized"] = z * target_std + target_mean
    return merged


row0_norm = normalize_with_row0_stats(row0)
row1_norm = normalize_with_row0_stats(row1)
normalized_df = pd.concat([row0_norm, row1_norm], ignore_index=True).sort_values(["Time", "Row", "Col"])

normalized_df.to_csv(OUTPUT_CSV, index=False)

# Plot normalized scatter.
plt.figure(figsize=(10, 6))
plt.scatter(
    row0_norm["Time"],
    row0_norm["A_star_median_normalized"],
    color="blue",
    s=40,
    alpha=0.8,
    label="No bact (normalized)",
)
plt.scatter(
    row1_norm["Time"],
    row1_norm["A_star_median_normalized"],
    color="pink",
    s=40,
    alpha=0.8,
    label='Bact (normalized with "No bact" stats)',
)
plt.xlabel("Timesteps (min)")
plt.ylabel("Normalized median a*")
plt.title("Normalized Median a* Scatter by Timestep")
plt.grid(True, alpha=0.3)
plt.legend()
plt.tight_layout()
plt.savefig(OUTPUT_PLOT)
plt.close()

print("Row 0 timestep means:")
print(row0_mean_by_time)
print("\nGlobal target stats from row-0 timestep means:")
print(f"target_mean={target_mean:.6f}, target_std={target_std:.6f}")
print("\nPer-time row-0 source stats used for normalization:")
print(row0_stats_by_time[["source_mean", "source_std"]])
print(f"\nSaved normalized data: {OUTPUT_CSV}")
print(f"Saved normalized scatter: {OUTPUT_PLOT}")
