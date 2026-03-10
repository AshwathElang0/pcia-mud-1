"""
Statistical and ML Analysis of Colorimetric MIC Data
=====================================================
1. PCA - Identify which color channels carry the most discriminative information
2. K-Means Clustering - Group samples by temporal behavior patterns
3. Early Prediction - Can we predict the final outcome from early timepoints?
4. Reaction Kinetics - Fit exponential/sigmoid curves to model the reaction trajectory
5. Replicate Consistency - Assess how reliable our 3-row replicates are
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy import stats
from scipy.optimize import curve_fit
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
import warnings
import os
warnings.filterwarnings('ignore')

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CSV_PATH = os.path.join(BASE_DIR, 'results', 'temporal', 'temporal_data.csv')
RESULTS_DIR = os.path.join(BASE_DIR, 'results', 'statistical')

df = pd.read_csv(CSV_PATH)

col_avg = df.groupby(['Time', 'Column']).mean(numeric_only=True).reset_index()

COLOR_FEATURES = ['R_median', 'G_median', 'B_median', 'L_median', 'A_median', 'B_lab_median', 'H_median', 'S_median', 'V_median']

# ============================================================
# 1. PCA - Which color channels matter most?
# ============================================================
print("=" * 60)
print("1. PCA: Feature Importance Across Color Spaces")
print("=" * 60)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(col_avg[COLOR_FEATURES])

pca = PCA()
pca.fit(X_scaled)

print("\nExplained variance ratio per component:")
for i, var in enumerate(pca.explained_variance_ratio_):
    print(f"  PC{i+1}: {var:.3f} ({sum(pca.explained_variance_ratio_[:i+1])*100:.1f}% cumulative)")

print("\nTop feature loadings for PC1 and PC2:")
loadings = pd.DataFrame(pca.components_[:2].T, columns=['PC1', 'PC2'], index=COLOR_FEATURES)
loadings['PC1_abs'] = loadings['PC1'].abs()
print(loadings.sort_values('PC1_abs', ascending=False)[['PC1', 'PC2']].to_string())

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

X_pca = pca.transform(X_scaled)
scatter = axes[0].scatter(X_pca[:, 0], X_pca[:, 1], c=col_avg['Column'], cmap='viridis', s=80, edgecolors='k', linewidths=0.5)
plt.colorbar(scatter, ax=axes[0], label='Column')
for i, row in col_avg.iterrows():
    axes[0].annotate(f"t{int(row['Time'])}", (X_pca[i, 0], X_pca[i, 1]), fontsize=6, alpha=0.7)
axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.1f}%)")
axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.1f}%)")
axes[0].set_title("PCA: Samples in PC Space (colored by Column)")
axes[0].grid(True, alpha=0.3)

loadings[['PC1', 'PC2']].plot(kind='bar', ax=axes[1])
axes[1].set_title("PCA Feature Loadings")
axes[1].set_ylabel("Loading")
axes[1].tick_params(axis='x', rotation=45)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'pca_analysis.png'), dpi=150)
print(f"\nSaved: pca_analysis.png")

# ============================================================
# 2. K-Means Clustering of Temporal Trajectories
# ============================================================
print("\n" + "=" * 60)
print("2. K-Means Clustering: Grouping Columns by Behavior")
print("=" * 60)

trajectory_features = []
columns_list = sorted(col_avg['Column'].unique())

for col in columns_list:
    col_data = col_avg[col_avg['Column'] == col].sort_values('Time')
    features = col_data[COLOR_FEATURES].values.flatten()
    trajectory_features.append(features)

trajectory_features = np.array(trajectory_features)
traj_scaled = StandardScaler().fit_transform(trajectory_features)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

for ax, k in zip(axes, [2, 3]):
    kmeans = KMeans(n_clusters=k, random_state=42, n_init='auto')
    labels = kmeans.fit_predict(traj_scaled)
    
    pca_traj = PCA(n_components=2)
    traj_2d = pca_traj.fit_transform(traj_scaled)
    
    for cluster_id in range(k):
        mask = labels == cluster_id
        ax.scatter(traj_2d[mask, 0], traj_2d[mask, 1], s=150, label=f'Cluster {cluster_id}', edgecolors='k', linewidths=0.5)
        for idx in np.where(mask)[0]:
            ax.annotate(f'Col {columns_list[idx]}', (traj_2d[idx, 0], traj_2d[idx, 1]), fontsize=9, ha='center', va='bottom')
    
    ax.set_title(f'K-Means (k={k}) on Temporal Trajectories')
    ax.set_xlabel('Trajectory PC1')
    ax.set_ylabel('Trajectory PC2')
    ax.legend()
    ax.grid(True, alpha=0.3)

    print(f"\nK={k} cluster assignments:")
    for col, label in zip(columns_list, labels):
        print(f"  Column {col} -> Cluster {label}")

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'clustering_analysis.png'), dpi=150)
print(f"\nSaved: clustering_analysis.png")

# ============================================================
# 3. Early Prediction: Can t=0 or t=5 predict the final state?
# ============================================================
print("\n" + "=" * 60)
print("3. Early Prediction: Forecasting Reaction Destination")
print("=" * 60)

growth_labels = {}
for col in columns_list:
    col_data = col_avg[col_avg['Column'] == col].sort_values('Time')
    slope, _, _, _, _ = stats.linregress(col_data['Time'], col_data['A_median'])
    growth_labels[col] = 1 if slope > 0.1 else 0

print("\nGround truth labels (from a* slope analysis):")
for col, label in growth_labels.items():
    print(f"  Column {col}: {'GROWTH (below MIC)' if label else 'INHIBITED (above MIC)'}")

for cutoff_time in [0, 5, 10]:
    print(f"\n--- Predicting from t=0 to t={cutoff_time} ---")
    
    early_data = df[df['Time'] <= cutoff_time]
    
    X_early = []
    y_early = []
    
    for col in columns_list:
        for row in df['Row'].unique():
            sample_data = early_data[(early_data['Column'] == col) & (early_data['Row'] == row)]
            if len(sample_data) > 0:
                features = sample_data[COLOR_FEATURES].mean().values
                X_early.append(features)
                y_early.append(growth_labels[col])
    
    X_early = np.array(X_early)
    y_early = np.array(y_early)
    
    if len(np.unique(y_early)) < 2:
        print("  Only one class present, skipping.")
        continue
    
    scaler_early = StandardScaler()
    X_early_scaled = scaler_early.fit_transform(X_early)
    
    clf = LogisticRegression(random_state=42, max_iter=1000)
    clf.fit(X_early_scaled, y_early)
    
    y_pred = clf.predict(X_early_scaled)
    accuracy = (y_pred == y_early).mean()
    
    print(f"  Training accuracy: {accuracy*100:.1f}%")
    print(f"  Classification report:")
    print(classification_report(y_early, y_pred, target_names=['Inhibited', 'Growth'], zero_division=0))
    
    importance = pd.Series(np.abs(clf.coef_[0]), index=COLOR_FEATURES).sort_values(ascending=False)
    print(f"  Top predictive features: {', '.join(importance.head(3).index.tolist())}")

# ============================================================
# 4. Reaction Kinetics: Curve Fitting
# ============================================================
print("\n" + "=" * 60)
print("4. Reaction Kinetics: Exponential/Linear Curve Fitting")
print("=" * 60)

def linear_model(t, a, b):
    return a * t + b

def exponential_model(t, a, b, c):
    return a * np.exp(b * t) + c

fig, axes = plt.subplots(2, 4, figsize=(18, 8))
axes = axes.flatten()

kinetics_results = []

for i, col in enumerate(columns_list):
    ax = axes[i]
    col_data = col_avg[col_avg['Column'] == col].sort_values('Time')
    
    t = col_data['Time'].values.astype(float)
    a_star = col_data['A_median'].values
    
    slope_lin, intercept_lin, r_lin, p_lin, se_lin = stats.linregress(t, a_star)
    
    try:
        popt_exp, _ = curve_fit(exponential_model, t, a_star, p0=[1, 0.01, 130], maxfev=5000)
        a_star_pred_exp = exponential_model(t, *popt_exp)
        ss_res_exp = np.sum((a_star - a_star_pred_exp)**2)
        ss_tot = np.sum((a_star - np.mean(a_star))**2)
        r2_exp = 1 - ss_res_exp / ss_tot if ss_tot > 0 else 0
    except:
        popt_exp = None
        r2_exp = 0
    
    kinetics_results.append({
        'Column': col,
        'Linear_Slope': round(slope_lin, 4),
        'Linear_R2': round(r_lin**2, 3),
        'Linear_p': round(p_lin, 4),
        'Exp_R2': round(r2_exp, 3),
        'Growth_Label': growth_labels[col]
    })
    
    t_smooth = np.linspace(0, 25, 100)
    ax.scatter(t, a_star, color='blue', s=60, zorder=5)
    ax.plot(t_smooth, linear_model(t_smooth, slope_lin, intercept_lin), 'r--', label=f'Linear (R²={r_lin**2:.2f})')
    if popt_exp is not None:
        ax.plot(t_smooth, exponential_model(t_smooth, *popt_exp), 'g-', label=f'Exp (R²={r2_exp:.2f})')
    ax.set_title(f'Col {col} {"🟢" if growth_labels[col] else "🔴"}')
    ax.set_xlabel('Time (min)')
    ax.set_ylabel('a* value')
    ax.legend(fontsize=7)
    ax.grid(True, alpha=0.3)

axes[7].axis('off')

plt.suptitle('Reaction Kinetics: a* Channel Over Time per Column', fontsize=14)
plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'kinetics_analysis.png'), dpi=150)
print(f"Saved: kinetics_analysis.png")

kinetics_df = pd.DataFrame(kinetics_results)
print("\nKinetics Summary:")
print(kinetics_df.to_string(index=False))

# ============================================================
# 5. Replicate Consistency: Inter-row variance
# ============================================================
print("\n" + "=" * 60)
print("5. Replicate Consistency: How Reliable Are the 3 Rows?")
print("=" * 60)

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

cv_data = df.groupby(['Time', 'Column'])['A_median'].agg(['mean', 'std']).reset_index()
cv_data['CV_pct'] = (cv_data['std'] / cv_data['mean']) * 100

pivot_cv = cv_data.pivot(index='Time', columns='Column', values='CV_pct')
pivot_cv.plot(kind='bar', ax=axes[0], colormap='viridis')
axes[0].set_title('Coefficient of Variation (%) of a* Across Replicates')
axes[0].set_xlabel('Time (min)')
axes[0].set_ylabel('CV (%)')
axes[0].legend(title='Column')
axes[0].grid(True, alpha=0.3)

feature_std = df.groupby('Column')[COLOR_FEATURES].std().mean(axis=0)
feature_std.plot(kind='bar', ax=axes[1], color='coral')
axes[1].set_title('Mean Std Dev Across Rows (all columns, all timepoints)')
axes[1].set_ylabel('Std Dev')
axes[1].tick_params(axis='x', rotation=45)
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(os.path.join(RESULTS_DIR, 'replicate_consistency.png'), dpi=150)
print(f"Saved: replicate_consistency.png")

print("\nMean CV% per column (a* channel):")
print(cv_data.groupby('Column')['CV_pct'].mean().round(2).to_string())

print("\n" + "=" * 60)
print("ANALYSIS COMPLETE")
print("=" * 60)
