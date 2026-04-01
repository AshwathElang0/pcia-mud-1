# Control Row Experiment Interpretation

## Scope
This report interprets the generated outputs for control-row tracking using a* medians across 5 timepoints (0, 5, 10, 20, 30 min), with 10 disks per row per timepoint.

Data sources used:
- `results/statistical/csv/a_medians_summary.csv`
- `results/statistical/jsd/jsd_over_time.csv`
- `results/statistical/hypothesis/ks_over_time.csv`

## Core Observations

### 1) Strong global shift at 5 min in both rows
- Dye row median a* rises from 103.5 to 139.0 (+35.5).
- Bacteria row median a* rises from 104.5 to 128.5 (+24.0).

Inference:
- Because both rows move in the same direction at 5 min, this is most consistent with a global imaging/illumination shift rather than biology-only change.

### 2) Rows are close at baseline, then diverge, then partially reconverge
- Baseline (0 min): bacteria-dye median gap is +1.0 (very close).
- 5 min: bacteria is lower than dye by -10.5.
- 10 min: gap remains large at -11.0.
- 20 min: bacteria is higher by +3.5.
- 30 min: bacteria is lower by -4.0.

Inference:
- Relative ordering between rows is not stable over time, supporting the need for control-based normalization and distributional tracking instead of single-point thresholds.

### 3) Bacteria row is generally more variable than dye row
- Standard deviation is higher for bacteria at every timepoint.
- Example at 20 min: bacteria std 7.25 vs dye std 3.72.

Inference:
- Even after per-disk median extraction, the bacteria row retains greater intra-row heterogeneity. This could reflect medium roughness, segmentation variation, or true row-level variability.

### 4) Distributional change is largest early, then decreases
Jensen-Shannon distance between consecutive timepoints:
- Dye: 0->5: 0.730, 5->10: 0.754, 10->20: 0.577, 20->30: 0.328
- Bacteria: 0->5: 0.724, 5->10: 0.833, 10->20: 0.428, 20->30: 0.270

Inference:
- Largest distribution reshaping occurs in the first two transitions (0->5 and 5->10), then both rows become more stable.

### 5) Hypothesis testing agrees with the JSD pattern
KS test between consecutive timepoints:
- Highly significant early changes:
  - Dye 0->5: p=1.08e-05
  - Dye 5->10: p=1.08e-05
  - Bacteria 0->5: p=2.17e-04
  - Bacteria 5->10: p=1.08e-05
- Later transitions are weaker/non-significant:
  - 10->20: p=0.052 (both rows; borderline)
  - 20->30: dye p=0.168, bacteria p=0.787

Inference:
- Statistically detectable distribution shifts are concentrated early; after ~10 min, observed changes are smaller and less consistently distinguishable from within-row variability.

## Practical Conclusions
1. Control-based correction is justified and necessary.
2. Timepoints 5 and 10 min are dominated by strong global/distributional shifts, likely including lighting effects.
3. Later timepoints (20 to 30 min) appear comparatively stable in distributional terms.
4. Distribution-aware metrics (JSD + KS), not only central tendency, provide better monitoring of assay behavior under medium irregularity.

## Recommended Next Steps
1. Add baseline-referenced JSD (each timepoint vs 0 min) in addition to consecutive JSD.
2. Track effect sizes (e.g., Cliff's delta or Wasserstein distance) along with p-values.
3. Use normalized a* distributions for primary interpretation and keep unnormalized outputs for QA/audit.
4. Keep visual QA of SAM masks per timestep to ensure segmentation drift is not confounding statistical drift.

## Caveats
- Sample size per row/timepoint is small (n=10), so KDE and KS sensitivity are limited.
- Conclusions depend on segmentation consistency; mask errors can inflate apparent variability.
- The current analysis focuses on a* only; adding L* and b* can help separate illumination from chromatic shifts.
