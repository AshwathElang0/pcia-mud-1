# Colorimetric Analysis Methodology

This document outlines the various analytical techniques and machine learning approaches implemented to extract meaningful biological insights from the temporal sequence of colorimetric images (`0th_min.jpeg` to `25th_min.jpeg`) of the bacterial-dye-antibiotic arrays.

## 1. Grid Detection and Median Color Extraction

To track the same physical samples across multiple timepoints, we use a robust spatial localization methodology:

- **Histogram Projection:** We convert the image to grayscale, threshold it, and apply a morphology opening operation to isolate the bright sample wells from the dark tray background. We then sum the pixel intensities along both the X and Y axes to produce 1D projection histograms. 
- **Peak Finding:** The `scipy.signal.find_peaks` algorithm identifies the coordinates of the well centers by picking the most prominent peaks in the projected histograms. This reliably locates our 3x9 grid matrix regardless of minor image rotations or translations between consecutive photo captures.
- **Median Extraction:** At each identified well location (skipping the control columns on the edges), we draw a 32-pixel radius circular mask. We extract the median intensity of all pixels inside this mask across three color spaces: RGB, CIELAB (L* a* b*), and HSV. Taking the median prevents artifact outliers (e.g. lighting glare) from skewing the measurement.

## 2. Temporal Minimum Inhibitory Concentration (MIC) Identification

The primary experimental objective is to identify the Minimum Inhibitory Concentration (MIC) — the minimum amount of antibiotic needed to completely stop bacterial growth. We automate this analysis through the `a*` channel (green-red axis) kinetics.

- **Growth Indicator:** The biological indicator dye transitions from blue (high $a^*$ value meaning less red) to pink (higher $a^*$ value meaning more red) as surviving bacteria metabolize it. 
- **Linear Kinetics Thresholding:** For each column (representing a single antibiotic concentration), we aggregate data across the 3 replicate rows and measure the trajectory of the median $a^*$ value over time ($t=0$ to $t=25$). 
- **Algorithmic Selection:** We apply monotonic linear regression (`scipy.stats.linregress`) to the $a^*$ points against time. We define a predetermined threshold slope (e.g. `m > 0.1`). Any column exceeding this growth slope indicates bacterial survival. 
- **Result:** The script scans from the highest antibiotic concentration to the lowest. The final concentration column immediately preceding the first column that exhibits a positive growth slope is programmatically flagged as the MIC.

## 3. Principal Component Analysis (PCA)

We use PCA to overcome the dimensionality of extracting 9 distinct color features (R, G, B, L*, a*, b*, H, S, V) per sample per timepoint.

- **Dimensionality Reduction:** By normalizing and transforming all features, PCA projects the high-dimensional data into orthogonal axes (Principal Components) ranked by the amount of variance they explain. 
- **Feature Importance:** By analyzing the component loadings for PC1 and PC2 (which capture >90% of total image dataset variance), we identify which color channels carry the most inherent discriminatory information. For example, our data shows that the Green (G), Lightness (L*), and Saturation (S) channels encode more variance across the sequence than the traditional red-blue axes.

## 4. K-Means Behavioral Clustering

Rather than looking at concentrations linearly, K-Means helps us identify unprompted discrete groupings in the sample reactions.

- **Trajectory Flattening:** We flatten each column's temporal series ($t_0$ features to $t_{25}$ features) into a single, massive 1D feature vector that represents its "behavioral signature" over the experiment.
- **Clustering:** We use unsupervised K-Means clustering (testing $k=2$ and $k=3$) to categorize these temporal signatures.
- **Insight:** This reveals if the reactions fall into expected binary categories (Growth vs Death), or if intermediate sub-clusters exist (e.g. partial inhibition, delayed growth kinetics), helping chemists understand if the transition out of MIC is a sharp binary cliff or a gradual spectrum.

## 5. Early Prediction Forecasting (Logistic Regression)

We perform a predictive modeling exercise to determine if the final biological outcome at $t=25$ can be predicted reliably without waiting for the full timecourse.

- **Model Setup:** We label the final outcome of each column using the MIC linear slope test (1 = Growth, 0 = Inhibited).
- **Classification:** We train a Logistic Regression classifier using *only* the colorimetric readings at $t=0$. The objective is to evaluate whether subtle initial hue interactions between the antibiotic and the dye provide enough data to forecast the final survival state.
- **Insight:** Although constrained by a small dataset size, evaluating the precision and classification report of early predictors can potentially help chemists optimize their protocols and terminate failing experiments immediately instead of waiting for full incubation.

## 6. Curvilear Reaction Kinetics Modeling 

While linear slope thresholding serves nicely for simple MIC identification, fitting true kinetic equations reveals the biological rate dynamics.

- **Non-linear Fits:** For every column, we use `scipy.optimize.curve_fit` to overlay exponential models over the $a^*$ channel datapoints. 
- **Insight:** Extracting the $R^2$ exponential decay/growth coefficients provides standard biological insights regarding the deceleration rate of the enzyme activity and the specific mechanism of the bacteriostatic vs bactericidal drug being tested.

## 7. Replicate Consistency and Variance Assessment

To validate the reliability of this colorimetric capture method in uncontrolled settings:

- **Coefficient of Variation (CV%):** We calculate the standard deviation expressed as a percentage of the mean ($\sigma / \mu \times 100$) across the 3 spatially distinct row replicates for every single column at every single timepoint. 
- **Insight:** High CV% localized to specific columns signals experimental noise (e.g. pipetting discrepancies, edge-effect evaporation). Plotting CV% verifies our analytical baseline and ensures outlier artifacts do not invalidate our algorithmic MIC calculations.
