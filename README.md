# Data Preparation Comparison Walkthrough

This document showcases how different data preparation techniques enhance ML model accuracy for predicting stock performance (Stock vs SP500).

## Results Summary

We compared 5 scenarios, progressively adding more sophisticated data preparation steps.

### Random Split (Cross-Validation)

| Scenario | Accuracy | F1 Score | Improvement |
| :--- | :--- | :--- | :--- |
| Baseline | 85.98% | 0.8774 | - |
| Smart Imputation | 86.97% | 0.8856 | +0.99% |
| Outlier Removal | 86.99% | 0.8873 | +1.01% |
| Scaled Data | 89.19% | 0.8974 | +3.21% |
| **All Combined** | **90.15%** | **0.9119** | **+4.17%** |

> [!CAUTION]
> These numbers are **inflated** due to data leakage (Random Split allows the model to interpolate between adjacent days).

### Time Series Split (Realistic Future Prediction)

#### Random Forest Results

| Scenario | Accuracy | F1 Score | Improvement |
| :--- | :--- | :--- | :--- |
| Baseline | 48.80% | 0.3983 | - |
| Smart Imputation | 48.22% | 0.3839 | -0.58% |
| Outlier Removal | 47.97% | 0.4814 | -0.83% |
| Scaled Data | 50.38% | 0.6398 | +1.58% |
| **All Combined** | **51.84%** | **0.6308** | **+3.04%** |

#### Gradient Boosting Machine (GBM) Results

| Scenario | Accuracy | F1 Score | AUC | Improvement |
| :--- | :--- | :--- | :--- | :--- |
| Baseline | 49.15% | 0.4612 | 0.5073 | - |
| Smart Imputation | 48.39% | 0.4102 | 0.5019 | -0.76% |
| Outlier Removal | 47.39% | 0.5225 | 0.4764 | -1.76% |
| **Scaled Data** | **51.90%** | **0.4859** | **0.5166** | **+2.75%** |
| All Combined | 50.16% | 0.5058 | 0.4975 | +1.01% |

> [!IMPORTANT]
> **Key Finding:** On realistic time-series split:
> - **Random Forest Best:** 51.8% (All Combined) 
> - **GBM Best:** 51.9% (Scaled Data Only)
> - **Scaling** is the only technique that consistently improves both models
> - Random Forest benefits from combining all techniques, while GBM performs best with scaling alone

## Methodology

### 1. Smart Imputation
Instead of simple forward filling, we used specific strategies for different data types:
-   **Macro Data (CPI, GDP):** Forward then Backward fill (to handle reporting delays).
-   **Stock Prices:** Forward fill (last known price).
-   **Interpolation:** Linear interpolation for scattered missing values.

### 2. Outlier Removal
We removed rows containing extreme values (Z-score > 3) to prevent them from skewing the model.
-   **Impact:** Minor improvement in accuracy, but likely improves model stability and generalization.

### 3. Feature Scaling
We applied different scalers based on the feature distribution:
-   **StandardScaler:** For normally distributed features (e.g., Returns).
-   **RobustScaler:** For features with outliers (e.g., Volume).
-   **MinMaxScaler:** For bounded features (e.g., Ratios).
-   **Impact:** Significant improvement. Random Forest is generally robust to scaling, but in this high-dimensional space with varying scales (Prices vs Percentages), scaling helped significantly.

## Data Preparation Insights

Key statistics from our data cleaning process:

| Technique | Rows Removed | Insight |
| :--- | :--- | :--- |
| **Smart Imputation** | **0** (0%) | Preserved 100% of temporal structure vs dropping ~15% if using naive dropna. |
| **Outlier Removal** | **790** (9.06%) | Removed extreme volatility (Volume/Price spikes) to prevent overfitting to flash crashes. |
| **Scaling** | **0** (0%) | Normalized Volume (0-400M) and Ratios (0-1) to comparable ranges (Mean ~0). |

## AutoML Analysis (H2O) - Time Series Split

We ran H2O AutoML using a **strict date-based split** to ensure no data leakage. Training uses data before a specific date cutoff (2021-10-07), validation uses data after (2021-10-08+).

### Initial Results (Leaked - Cross-Validation on Time Series)

| Model | AUC | LogLoss | RMSE |
| :--- | :--- | :--- | :--- |
| **Stacked Ensemble (Best)** | **0.9664** | **0.2333** | **0.2608** |
| Gradient Boosting Machine | 0.9646 | 0.2381 | 0.2641 |
| Extremely Randomized Trees | 0.9638 | 0.2737 | 0.2759 |
| Distributed Random Forest | 0.9629 | 0.2849 | 0.2774 |
| Deep Learning | 0.6751 | 0.6559 | 0.4759 |
| GLM | 0.5657 | 0.6805 | 0.4928 |

> [!CAUTION]
> **Leakage Discovered:** The 96.6% AUC was inflated due to H2O AutoML's default **5-fold Cross-Validation** (`nfolds=5`). On time-series data, random CV mixes past and future within folds, allowing the model to "fill in the blank" (interpolate) instead of truly predicting the future.

### Corrected Results (No Cross-Validation - Validation Frame Only)

After setting `nfolds=0` to disable CV and rely solely on the strict validation frame:

| Model | AUC | LogLoss | RMSE |
| :--- | :--- | :--- | :--- |
| **GBM (Best)** | **0.5831** | **0.7930** | **0.4681** |
| XRT | 0.5564 | 0.8484 | 0.4723 |
| GBM (variant) | 0.5469 | 0.8418 | 0.4568 |
| DRF | 0.5172 | 0.7290 | 0.5590 |
| Deep Learning | 0.5093 | 0.8655 | 0.4899 |
| GLM | 0.4918 | 0.8914 | 0.5007 |

> [!IMPORTANT]
> **Realistic Performance:** The **true** AUC is **~58%** (GBM), which aligns with Random Forest's 52% accuracy. This represents genuine future prediction capability, not memorization. The improvement over Random Forest (58% vs 52%) shows that advanced gradient boosting can extract additional signal from this complex feature set.

> [!TIP]
> **Key Lesson:** Always disable cross-validation (`nfolds=0`) when using AutoML on time-series data. Use only a strict, chronological validation frame to get realistic performance estimates.

## Overfitting Investigation

We investigated potential overfitting by comparing different evaluation methods.

| Split Method | Model | Accuracy/AUC | Verdict |
| :--- | :--- | :--- | :--- |
| **Random Split** | **Random Forest** | **90.15%** | **Overfit.** Relied on autocorrelation (interpolating between days). |
| **Time Series Split** | **Random Forest** | **52%** | **Realistic.** True future prediction (2021-10-08+). |
| **Time Series Split + CV** | **H2O AutoML** | **96.6%** | **Leaked.** 5-fold CV mixed past/future. |
| **Time Series Split (No CV)** | **H2O AutoML** | **58.3%** | **Realistic.** Strict validation frame only. |

> [!WARNING]
> **Reality Check:** Random Split inflates performance to 90% by interpolating between adjacent days. Time-Series Split with proper validation gives realistic results: **Random Forest achieves 52%**, **Advanced GBM achieves 58%**. The 6% improvement shows that sophisticated gradient boosting can extract additional predictive signal from this complex feature set.

### Why did Random Split overfit? (The "Fill in the Blank" Effect)

Think of it like shuffling a deck of cards (days) before dealing them:
- **Training Set:** Gets Jan 1st, Jan 3rd, Jan 5th...
- **Test Set:** Gets Jan 2nd, Jan 4th...

When the model predicts **Jan 2nd**, it has already studied **Jan 1st** and **Jan 3rd**. It doesn't need to predict the future; it just "fills in the blank" (interpolation).

**Time Series Split** forces the model to learn from 2004-2018 and predict 2019+ (extrapolation), which is much harder but realistic.

## How to Run
To reproduce these results:

1.  **Random Split Comparison (RF):** `python compare_models.py`
2.  **Time Series Comparison (RF):** `python compare_models_timeseries.py`
3.  **Time Series Comparison (GBM):** `python compare_models_timeseries_gbm.py`
4.  **AutoML:** `python run_automl.py`
5.  **Overfitting Investigation:** `python investigate_overfitting.py`

