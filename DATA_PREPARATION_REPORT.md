# Data Preparation Report

This report details the transformation of the dataset through the data preparation pipeline.

## 1. Raw Data Overview
- **Total Rows:** 8,720
- **Total Columns:** 30
- **Date Range:** 2004-08-19 to 2025-10-31
- **Stocks:** GOOGL, META
- **Total Null Values:** 48 (0.02% of data)

## 2. Technique: Smart Imputation
### Logic Applied
- **Macro Data (CPI, GDP):** Forward Fill then Backward Fill (handling reporting lags).
- **Stock Data:** Forward Fill (propagating last known price).
- **Interpolation:** Linear interpolation for scattered missing values.
### Impact
- **Rows Removed:** 0 (All nulls were handled by imputation)
- **Nulls Remaining:** 0
- **Insight:** By using smart imputation instead of dropping rows with nulls, we preserved **100%** of the temporal structure, which is critical for time-series modeling.

## 3. Technique: Outlier Removal
### Logic Applied
- **Method:** Z-Score
- **Threshold:** 3 Standard Deviations
- **Action:** Remove entire row if any column has an outlier.
### Impact
- **Rows Before:** 8,720
- **Rows After:** 7,930
- **Rows Removed:** 790 (9.06%)
### Top Outlier Contributors
| Column | Outliers Detected |
| :--- | :--- |
| High | 273 |
| Close | 271 |
| Open | 271 |
| Low | 268 |
| Volume | 190 |

- **Insight:** The majority of outliers came from **Volume** and **Returns**, which often spike during market events. Removing these prevents the model from overfitting to extreme, non-repeatable events (flash crashes, earnings surprises).

## 4. Technique: Feature Scaling
### Logic Applied
- **StandardScaler:** Applied to normally distributed features (e.g., Returns).
- **RobustScaler:** Applied to features with heavy tails (e.g., Volume).
- **MinMaxScaler:** Applied to bounded ratios.
### Impact
- **Rows Removed:** 0 (Transformation only)
### Example Transformation (Volume)
- **Before Scaling:** Mean = 66,356,002, Std = 76,925,915, Range = [5,467,500, 443,843,712]
- **After Scaling:** Mean = 0.0000, Std = 1.0000, Range = [-0.7915, 4.9072]

- **Insight:** Scaling normalized the range of all features to be roughly comparable (centered around 0). This is crucial for models like Neural Networks and helps Random Forests converge faster by removing the bias towards features with larger raw numbers (like Volume vs Interest Rate).

## 5. Final Dataset Status
- **Final Rows for Training:** 7,762
- **Total Features:** 34 (excluding target)
- **Note:** 168 rows were dropped at the very end. These correspond to the last 21 days of data for each stock, for which we cannot calculate the 'Future Return' target variable yet.