======================================================================
OUTLIER DETECTION ANALYSIS
======================================================================

1. Loading data...
   Loaded 8,720 rows, 30 columns

2. Analyzing 28 numeric columns for outliers...

======================================================================
OUTLIER DETECTION - Z-SCORE METHOD (Threshold: 3 std)
======================================================================

Columns with outliers (Z-score > 3):
Column                    Outliers     Percentage   Max Z-Score
----------------------------------------------------------------------
High                      273          3.13%      4.79
Close                     271          3.11%      4.81
Open                      271          3.11%      4.82
Low                       268          3.07%      4.82
Volume                    190          2.18%      12.86
VIX                       174          2.00%      8.06
Put_Call_Ratio            174          2.00%      8.06
sp500_Volume              135          1.55%      6.53
Unemployment_Rate         82           0.94%      4.62
sector_XLK                28           0.32%      3.27
spy_RSI                   17           0.19%      3.45
qqq_RSI                   6            0.07%      3.15

======================================================================
OUTLIER DETECTION - IQR METHOD (1.5 * IQR)
======================================================================

Columns with outliers (IQR method):
Column                    Outliers     Percentage   Range
----------------------------------------------------------------------
sector_XLE                1109         12.72%      [17.76, 79.78]
Volume                    1024         11.74%      [-69243475.00, 173746725.00]
sp500_Volume              530          6.08%      [1556878750.00, 6122428750.00]
Low                       461          5.29%      [-189.68, 368.12]
Close                     460          5.28%      [-192.57, 373.60]
Open                      459          5.26%      [-191.95, 372.66]
High                      456          5.23%      [-195.32, 378.83]
VIX                       391          4.48%      [1.59, 33.08]
Put_Call_Ratio            391          4.48%      [1.59, 33.08]
Unemployment_Rate         126          1.44%      [0.10, 10.50]

======================================================================
CONCRETE OUTLIER EXAMPLES
======================================================================

EXAMPLE 1: Volume Outliers (Extreme Trading Days)
----------------------------------------------------------------------

Top 5 Volume Outliers:
Date         Stock  Volume               Z-Score    Context
----------------------------------------------------------------------
2004-08-19   GOOGL         893,181,924     6.69 Close=$2.49
2004-09-29   GOOGL         610,329,060     4.36 Close=$3.26
2004-10-21   GOOGL         582,996,420     4.14 Close=$3.71
2004-08-20   GOOGL         456,686,856     3.10 Close=$2.69
2004-10-20   GOOGL         454,453,092     3.08 Close=$3.49

EXAMPLE 2: Close Price Outliers (Extreme Stock Prices)
----------------------------------------------------------------------

Top 5 Close Price Outliers (Highest):
Date         Stock  Close ($)    Z-Score    Volume
----------------------------------------------------------------------
2024-09-23   META   $   562.64     3.19     12,830,700
2024-09-20   META   $   559.59     3.17     22,066,800
2024-09-19   META   $   557.35     3.15     15,647,000
2024-07-05   META   $   537.70     3.01     21,354,100
2024-09-18   META   $   536.26     3.00     10,323,500

Lowest Close Price Outliers:
Date         Stock  Close ($)    Z-Score    Context
----------------------------------------------------------------------
2024-09-18   META   $   536.26     3.00 (IPO period)
2024-07-05   META   $   537.70     3.01 (IPO period)
2025-04-11   META   $   542.79     3.05 (IPO period)

EXAMPLE 3: VIX Outliers (Market Stress/Fear)
----------------------------------------------------------------------

Top 5 VIX Outliers (High Fear/Volatility):
Date         VIX        Z-Score    Stock  Close      Context
----------------------------------------------------------------------
2008-10-07      53.68     4.42 GOOGL  $   8.60 Financial Crisis
2008-10-06      52.05     4.21 GOOGL  $   9.23 Financial Crisis
2008-09-29      46.72     3.54 GOOGL  $   9.47 Financial Crisis
2008-10-02      45.26     3.36 GOOGL  $   9.71 Financial Crisis
2008-10-03      45.14     3.34 GOOGL  $   9.62 Financial Crisis

EXAMPLE 4: Rows with Multiple Outliers
----------------------------------------------------------------------

Rows with Most Outlier Columns:
Date         Stock  Outlier Count   Example Outliers
----------------------------------------------------------------------
2025-04-08   META   6               High, Open, sp500_Volume
2024-09-20   META   5               Close, High, Low
2024-12-20   META   5               Close, High, Low

======================================================================
METHOD COMPARISON: Z-SCORE vs IQR
======================================================================

Columns detected by both methods: 12

Sample comparison for Volume:
  Z-Score method: 190 outliers
  IQR method: 1024 outliers
  Difference: 834 outliers

  Why difference?
    - Z-Score: Based on mean/std (sensitive to extreme values)
    - IQR: Based on quartiles (more robust to extremes)

======================================================================
OUTLIER STATISTICS SUMMARY
======================================================================

Z-Score Method:
  Columns with outliers: 12
  Total outlier instances: 1,889
  Percentage of dataset: 0.77%

IQR Method:
  Columns with outliers: 13
  Total outlier instances: 5,477
  Percentage of dataset: 2.24%

======================================================================
SAVING OUTLIER DETECTION RESULTS
======================================================================

   OK Outlier summary saved to: data/outlier_detection_summary.csv

======================================================================
OUTLIER DETECTION COMPLETE
======================================================================