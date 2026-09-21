# E-Commerce Purchase Prediction

A regression project predicting **transaction value** for e-commerce orders from customer, cart, and delivery signals — built end-to-end on a deliberately messy real-world-style dataset.

## Problem Statement

An online retailer wants to estimate how much a customer is likely to spend on a given order *before* it's finalized. This estimate can support dynamic discount sizing, delivery cost forecasting, and inventory/promotion planning by category. The target variable, `Transaction_Amount`, is continuous, making this a **regression** problem.

## Dataset

- **Source:** `ecommerce_purchase.csv` — 5,055 raw transactions, 12 columns (11 features + target)
- **Features:** `Customer_Age`, `Membership_Years`, `Items_In_Cart`, `Discount_Pct`, `Delivery_Distance_KM`, `Product_Category`, `Payment_Method`, `Is_Premium_Member`, `Season`, `Day_Of_Week`
- **Target:** `Transaction_Amount`
- The dataset was supplied intentionally messy — inconsistent casing, out-of-range values, and missing data are part of the problem, not an oversight.

## Data Quality Issues Found

| Issue | Detail |
|---|---|
| Inconsistent categories | 16 spellings of 4 payment methods; 10 variants of a 2-value membership flag (including `Y`/`N` shorthand) |
| Malformed numeric column | `Discount_Pct` stored as text, some values with a trailing `%` |
| Implausible values | 46 rows with negative `Delivery_Distance_KM` |
| Missing target | 25 rows with no `Transaction_Amount` — dropped rather than imputed |
| Missing values | ~680 cells across 6 columns |
| Duplicate identifiers | 55 duplicate `OrderID`s with differing row data |

## Workflow

1. **EDA** — shape/dtype/missingness audit, distribution and skew analysis, category-wise spend comparison, correlation heatmap
2. **Cleaning** — standardized categorical text, fixed the malformed discount column, resolved implausible distance values, imputed remaining gaps
3. **Feature Engineering** — `spend_per_item`, `is_weekend`, `loyalty_segment` (ordinal-encoded), one-hot encoding for nominal categories
4. **Modeling** — 80/20 train-test split; Linear Regression and Random Forest Regressor
5. **Evaluation** — MAE, RMSE, R² on train and test sets; actual-vs-predicted diagnostics

**Note on leakage:** `spend_per_item` (`Transaction_Amount` / `Items_In_Cart`) is retained as an EDA insight feature but **excluded from the model inputs**, since it is mathematically derived from the target.

## Key Finding: Target Skew

`Transaction_Amount` is severely right-skewed (skewness ≈ 12.09, median 97, mean 155, max 5,349), driven by a small number of extreme high-value orders. This shapes model selection and explains the accuracy gap on the highest-value transactions.

## Results

| Model | Train R² | Test R² | MAE | RMSE |
|---|---|---|---|---|
| Linear Regression | 0.277 | 0.322 | 44.16 | 166.32 |
| Random Forest (tuned: `max_depth=11`, `min_samples_leaf=14`) | 0.378 | 0.316 | **39.19** | 167.07 |

R² and RMSE are effectively tied between the two models — both metrics are dominated by the same extreme outliers. The meaningful difference is **MAE**, where Random Forest is ~11% lower, reflecting more accurate predictions on typical, everyday transactions. Random Forest is recommended for deployment.

*(An earlier, untuned Random Forest scored Test R² = 0.055 against a Train R² of 0.881 — a clear overfit, resolved by constraining tree depth and leaf size.)*

## Tech Stack

- Python, Jupyter Notebook
- `pandas`, `numpy` — data cleaning and manipulation
- `matplotlib`, `seaborn` — visualization
- `scikit-learn` — modeling and evaluation
- `scipy` — statistical checks (skewness)

## Repository Structure

```
├── ecommerce_purchase.csv              # raw dataset
├── cleaned_ecommerce_purchase.csv      # cleaned, feature-engineered dataset
├── purchase_prediction.ipynb           # full analysis notebook
└── README.md
```

## Deliverables

- Cleaned dataset (CSV)
- Fully executed Jupyter notebook with EDA, cleaning, feature engineering, and modeling
- 6+ visualizations (distribution, category comparison, correlation heatmap, actual-vs-predicted, train/test comparison)
- Model comparison table with a written recommendation

## Author

**D. Gourisankar Mohanty**
GitHub: [Gyro-2k](https://github.com/Gyro-2k)
