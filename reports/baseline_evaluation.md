# ETAFlow: Baseline & Ridge Regression Evaluation Report

**Evaluation Date**: 2026-10-24  
**Split Protocol**: 80% Train (`31,171` rows), 20% Test (`7,793` rows), `random_state=42`  
**Feature Set**: 11 Tier 1 Booking-Time Features (Transformed into `17` encoded columns)  
**Target Variable**: `delivery_duration_min` (Continuous duration in minutes)  

---

## 1. Experimental Setup & Safeguards

1. **Strict Data Boundary**: Models were trained solely on `data/processed/train_features.csv` and evaluated solely on `data/processed/test_features.csv`. The raw CSV was never accessed.
2. **Untouched Test Split**: The test split was transformed exclusively using the preprocessor fitted on the training split (`models/preprocessor.joblib`).
3. **Artifact Integrity**:
   - `models/dummy_regressor.joblib` (Serialized baseline)
   - `models/ridge_regressor.joblib` (Serialized Ridge model)
   - Reload verification confirmed exact numerical equivalence across test predictions.

---

## 2. Test-Set Performance Benchmarking Table

| Model Architecture | Hyperparameters | MAE (minutes) | RMSE (minutes) | $R^2$ Score | MAPE (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **DummyRegressor (Floor Baseline)** | `strategy='mean'` | `7.4305` | `9.2257` | `-0.0001` | `33.33%` |
| **Ridge Regression** | `alpha=1.0, random_state=42` | `6.6154` | `8.1635` | `0.2169` | `29.54%` |
| **Improvement (Ridge vs. Baseline)** | | **`-0.8151 mins`** | **`-1.0622 mins`** | **`+0.2170`** | **`-3.78%`** |

*Note: For MAE and RMSE, negative differences represent improvement (lower error). For $R^2$, positive difference represents improvement (more variance explained).*

---

## 3. Analysis & Interpretation of Results

1. **Dummy Baseline Performance**:
   - The DummyRegressor always predicts the training target mean duration (`26.60` minutes).
   - On the untouched test set, this constant naive forecast yields an MAE of **`7.43` minutes**, an RMSE of **`9.23` minutes**, and an $R^2$ of **`-0.0001`**.
   - Note: The test $R^2$ for Dummy is slightly below zero (`-0.0001`) because the test split mean (`26.5090`) differs marginally from the training mean (`26.5996`).

2. **Ridge Regression Performance**:
   - Ridge regression achieves an MAE of **`6.62` minutes**, an RMSE of **`8.16` minutes**, and an $R^2$ of **`0.2169`**.
   - Compared to the baseline floor, Ridge reduces MAE by **`0.82` minutes** and RMSE by **`1.06` minutes**, while explaining approximately **`21.7%`** of delivery duration variance.

3. **Key Findings for Next Milestone**:
   - The linear model confirms that spatial distance, order timing, and market geography provide genuine predictive signal beyond a naive mean forecast.
   - However, a purely linear formulation has limitations capturing complex nonlinear interactions (e.g. traffic surges during peak dining hours or non-linear distance-duration relationships). Tree-based ensembles (Gradient Boosting, Random Forest) will be benchmarked in subsequent milestones.
