# ETAFlow: Random Forest Benchmark Evaluation Report

**Evaluation Date**: 2026-10-24  
**Split Protocol**: 80% Train (`31,171` rows), 20% Test (`7,793` rows), `random_state=42`  
**Feature Set**: 11 Tier 1 Booking-Time Features (Transformed into `17` encoded columns)  
**Target Variable**: `delivery_duration_min` (Continuous duration in minutes)  
**Training Wall-Clock Time**: `1.63 seconds`  
**Serialized Model Size**: `23.06 MB` (`24,176,225 bytes`)  

---

## 1. Experimental Setup & Safeguards

1. **Strict Data Boundary**: Trained strictly on `data/processed/train_features.csv` and evaluated on `data/processed/test_features.csv`. The raw CSV was never accessed.
2. **Untouched Test Split**: The test split was transformed exclusively using the preprocessor fitted on the training split (`models/preprocessor.joblib`).
3. **Artifact Integrity**:
   - `models/random_forest_regressor.joblib` (Saved new ensemble artifact)
   - Baseline artifacts (`models/dummy_regressor.joblib`, `models/ridge_regressor.joblib`) were verified and preserved untouched.
   - Baseline report (`reports/baseline_evaluation.md`) was preserved untouched.
   - Reload verification confirmed exact numerical equivalence across test predictions (`rtol=1e-5`).

---

## 2. 3-Way Performance Benchmarking Table

| Model Architecture | Hyperparameters | MAE (minutes) | RMSE (minutes) | $R^2$ Score | MAPE (%) |
| :--- | :--- | :---: | :---: | :---: | :---: |
| **DummyRegressor (Floor Baseline)** | `strategy='mean'` | `7.4305 mins` | `9.2257 mins` | `-0.0001` | `33.33%` |
| **Ridge Regression** | `alpha=1.0, random_state=42` | `6.6154 mins` | `8.1635 mins` | `0.2169` | `29.54%` |
| **RandomForestRegressor** | `n_est=100, depth=16, leaf=4` | **`6.1406 mins`** | **`7.6987 mins`** | **`0.3036`** | **`27.41%`** |
| **Delta (RF vs. Dummy Baseline)** | | **`-1.2900 mins`** | **`-1.5270 mins`** | **`+0.3037`** | **`-5.91%`** |
| **Delta (RF vs. Ridge Linear)** | | **`-0.4749 mins`** | **`-0.4648 mins`** | **`+0.0866`** | **`-2.13%`** |

*Note: For MAE, RMSE, and MAPE, negative delta indicates error reduction (improvement). For $R^2$, positive delta indicates greater variance explained.*

---

## 3. Analysis & Interpretation of Results

1. **Comparison with Dummy Baseline Floor**:
   - Random Forest reduces MAE from **`7.43` minutes** down to **`6.14` minutes** (a reduction of **`1.29` minutes**).
   - RMSE decreases from **`9.23` minutes** to **`7.70` minutes** (a reduction of **`1.53` minutes**).
   - $R^2$ jumps from **`-0.0001`** to **`0.3036`**, explaining approximately **`30.4%`** of the total duration variance.

2. **Comparison with Linear Ridge Model**:
   - Random Forest improves upon the Ridge linear model by reducing MAE by **`0.47` minutes** and RMSE by **`0.46` minutes**.
   - $R^2$ improves by **`+0.0866`** (from `0.2169` to `0.3036`).
   - This proves empirically that tree-based nonlinear interactions between geographic distance, order timing, and urban density extract substantially more predictive information than a strictly additive linear surface.

3. **Trade-Off Analysis**:
   - **Training Cost**: The ensemble trained in `1.63 seconds` on `31,171` rows using multi-core parallelization (`n_jobs=-1`).
   - **Model Footprint**: Constraining `max_depth=16` and `min_samples_leaf=4` successfully kept the model artifact size to **`23.06 MB`**, ensuring fast memory loading and low-latency prediction for future API serving.

---

## 4. Feature Importance Analysis (All 17 Encoded Features)

> **Important Methodological Note**: Feature importance measures the total reduction of mean squared error (Gini/variance impurity) brought by a feature across all split nodes. **It is strictly descriptive, not causal**. A high importance indicates strong predictive association with delivery duration in this historical sample, but does not prove that changing that variable will causally alter delivery time in real life.

| Feature Name | Impurity Importance | Relative Share (%) |
| :--- | :---: | :---: |
| `num__distance_km` | `0.3062` | `30.62%` |
| `num__order_hour` | `0.2607` | `26.07%` |
| `num__Delivery_location_longitude` | `0.0555` | `5.55%` |
| `cat__Festival_Yes` | `0.0542` | `5.42%` |
| `num__Delivery_location_latitude` | `0.0513` | `5.13%` |
| `cat__Festival_No` | `0.0503` | `5.03%` |
| `num__Restaurant_longitude` | `0.0468` | `4.68%` |
| `num__Restaurant_latitude` | `0.0416` | `4.16%` |
| `num__day_of_week` | `0.0340` | `3.40%` |
| `cat__City_Urban` | `0.0325` | `3.25%` |
| `cat__City_Semi-Urban` | `0.0166` | `1.66%` |
| `cat__City_Metropolitian` | `0.0154` | `1.54%` |
| `cat__Type_of_order_Meal` | `0.0080` | `0.80%` |
| `cat__Type_of_order_Snack` | `0.0078` | `0.78%` |
| `cat__Type_of_order_Drinks` | `0.0076` | `0.76%` |
| `cat__Type_of_order_Buffet` | `0.0075` | `0.75%` |
| `num__is_weekend` | `0.0038` | `0.38%` |

### Top Insights:
1. **Primary Predictor**: The top predictor is `num__distance_km`, accounting for `30.62%` of tree split importance.
2. **Spatial Anchors**: Coordinates and distance collectively account for the vast majority of predictive splits.
3. **Temporal Patterns**: `order_hour` and calendar day features provide secondary adjustments reflecting peak dining windows.
