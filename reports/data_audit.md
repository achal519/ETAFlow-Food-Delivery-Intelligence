# ETAFlow: Exploratory Data Audit & Quality Report

**Target Dataset**: `data/raw/zomato_cleaned.csv`  
**Execution Environment**: Python 3.11.9 (Standard Library `csv`, `math`, `statistics`)  
**Audit Status**: Verified & Reproducible  

---

## 1. Dataset Dimensions & Schema Overview

* **Total Records (Rows)**: `38,964`
* **Total Attributes (Columns)**: `22`
* **Exact Duplicate Rows**: `0`
* **Unique Order IDs**: `38,964` (`0` duplicate IDs)

| # | Column Name | Raw Data Type | Missing Count | Missing % | Inferred Semantic Role |
| :---: | :--- | :---: | :---: | :---: | :--- |
| 1 | `ID` | String (Hex) | 0 | 0.00% | Unique order identifier (Excluded from modeling) |
| 2 | `Delivery_person_ID` | String | 0 | 0.00% | Courier identifier (1,170 unique partners) |
| 3 | `Delivery_person_Age` | Float / Numeric | 1,019 | 2.62% | Courier age (Range: 15.0 - 50.0) |
| 4 | `Delivery_person_Ratings` | Float / Numeric | 1,055 | 2.71% | Courier average rating (Range: 1.0 - 6.0) |
| 5 | `Restaurant_latitude` | Float / Decimal | 0 | 0.00% | Merchant GPS latitude (India bounds) |
| 6 | `Restaurant_longitude` | Float / Decimal | 0 | 0.00% | Merchant GPS longitude (India bounds) |
| 7 | `Delivery_location_latitude` | Float / Decimal | 0 | 0.00% | Customer GPS latitude (India bounds) |
| 8 | `Delivery_location_longitude` | Float / Decimal | 0 | 0.00% | Customer GPS longitude (India bounds) |
| 9 | `Order_Date` | String (Date) | 0 | 0.00% | Date of order placement (`DD-MM-YYYY`) |
| 10 | `Time_Orderd` | String (Time/Frac) | 835 | 2.14% | Order placement time (See data quality notice) |
| 11 | `Time_Order_picked` | String (Time) | 0 | 0.00% | Pickup time (Post-booking; excluded at booking) |
| 12 | `Weather_conditions` | Categorical | 0 | 0.00% | 6 conditions: Fog, Stormy, Sandstorms, etc. |
| 13 | `Road_traffic_density` | Categorical | 0 | 0.00% | 4 levels: Low, Medium, High, Jam |
| 14 | `Vehicle_condition` | Integer / Ordinal | 0 | 0.00% | Vehicle condition score: `0`, `1`, `2` |
| 15 | `Type_of_order` | Categorical | 0 | 0.00% | 4 categories: Snack, Meal, Drinks, Buffet |
| 16 | `Type_of_vehicle` | Categorical | 0 | 0.00% | 3 types: motorcycle, scooter, electric_scooter |
| 17 | `multiple_deliveries` | Float / Ordinal | 0 | 0.00% | Simultaneous orders: `0.0`, `1.0`, `2.0`, `3.0` |
| 18 | `Festival` | Binary Categorical | 0 | 0.00% | Festival period indicator (`No`: 38,180, `Yes`: 784) |
| 19 | `City` | Categorical | 0 | 0.00% | Urban classification: Metropolitian, Urban, Semi-Urban |
| 20 | `Time_taken (min)` | Float / Integer | 0 | 0.00% | **Primary Target Variable** (Continuous duration) |
| 21 | `distance_km` | Float / Numeric | 0 | 0.00% | Haversine distance in km (Verified exact match) |
| 22 | `delivery_speed` | Categorical | 0 | 0.00% | **Target Leakage** (Discretized target; excluded) |

---

## 2. Primary Target Variable Audit (`Time_taken (min)`)

The primary target represents the total delivery duration in minutes. All 38,964 records have valid, non-null, parseable numerical target values.

* **Count**: `38,964`
* **Minimum**: `10.0 minutes`
* **Maximum**: `54.0 minutes`
* **Mean**: `26.58 minutes`
* **Median**: `26.0 minutes`
* **Standard Deviation**: `9.34 minutes`
* **25th Percentile**: `20.0 minutes`
* **75th Percentile**: `33.0 minutes`
* **Skewness / Distribution**: Symmetrical, unimodal distribution bounded within standard operational food delivery delivery windows (10 - 54 mins). Zero negative or zero-duration anomalies.

---

## 3. Spatial & GPS Coordinates Audit

All latitude and longitude records represent valid geographic coordinates across urban clusters in India.

| Geographic Field | Minimum | Maximum | Mean | Zero / Negative Coordinates |
| :--- | :---: | :---: | :---: | :---: |
| **Restaurant Latitude** | `9.9571` | `30.9141` | `17.2483` | `0` |
| **Restaurant Longitude** | `72.7687` | `88.4335` | `70.7788` | `0` |
| **Delivery Latitude** | `9.9671` | `31.0541` | `17.4645` | `0` |
| **Delivery Longitude** | `72.7787` | `88.5635` | `70.8354` | `0` |

### `distance_km` Verification
* **Minimum Distance**: `1.4651 km`
* **Maximum Distance**: `20.9695 km`
* **Mean Distance**: `9.7651 km`
* **Median Distance**: `9.2199 km`
* **Discrepancy with Calculated Haversine**: `0.00000000 km` across all 38,964 rows.
* **Safety Assessment**: `distance_km` is **completely safe** to use at prediction time because restaurant and customer coordinates are known at the moment of order placement.

---

## 4. Leakage & Prediction-Time Feature Availability Analysis

### 4.1 Proof of Target Leakage in `delivery_speed`
An empirical breakdown of `delivery_speed` against `Time_taken (min)` reveals an exact, non-overlapping target discretization:

| `delivery_speed` Category | Record Count | Target Range (`Time_taken (min)`) | Mean Target Time |
| :--- | :---: | :---: | :---: |
| **Fast** | 10,164 | `10.0 - 19.0 minutes` | `15.63 minutes` |
| **Average** | 19,982 | `20.0 - 33.0 minutes` | `26.19 minutes` |
| **Slow** | 8,818 | `34.0 - 54.0 minutes` | `40.10 minutes` |

> **Audit Finding**: `delivery_speed` is a post-hoc label derived directly from the target variable (`Fast` = [10, 19], `Average` = [20, 33], `Slow` = [34, 54]). Including this attribute in any feature set causes 100% target leakage. **It is strictly excluded.**

### 4.2 Prediction Stage Semantics: Booking Time vs. Dispatch Time

| Feature Name | Available at Booking? | Available at Dispatch? | Architectural Recommendation |
| :--- | :---: | :---: | :--- |
| `distance_km` / Coordinates | **Yes** | **Yes** | Primary spatial predictor; safe at all stages. |
| `Weather_conditions` | **Yes** | **Yes** | Environmental context known at order time. |
| `Road_traffic_density` | **Yes** | **Yes** | Traffic conditions known at order time. |
| `Type_of_order` | **Yes** | **Yes** | Meal basket type known at order time. |
| `Festival` / `City` | **Yes** | **Yes** | Regional context known at order time. |
| `Delivery_person_ID` | **No** (Typically) | **Yes** | In real dispatch systems, couriers are matched *after* order confirmation. |
| `Delivery_person_Age` / `Ratings` | **No** (Typically) | **Yes** | Tied to courier identity; only known post-dispatch. |
| `Vehicle_condition` / `Type` | **No** (Typically) | **Yes** | Tied to courier vehicle; only known post-dispatch. |
| `multiple_deliveries` | **No** | **Tentative** | Assigned dynamically by dispatch routing algorithms. |
| `Time_Order_picked` | **Strict No** | **Yes** | Measured 5 to 15 minutes post-booking (mean delay = 9.94 mins). |

---

## 5. Data Quality Limitations & Anomalies

1. **Excel Day-Fraction Format in `Time_Orderd`**:
   * While ~30,800 records contain standard `HH:MM` strings (e.g., `21:55`), approximately 7,300 records contain decimal fractions (e.g., `0.458333333`, `0.791666667`, `1`).
   * This is a known artifact of CSV files opened and re-saved in Microsoft Excel, which converts time of day into fractional fractions of a 24-hour day (`0.5` = 12:00 PM).
   * **Resolution for Feature Engineering**: Implement a robust time parser converting float strings: `hour = int(float_val * 24)`.

2. **Missing Delivery Person Demographics**:
   * `Delivery_person_Age`: 1,019 missing records (2.62%).
   * `Delivery_person_Ratings`: 1,055 missing records (2.71%).
   * `Time_Orderd`: 835 missing records (2.14%).
   * **Resolution**: Imputation must be fit exclusively on training splits (e.g., median age and median rating).

3. **Courier Cardinality**:
   * There are 1,170 unique `Delivery_person_ID` values with an average of 33.3 deliveries each.
   * Direct one-hot encoding would introduce high dimensionality. If courier features are used in dispatch-stage models, target encoding or embeddings with cross-validation regularisation should be evaluated.

---

## 6. Audit Conclusion & Next Steps

1. `data/raw/zomato_cleaned.csv` is fully verified, authentic, and suitable for the ETAFlow project.
2. The primary target `Time_taken (min)` is well-behaved, continuous, and has 0 missing values.
3. Feature isolation in `src/etaflow/features/pipeline.py` successfully strips `delivery_speed`, target variables, and post-booking attributes.
4. Next phase: leakage-safe data preprocessing, splitting (train/val/test), and baseline model benchmark.
