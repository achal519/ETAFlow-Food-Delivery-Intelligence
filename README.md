# ETAFlow: Smart Delivery Intelligence Platform

[![Python Version](https://img.shields.io/badge/python-3.11-blue.svg)](https://www.python.org/)
[![Project Status](https://img.shields.io/badge/status-production--ready-brightgreen.svg)](#)
[![Model](https://img.shields.io/badge/model-Random%20Forest%20Ensemble-purple.svg)](#)
[![Test Suite](https://img.shields.io/badge/tests-54%20passed-success.svg)](#)
[![License](https://img.shields.io/badge/license-Academic-green.svg)](#)

ETAFlow is a full-featured, ML-powered food delivery ETA and delay intelligence platform developed for a third-year Artificial Intelligence & Data Science engineering demonstration. It combines machine learning inference, interactive geospatial routing, deterministic operations simulation, and leakage-safe SLA governance into a unified platform.

---

## 1. System Architecture & Platform Overview

```text
                                 [ Customer Checkout / Booking Time ]
                                                  │
                                                  ▼
                                      [ POST /predict Endpoint ]
                                      (Validates 11 Tier 1 Features)
                                                  │
                                                  ▼
                                  [ Preprocessor Pipeline (17 cols) ]
                                  - Great-circle Haversine Distance
                                  - Temporal features (order_hour, DOW)
                                  - One-hot encoded categories
                                                  │
                                                  ▼
                                    [ Random Forest Regressor ]
                                    (100 Trees, Depth 16, Leaf 4)
                                                  │
                                                  ▼
                                  [ Booking-Time Predicted ETA (min) ]
                                                  │
                        ┌─────────────────────────┴─────────────────────────┐
                        ▼                                                   ▼
         [ Operations & Simulation Engine ]                  [ What-If Scenario Lab ]
         - Deterministic multi-order fleet                   - Parameter sensitivity exploration
         - Animated courier routing (OSRM)                   - Delta vs. baseline analysis
         - Lifecycle state progression                       - Export to CSV
         - Delay intelligence & SLA analysis
```

### Key Platform Modules:
1. **Operations Overview**: High-level telemetry of the demo fleet, active dispatches, completed orders, SLA compliance rate, and real-time operational activity log.
2. **Live Dispatch Simulation**: Deterministic multi-order execution engine with an explicit simulation clock (1x, 2x, 5x, 10x), courier movement along road routes, and stage decomposition.
3. **Live ETA Prediction**: Interactive booking-time workbench with coordinate sync via draggable map pins, order hour slider, and instant latency telemetry.
4. **Fleet & Orders Management**: Searchable and filterable data grid supporting order creation, status filtering, and map inspection.
5. **What-If Scenario Lab**: Empirical sensitivity analysis querying the live `POST /predict` API across spatial, temporal, and festival variations with CSV export.
6. **Delay Intelligence**: Pairwise evaluation of model predictions versus simulated outcomes with SLA classifications (Early, On-Time, Late) and root cause attribution.
7. **Model Intelligence**: 3-way held-out test benchmarks comparing DummyRegressor, Ridge Regression, and RandomForestRegressor, plus all 17 Gini impurity feature importances.
8. **Reports & Governance**: Strict data card audit, feature availability tiers, and anti-leakage rejection protocols.

---

## 2. Dataset Description & Safeguards

- **Data Source**: Historical public food delivery records (`data/raw/zomato_cleaned.csv`).
- **Split Protocol**: 80% Train (`31,171` rows) and 20% Held-Out Test (`7,793` rows) split using `random_state=42`.
- **Target Variable**: `delivery_duration_min` (total delivery duration in minutes from checkout to doorstep dropoff).
- **Leakage Prevention Safeguards**:
  * **Tier 1 (Allowed at Checkout)**: `Restaurant_latitude`, `Restaurant_longitude`, `Delivery_location_latitude`, `Delivery_location_longitude`, `distance_km` (Haversine), `order_hour`, `day_of_week`, `is_weekend`, `Type_of_order` (`Meal`, `Snack`, `Drinks`, `Buffet`), `Festival` (`No`, `Yes`), `City` (`Metropolitian`, `Urban`, `Semi-Urban`).
  * **Forbidden Post-Booking Attributes (HTTP 400 Rejected)**: `delivery_speed` (direct target leakage), `Time_taken (min)` (ground truth), `Time_Order_picked` (kitchen dispatch timestamp), `actual_pickup_time`, `actual_transit_time`, `Delivery_person_ID`, `Delivery_person_Ratings`, `multiple_deliveries`, `Type_of_vehicle`.

---

## 3. Model Benchmark Evaluation (Held-Out Test Set: 7,793 Rows)

Models were evaluated on the held-out test split using the exact preprocessor fitted exclusively on the training split:

| Model Architecture | Hyperparameters | MAE (Minutes) | RMSE (Minutes) | $R^2$ Score | MAPE (%) | Role |
| :--- | :--- | :---: | :---: | :---: | :---: | :--- |
| **DummyRegressor** | `strategy='mean'` | 7.4305 | 9.2257 | -0.0001 | 33.33% | Floor Baseline |
| **Ridge Regression** | `alpha=1.0, seed=42` | 6.6154 | 8.1635 | 0.2169 | 29.54% | Linear Surface |
| **RandomForestRegressor** | `n=100, depth=16, leaf=4` | **6.1406** | **7.6987** | **0.3036** | **27.41%** | ⭐ Production Champion |

### Feature Importance Hierarchy (17 Encoded Columns):
- `num__distance_km`: **30.62%** (primary predictor)
- `num__order_hour`: **26.07%** (dining peak temporal adjustment)
- `num__Delivery_location_longitude`: **5.55%**
- `cat__Festival_Yes`: **5.42%**
- `num__Delivery_location_latitude`: **5.13%**
- `cat__Festival_No`: **5.03%**
- Spatial coordinates and urban density constitute the remaining predictive splits.

*Methodological Note: Feature importance represents Gini impurity reduction across tree split nodes. It reflects empirical correlation in this historical sample, not causal proof.*

---

## 4. API Specification & Request/Response Contracts

### Health Check Endpoint
- **URL**: `GET /health`
- **Response**:
```json
{
  "status": "healthy",
  "model_identity": "random_forest_regressor",
  "artifacts_loaded": true
}
```

### Prediction Endpoint
- **URL**: `POST /predict`
- **Request Headers**: `Content-Type: application/json`
- **Request Payload**:
```json
{
  "restaurant_lat": 12.9352,
  "restaurant_lon": 77.6245,
  "customer_lat": 12.9550,
  "customer_lon": 77.6400,
  "order_hour": 20,
  "day_of_week": 4,
  "type_of_order": "Meal",
  "festival": "No",
  "city": "Metropolitian",
  "order_id": "KOR-DINNER-88"
}
```
- **Response Payload (HTTP 200)**:
```json
{
  "predicted_duration_minutes": 20.29,
  "model_identity": "random_forest_regressor",
  "distance_km": 2.7693,
  "prediction_stage": "booking_time",
  "order_id": "KOR-DINNER-88"
}
```

---

## 5. Simulation Design & Delay Intelligence

### Deterministic State Machine
The operations simulation models realistic multi-order lifecycle transitions:
$$\text{Order Received} \longrightarrow \text{Restaurant Preparing} \longrightarrow \text{Ready for Pickup} \longrightarrow \text{Out for Delivery} \longrightarrow \text{Delivered}$$
*(Orders in active states may also transition to `CANCELLED`.)*

### Transparent Operational Mechanics
- The Random Forest model predicts **total end-to-end delivery duration** at checkout.
- Individual stages (kitchen preparation, dispatch handoff, courier road travel) are **simulated operational mechanics** to demonstrate realistic platform behavior, not separate model sub-predictions.

### Delay Intelligence Protocol
For each completed simulated delivery, the system computes:
- $\text{Signed Error} = \text{Simulated Actual Duration} - \text{Predicted Duration}$
- $\text{Absolute Error} = |\text{Signed Error}|$
- **SLA Classification**:
  * **On-Time**: $|\text{Signed Error}| \le 3.0\text{ minutes}$
  * **Early**: $\text{Signed Error} < -3.0\text{ minutes}$
  * **Late**: $\text{Signed Error} > +3.0\text{ minutes}$
- **Simulated Delay Attribution**: Identifies simulated factors such as kitchen queue surge, corridor congestion, or expedited transit.
- **Mandatory Label**: `SIMULATED OPERATIONAL OUTCOMES — NOT REAL-WORLD MODEL VALIDATION`.

---

## 6. Interactive Geospatial Mapping & Routing

- **Map Engine**: Leaflet.js with CartoDB Dark Matter tile layer.
- **Routing Engine**: Free Open Source Routing Machine (OSRM) driving API (`https://router.project-osrm.org/`).
- **Resilient Fallback**: If the public routing service is unavailable, errors, or times out, the system automatically falls back to straight-line Haversine geodesic interpolation.
- **Attribution**: The map UI clearly displays the active routing provider (`OSRM Road Route` vs. `Geodesic Fallback`).
- **Zero GPS Fabrication**: Courier positions are animated along deterministic route segments for demonstration; no claims of real driver GPS tracking are made.

---

## 7. Installation, Launch & Testing

### Prerequisites
- Python 3.11+ in a local virtual environment (`.venv`).
- PowerShell terminal on Windows (or bash on Linux/macOS).

### Running Unit & Integration Tests (54 Tests)
Run the full test suite verifying preprocessors, models, API contracts, simulation state machines, and delay intelligence:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe -m unittest discover -s tests
```

### Launching the Backend API
Start the prediction service on port 8000:

```powershell
$env:PYTHONPATH = "src"
.\.venv\Scripts\python.exe pipelines\serve_api.py
```
- Serving at: `http://127.0.0.1:8000`
- Available endpoints: `GET /health`, `POST /predict`

### Launching the Frontend Platform
In a separate terminal, launch the web dashboard on port 8501:

```powershell
.\.venv\Scripts\python.exe -m http.server 8501 --directory app
```
- Open in your web browser: **`http://127.0.0.1:8501`**

---

## 8. Limitations & Future Scope

1. **Routing Dependency**: Road geometry queries depend on public OSRM servers. Self-hosting an internal OSRM instance with OpenStreetMap extracts would provide lower latency and offline capability.
2. **Post-Dispatch Re-estimation**: Current predictions occur strictly at checkout. A future milestone could introduce Tier 2 dispatch-time inference once a driver is assigned.
3. **Database Integration**: Demo orders are stored locally in the browser's `localStorage` and Python memory. Future iterations can integrate SQLite/PostgreSQL for multi-tenant persistence.
