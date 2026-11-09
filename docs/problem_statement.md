# ETAFlow: Problem Statement & Objectives

## 1. Executive Summary & Context
In on-demand food delivery platforms, accurate Estimated Time of Arrival (ETA) prediction is essential for platform efficiency, merchant kitchen scheduling, delivery partner (courier) dispatching, and consumer satisfaction. Underestimating delivery time leads to customer frustration and complaints; overestimating leads to lost orders and suboptimal courier allocation.

**ETAFlow** is an applied machine learning and MLOps project developed by third-year undergraduate students to design, benchmark, and deploy a robust, leakage-safe delivery duration prediction and delay intelligence system using real-world public food delivery data.

---

## 2. Formal Problem Formulation

### 2.1 Primary Target Variable
- **Proposed Target**: `delivery_duration_min` (continuous positive real value: total delivery duration in minutes).
- **Measurement Semantic**: The elapsed duration from order placement (or kitchen dispatch, subject to verified dataset timestamps) until successful customer drop-off.
- **Verification Condition**: This primary target is subject to empirical schema verification against public dataset ground truth before any model training begins.

### 2.2 Delay Classification & Service Level Agreements (SLAs)
- **Strict Methodological Rule**: No binary "late delivery" target will be fabricated, assumed, or created unless:
  1. An explicit ground-truth delay indicator is already present in the validated public dataset, or
  2. A rigorous, domain-defensible, and documented Service Level Agreement (SLA) threshold (e.g., standard promised delivery window) is empirically justified and agreed upon.
- Until verified, the problem remains strictly formulated as a **continuous regression task** on delivery duration.

---

## 3. Prediction Timing Semantics & Leakage Prevention

A fundamental failure mode in production ETA systems is **target leakage** and **temporal feature leakage** caused by using variables known only after an order has progressed through the delivery lifecycle.

### 3.1 Potential Prediction Stages
1. **Stage A: Order Booking / Placement Time**
   - Available inputs: Customer location, restaurant location, order timestamp (time of day, day of week), order size/type, weather/traffic conditions at booking time.
   - Unavailable inputs: Courier dispatch time, actual kitchen preparation time, actual courier pickup time, live courier drop-off transit time.
2. **Stage B: Courier Dispatch / Kitchen Pickup Time**
   - Available inputs: All Stage A inputs, plus actual preparation duration, assigned courier attributes (ratings, vehicle type), and courier departure timestamp.

> **Methodological Policy**: We make no premature assumptions that Stage A or Stage B is feasible until the candidate dataset's column definitions, timestamps, and data collection methodologies have been verified. Features generated after the prediction trigger event are strictly disallowed.

---

## 4. Key Constraints & Non-Goals

1. **No Synthetic Data**: The project must use genuine, publicly accessible delivery data. Synthetic data generation is strictly prohibited.
2. **No Data Snooping / Leakage**: All data transformations (imputation, scaling, encoding) must be fitted strictly on training folds and applied to test folds without lookahead bias.
3. **Reproducibility & Verification First**: Data schemas, licenses, missing values, and anomalies must be documented in a Data Card prior to model experimentation.
4. **Appropriate Undergraduate Scope**: Focus on rigorous foundational ML and clean software engineering before layering advanced infrastructure.
