# ETAFlow: Team Ownership & Contributions Matrix

This project is partitioned into three distinct, non-overlapping ownership tracks tailored for a three-member third-year undergraduate engineering team.

---

## 1. Ownership Tracks & Role Allocation

### Track 1: Data Engineering & Validation
- **Primary Owner**: Student 1
- **Focus Area**: Data sourcing, governance, schema validation, data pipeline, and quality assurance.
- **Core Responsibilities**:
  - Audit candidate public datasets (Kaggle DSCLP 2024, Parth-Malik Zomato dataset) for provenance, schema integrity, and licensing.
  - Maintain and update the [Data Card](data_card.md).
  - Implement data loading and validation routines in `src/etaflow/data/loader.py` and `src/etaflow/data/validation.py`.
  - Design data quality reports (missingness, distribution anomalies, coordinate sanity).
  - Own the DVC (Data Version Control) configuration and data storage strategy for `data/raw/` and `data/processed/`.

### Track 2: ML Experimentation & Modeling
- **Primary Owner**: Student 2
- **Focus Area**: Feature engineering, leakage prevention, baseline regressors, model benchmarking, and explainability.
- **Core Responsibilities**:
  - Implement leakage-safe feature transformations in `src/etaflow/features/pipeline.py` (spatial distance, cyclical timestamps, categorical encoding).
  - Ensure strict separation between training fold fitting and evaluation fold transforming.
  - Establish baseline regression models (mean/median, simple heuristic) in `src/etaflow/models/baseline.py`.
  - Benchmark candidate regressors (Linear, Ridge, Random Forest, Gradient Boosting).
  - Implement evaluation metrics (MAE, RMSE, MAPE) in `src/etaflow/evaluation/metrics.py`.
  - Produce interpretability and feature importance analysis (SHAP / permutation importance) in `reports/`.

### Track 3: MLOps, API & Deployment
- **Primary Owner**: Student 3
- **Focus Area**: Software architecture, configuration management, prediction API, interactive dashboard, and automated testing.
- **Core Responsibilities**:
  - Maintain centralized project configuration in `configs/config.yaml` and environment variables in `.env.example`.
  - Implement the prediction API service and Pydantic validation schemas in `src/etaflow/api/`.
  - Develop the interactive demonstration dashboard in `app/`.
  - Maintain reproducible pipeline entry points in `pipelines/`.
  - Establish automated test suites in `tests/unit/` and `tests/integration/`.
  - Prepare CI/CD workflows and deployment readiness.

---

## 2. Project Milestone Timeline

| Sprint / Phase | Deliverable | Lead Track | Supporting Tracks |
| :--- | :--- | :---: | :---: |
| **Phase 1: Foundation & Scaffold** | Scaffold, docs, configuration, git hygiene, baseline stubs | All | All |
| **Phase 2: Data Audit & Validation** | Public dataset verification, Data Card approval, quality checks | Track 1 | Track 2, Track 3 |
| **Phase 3: Features & Baselines** | Spatial & temporal features, baseline models, leakage-safe pipeline | Track 2 | Track 1, Track 3 |
| **Phase 4: Model Benchmarking** | Model comparisons, cross-validation, metric benchmarking | Track 2 | Track 1, Track 3 |
| **Phase 5: API & Dashboard** | FastAPI inference endpoint, Streamlit/dashboard interface | Track 3 | Track 2 |
| **Phase 6: Explainability & Reports** | Error analysis, SHAP reports, integration test suite | Track 2 & 3 | Track 1 |
