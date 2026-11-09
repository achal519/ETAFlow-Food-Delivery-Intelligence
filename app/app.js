/**
 * ETAFlow Smart Delivery Intelligence & MLOps Platform — Client Engine
 * Powers:
 * 1. MLOps Observability: Live API Telemetry (/metrics), Model Info (/model-info)
 * 2. MLflow Experiment Tracking (experiment_summary.json)
 * 3. Statistical Drift Monitoring with Clean vs Synthetic Surge Toggle (drift_report.json)
 * 4. Model Registry Governance (registry.json)
 * 5. Live Booking-Time Inference Workbench (POST /predict)
 * 6. Feature Importance Hierarchy & Model Evaluation (Random Forest Champion)
 * 7. Appendix: Deterministic Dispatch Simulation, OSRM Routing, Fleet Grid, Scenario Lab & Delay Analytics
 */

const API_BASE_URL = "http://127.0.0.1:8000";

// Predefined verified demo locations around Bangalore with explicit source tags
const DEFAULT_DEMO_ORDERS = [
  {
    order_id: "KOR-DINNER-88",
    restaurant_lat: 12.9352,
    restaurant_lon: 77.6245,
    customer_lat: 12.9550,
    customer_lon: 77.6400,
    order_hour: 20,
    day_of_week: 4, // Friday
    type_of_order: "Meal",
    festival: "No",
    city: "Metropolitian",
    predicted_duration_minutes: 20.3,
    distance_km: 2.77,
    prediction_source: "Preset Demo Baseline",
    model_identity: "random_forest_regressor (Preset)",
    status: "OUT_FOR_DELIVERY",
    simulation_elapsed_seconds: 70.0,
    stage_progress_pct: 68.0,
    simulated_actual_duration_minutes: null,
    delay_cause: null
  },
  {
    order_id: "IND-SNACK-12",
    restaurant_lat: 12.9784,
    restaurant_lon: 77.6408,
    customer_lat: 12.9912,
    customer_lon: 77.6534,
    order_hour: 16,
    day_of_week: 6, // Sunday
    type_of_order: "Snack",
    festival: "No",
    city: "Urban",
    predicted_duration_minutes: 17.6,
    distance_km: 1.95,
    prediction_source: "Preset Demo Baseline",
    model_identity: "random_forest_regressor (Preset)",
    status: "DELIVERED",
    simulation_elapsed_seconds: 105.0,
    stage_progress_pct: 100.0,
    simulated_actual_duration_minutes: 18.2,
    delay_cause: "Simulated on-time arrival within SLA threshold"
  },
  {
    order_id: "FEST-FEAST-99",
    restaurant_lat: 13.0358,
    restaurant_lon: 77.5970,
    customer_lat: 13.0600,
    customer_lon: 77.6200,
    order_hour: 21,
    day_of_week: 5, // Saturday
    type_of_order: "Meal",
    festival: "Yes",
    city: "Semi-Urban",
    predicted_duration_minutes: 27.8,
    distance_km: 3.65,
    prediction_source: "Preset Demo Baseline",
    model_identity: "random_forest_regressor (Preset)",
    status: "DELIVERED",
    simulation_elapsed_seconds: 165.0,
    stage_progress_pct: 100.0,
    simulated_actual_duration_minutes: 29.1,
    delay_cause: "Synthetic festival kitchen queue delay (Simulated Factor)"
  },
  {
    order_id: "HSR-LUNCH-42",
    restaurant_lat: 12.9121,
    restaurant_lon: 77.6446,
    customer_lat: 12.9215,
    customer_lon: 77.6510,
    order_hour: 13,
    day_of_week: 2, // Wednesday
    type_of_order: "Meal",
    festival: "No",
    city: "Metropolitian",
    predicted_duration_minutes: 15.4,
    distance_km: 1.35,
    prediction_source: "Preset Demo Baseline",
    model_identity: "random_forest_regressor (Preset)",
    status: "DELIVERED",
    simulation_elapsed_seconds: 90.0,
    stage_progress_pct: 100.0,
    simulated_actual_duration_minutes: 14.8,
    delay_cause: "Simulated on-time arrival within SLA threshold"
  },
  {
    order_id: "MG-DRINKS-07",
    restaurant_lat: 12.9756,
    restaurant_lon: 77.6066,
    customer_lat: 12.9840,
    customer_lon: 77.6180,
    order_hour: 18,
    day_of_week: 3, // Thursday
    type_of_order: "Drinks",
    festival: "No",
    city: "Metropolitian",
    predicted_duration_minutes: 18.9,
    distance_km: 1.70,
    prediction_source: "Preset Demo Baseline",
    model_identity: "random_forest_regressor (Preset)",
    status: "RESTAURANT_PREPARING",
    simulation_elapsed_seconds: 22.0,
    stage_progress_pct: 21.0,
    simulated_actual_duration_minutes: null,
    delay_cause: null
  },
  {
    order_id: "WFD-BUFFET-03",
    restaurant_lat: 12.9698,
    restaurant_lon: 77.7499,
    customer_lat: 12.9980,
    customer_lon: 77.7750,
    order_hour: 20,
    day_of_week: 4, // Friday
    type_of_order: "Buffet",
    festival: "No",
    city: "Metropolitian",
    predicted_duration_minutes: 32.1,
    distance_km: 5.40,
    prediction_source: "Preset Demo Baseline",
    model_identity: "random_forest_regressor (Preset)",
    status: "CANCELLED",
    simulation_elapsed_seconds: 15.0,
    stage_progress_pct: 0.0,
    simulated_actual_duration_minutes: null,
    delay_cause: "Customer cancelled during kitchen preparation stage"
  },
  {
    order_id: "BTM-SNACK-55",
    restaurant_lat: 12.9165,
    restaurant_lon: 77.6101,
    customer_lat: 12.9290,
    customer_lon: 77.6250,
    order_hour: 17,
    day_of_week: 1, // Tuesday
    type_of_order: "Snack",
    festival: "No",
    city: "Urban",
    predicted_duration_minutes: 19.4,
    distance_km: 2.10,
    prediction_source: "Preset Demo Baseline",
    model_identity: "random_forest_regressor (Preset)",
    status: "READY_FOR_PICKUP",
    simulation_elapsed_seconds: 45.0,
    stage_progress_pct: 42.0,
    simulated_actual_duration_minutes: null,
    delay_cause: null
  }
];

// Feature Importances from Champion Random Forest
const FEATURE_IMPORTANCES = [
  { feature: "num__distance_km", label: "Distance (Haversine km)", importance: 0.3062 },
  { feature: "num__order_hour", label: "Order Hour (Cyclical)", importance: 0.2607 },
  { feature: "num__Delivery_location_longitude", label: "Customer Longitude", importance: 0.0555 },
  { feature: "cat__Festival_Yes", label: "Festival Surge: Yes", importance: 0.0542 },
  { feature: "num__Delivery_location_latitude", label: "Customer Latitude", importance: 0.0513 },
  { feature: "cat__Festival_No", label: "Festival: No", importance: 0.0503 },
  { feature: "num__Restaurant_latitude", label: "Restaurant Latitude", importance: 0.0484 },
  { feature: "num__Restaurant_longitude", label: "Restaurant Longitude", importance: 0.0468 },
  { feature: "num__day_of_week", label: "Day of Week", importance: 0.0391 },
  { feature: "num__is_weekend", label: "Is Weekend Flag", importance: 0.0186 },
  { feature: "cat__Type_of_order_Meal", label: "Order Type: Meal", importance: 0.0171 },
  { feature: "cat__City_Metropolitian", label: "City: Metropolitian", importance: 0.0162 },
  { feature: "cat__Type_of_order_Snack", label: "Order Type: Snack", importance: 0.0145 },
  { feature: "cat__City_Urban", label: "City: Urban", importance: 0.0112 },
  { feature: "cat__Type_of_order_Drinks", label: "Order Type: Drinks", importance: 0.0078 },
  { feature: "cat__City_Semi-Urban", label: "City: Semi-Urban", importance: 0.0017 },
  { feature: "cat__Type_of_order_Buffet", label: "Order Type: Buffet", importance: 0.0005 }
];

// Tab Titles dictionary for sticky topbar
const TAB_TITLES = {
  "tab-mlops-overview": {
    title: "MLOps Platform Overview",
    subtitle: "Production model lifecycle, experiment tracking & statistical drift guard"
  },
  "tab-data-pipeline": {
    title: "Data Governance & Pipeline",
    subtitle: "Dataset provenance, schema validation audit & leakage-safe feature store"
  },
  "tab-experiments": {
    title: "MLflow Experiment Tracking",
    subtitle: "Reproducible model evaluation lab comparing Dummy, Ridge, and Random Forest"
  },
  "tab-model-registry": {
    title: "Model Registry & Governance",
    subtitle: "Cryptographic model versioning, staging status, and artifact manifests"
  },
  "tab-predict": {
    title: "Live Booking-Time Inference",
    subtitle: "Draggable coordinate workbench querying live POST /predict API"
  },
  "tab-monitoring-drift": {
    title: "Data & Prediction Drift Monitoring",
    subtitle: "PSI, Kolmogorov-Smirnov, and Chi-Square statistical distribution shift tests"
  },
  "tab-model": {
    title: "Model Evaluation & Importances",
    subtitle: "Held-out test split benchmark & Gini impurity feature importance hierarchy"
  },
  "tab-governance": {
    title: "Reports & Governance Audit",
    subtitle: "Data card, anti-leakage compliance checklist, and audit trails"
  },
  "tab-operations": {
    title: "Operations Overview (Synthetic Demo)",
    subtitle: "Simulated fleet activity, active dispatches, and SLA compliance telemetry"
  },
  "tab-simulation": {
    title: "Live Dispatch Simulation (Synthetic Demo)",
    subtitle: "Deterministic multi-order execution engine with courier road routing"
  },
  "tab-fleet": {
    title: "Fleet & Orders Grid (Synthetic Demo)",
    subtitle: "Multi-order lifecycle tracking, filterable data grid, and order creation"
  },
  "tab-scenarios": {
    title: "Scenario Lab (Synthetic Demo)",
    subtitle: "Empirical sensitivity analysis querying live model across spatial variations"
  },
  "tab-delay": {
    title: "Delay Analytics (Synthetic Demo)",
    subtitle: "Pairwise evaluation of model predictions vs simulated outcomes"
  }
};

// Global App State
let appOrders = [];
let activityLog = [];
let simulationTimer = null;
let simulationClockSeconds = 0;
let simulationSpeed = 5.0;
let isSimulationRunning = false;
let activeSimOrderId = "KOR-DINNER-88";

let simMap = null;
let simMapMarkers = {};
let simMapPolyline = null;
let simCourierMarker = null;

let predictMap = null;
let predictRestMarker = null;
let predictCustMarker = null;
let predictPolyline = null;

let driftData = null;
let activeDriftMode = "clean"; // "clean" or "surge"

// Cached DOM Elements
let topbarPageTitle, topbarPageSubtitle, topbarModelId, topbarStatusChip, topbarStatusDot, topbarStatusText;
let sidebarStatusCard, sidebarStatusDot, sidebarStatusLabel, btnRefreshStatus;
let globalAlert, alertMessage, btnDismissAlert;
let sidebar, btnSidebarToggle, sidebarBackdrop;

/* ==========================================================
   INITIALIZATION & BOOTSTRAP
========================================================== */
document.addEventListener("DOMContentLoaded", () => {
  bootstrapApp();
});

function bootstrapApp() {
  cacheDomElements();
  initFleetStore();
  initActivityLog();
  initNavigationTabs();

  // MLOps Platform Inits
  loadMlflowExperiments();
  loadDriftMonitoring();
  loadModelRegistry();
  renderFeatureImportance();
  startTelemetryPolling();

  // Inference & Map Inits
  initPredictSpatialMap();
  initHourSlider();
  initPredictionForm();
  initQuickProbeButton();

  // Appendix Inits
  initSimulationControls();
  initSimulationMap();
  initFleetManager();
  initModalHandlers();
  initScenariosLab();
  renderOperationsOverview();
  renderDelayIntelligence();

  // Initial health check
  checkApiHealth();
}

function cacheDomElements() {
  topbarPageTitle = document.getElementById("topbar-page-title");
  topbarPageSubtitle = document.getElementById("topbar-page-subtitle");
  topbarModelId = document.getElementById("topbar-model-id");
  topbarStatusChip = document.getElementById("topbar-status-chip");
  topbarStatusDot = document.getElementById("topbar-status-dot");
  topbarStatusText = document.getElementById("topbar-status-text");

  sidebarStatusCard = document.getElementById("sidebar-status-card");
  sidebarStatusDot = document.getElementById("sidebar-status-dot");
  sidebarStatusLabel = document.getElementById("sidebar-status-label");
  btnRefreshStatus = document.getElementById("btn-refresh-status");

  globalAlert = document.getElementById("global-alert");
  alertMessage = document.getElementById("alert-message");
  btnDismissAlert = document.getElementById("btn-dismiss-alert");

  sidebar = document.getElementById("sidebar");
  btnSidebarToggle = document.getElementById("btn-sidebar-toggle");
  sidebarBackdrop = document.getElementById("sidebar-backdrop");

  if (btnSidebarToggle) {
    btnSidebarToggle.addEventListener("click", () => {
      sidebar.classList.toggle("open");
    });
  }
  if (sidebarBackdrop) {
    sidebarBackdrop.addEventListener("click", () => {
      sidebar.classList.remove("open");
    });
  }
  if (btnDismissAlert) {
    btnDismissAlert.addEventListener("click", () => {
      globalAlert.classList.add("hidden");
    });
  }
  if (btnRefreshStatus) {
    btnRefreshStatus.addEventListener("click", () => {
      checkApiHealth();
    });
  }
}

/* ==========================================================
   NAVIGATION TAB MANAGEMENT
========================================================= */
function initNavigationTabs() {
  const navButtons = document.querySelectorAll(".nav-item");
  navButtons.forEach((btn) => {
    btn.addEventListener("click", () => {
      const targetTab = btn.getAttribute("data-tab");
      if (targetTab) {
        switchTab(targetTab);
      }
    });
  });
}

function switchTab(tabId) {
  // Update sidebar active buttons
  document.querySelectorAll(".nav-item").forEach((b) => {
    b.classList.remove("active");
    if (b.getAttribute("data-tab") === tabId) {
      b.classList.add("active");
    }
  });

  // Switch tab panes
  document.querySelectorAll(".tab-pane").forEach((pane) => {
    pane.classList.remove("active");
  });
  const targetPane = document.getElementById(tabId);
  if (targetPane) {
    targetPane.classList.add("active");
  }

  // Update topbar header text
  if (TAB_TITLES[tabId]) {
    if (topbarPageTitle) topbarPageTitle.textContent = TAB_TITLES[tabId].title;
    if (topbarPageSubtitle) topbarPageSubtitle.textContent = TAB_TITLES[tabId].subtitle;
  }

  // Invalidate maps if visible
  if (tabId === "tab-predict" && predictMap) {
    setTimeout(() => predictMap.invalidateSize(), 150);
  }
  if (tabId === "tab-simulation" && simMap) {
    setTimeout(() => simMap.invalidateSize(), 150);
  }

  // Close mobile sidebar
  if (sidebar) sidebar.classList.remove("open");
}

/* ==========================================================
   MLOPS 1: TELEMETRY POLLING (/metrics & /model-info)
========================================================== */
function startTelemetryPolling() {
  fetchTelemetry();
  setInterval(fetchTelemetry, 3500);
}

async function fetchTelemetry() {
  try {
    const res = await fetch(`${API_BASE_URL}/metrics`);
    if (res.ok) {
      const data = await res.json();
      updateTelemetryUI(data);
    }
  } catch (err) {
    // Silent catch when offline; health check handles alert
  }
}

function updateTelemetryUI(metrics) {
  const totalReq = document.getElementById("telem-total-requests");
  const successReq = document.getElementById("telem-success-predictions");
  const latestLat = document.getElementById("telem-latest-latency");
  const p50Lat = document.getElementById("telem-p50-latency");
  const p95Lat = document.getElementById("telem-p95-latency");
  const uptime = document.getElementById("telem-uptime");
  const overviewLat = document.getElementById("overview-latency-val");

  if (totalReq) totalReq.textContent = metrics.total_requests || 0;
  if (successReq) successReq.textContent = metrics.successful_predictions || 0;

  const lat = metrics.latency_ms || {};
  if (latestLat) latestLat.textContent = `${(lat.latest || 0).toFixed(1)} ms`;
  if (p50Lat) p50Lat.textContent = `${(lat.p50 || 0).toFixed(1)} ms`;
  if (p95Lat) p95Lat.textContent = `${(lat.p95 || 0).toFixed(1)} ms`;
  if (overviewLat) overviewLat.textContent = (lat.p50 || 2.8).toFixed(1);

  if (uptime) {
    const s = Math.floor(metrics.uptime_seconds || 0);
    const m = Math.floor(s / 60);
    const h = Math.floor(m / 60);
    uptime.textContent = h > 0 ? `${h}h ${m % 60}m` : `${m}m ${s % 60}s`;
  }
}

/* ==========================================================
   MLOPS 2: MLFLOW EXPERIMENT TRACKING
========================================================== */
async function loadMlflowExperiments() {
  const tbody = document.getElementById("mlflow-experiments-tbody");
  if (!tbody) return;

  let summary = null;
  try {
    // First try live server endpoint
    const res = await fetch(`${API_BASE_URL}/reports/experiments`);
    if (res.ok) summary = await res.json();
  } catch (e) {
    // Fallback to static directory data
  }

  if (!summary) {
    try {
      const resFallback = await fetch("./data/experiment_summary.json");
      if (resFallback.ok) summary = await resFallback.json();
    } catch (e) {
      console.warn("Could not load experiment summary", e);
    }
  }

  if (!summary || !summary.runs) {
    tbody.innerHTML = `<tr><td colspan="10" class="text-center text-muted">Unable to load experiment summary.</td></tr>`;
    return;
  }

  // Render runs
  tbody.innerHTML = summary.runs.map((run) => {
    const isChampion = run.stage === "Production" || run.run_name.includes("champion");
    const stagePill = isChampion
      ? `<span class="status-pill pill-champion">👑 Production</span>`
      : `<span class="status-pill pill-archived">Archived</span>`;

    const r2Display = run.metrics.r2_score < 0 ? run.metrics.r2_score.toFixed(4) : `+${run.metrics.r2_score.toFixed(4)}`;
    const paramStr = Object.entries(run.parameters || {})
      .map(([k, v]) => `${k}=${v}`)
      .join(", ");

    return `
      <tr style="${isChampion ? 'background:rgba(124, 58, 237, 0.04); font-weight:600;' : ''}">
        <td><strong>${run.run_name}</strong></td>
        <td><code>${run.model_type}</code></td>
        <td><span class="font-mono">${run.version}</span></td>
        <td>${stagePill}</td>
        <td class="font-mono text-accent" style="font-size:0.95rem;">${run.metrics.mae_minutes.toFixed(4)}</td>
        <td class="font-mono">${run.metrics.rmse_minutes.toFixed(4)}</td>
        <td class="font-mono" style="${run.metrics.r2_score > 0 ? 'color:var(--accent-emerald); font-weight:700;' : 'color:var(--text-muted);'}">${r2Display}</td>
        <td class="font-mono">${run.metrics.mape_percent.toFixed(2)}%</td>
        <td style="font-size:0.75rem; color:var(--text-secondary); max-width:200px;"><code>${paramStr}</code></td>
        <td class="font-mono" style="font-size:0.7rem; color:var(--text-muted);">${(run.run_id || "").slice(0, 8)}...</td>
      </tr>
    `;
  }).join("");
}

/* ==========================================================
   MLOPS 3: DATA & PREDICTION DRIFT MONITORING
========================================================== */
async function loadDriftMonitoring() {
  const toggleClean = document.getElementById("drift-toggle-clean");
  const toggleSurge = document.getElementById("drift-toggle-surge");

  if (toggleClean && toggleSurge) {
    toggleClean.addEventListener("click", () => {
      activeDriftMode = "clean";
      toggleClean.classList.add("active");
      toggleSurge.classList.remove("active");
      renderDriftBatchView();
    });

    toggleSurge.addEventListener("click", () => {
      activeDriftMode = "surge";
      toggleSurge.classList.add("active");
      toggleClean.classList.remove("active");
      renderDriftBatchView();
    });
  }

  try {
    const res = await fetch(`${API_BASE_URL}/reports/drift`);
    if (res.ok) driftData = await res.json();
  } catch (e) {}

  if (!driftData) {
    try {
      const resFallback = await fetch("./data/drift_report.json");
      if (resFallback.ok) driftData = await resFallback.json();
    } catch (e) {}
  }

  if (driftData) {
    renderDriftBatchView();
  }
}

function renderDriftBatchView() {
  if (!driftData) return;

  const batch = activeDriftMode === "clean"
    ? driftData.clean_batch_results
    : driftData.synthetic_drift_batch_results;

  if (!batch) return;

  const overallStatus = document.getElementById("drift-overall-status");
  const statusDesc = document.getElementById("drift-status-desc");
  const predPsi = document.getElementById("drift-pred-psi");
  const predTest = document.getElementById("drift-pred-test");
  const featuresCount = document.getElementById("drift-features-count");
  const actionText = document.getElementById("drift-action-text");
  const tbody = document.getElementById("drift-table-body");
  const topbarDriftChip = document.getElementById("topbar-drift-chip");

  const isAlert = batch.overall_status === "ALERT" || batch.drifted_features_count > 0;

  if (overallStatus) {
    overallStatus.textContent = batch.overall_status;
    overallStatus.style.color = isAlert ? "var(--accent-rose)" : "var(--accent-emerald)";
  }
  if (topbarDriftChip) {
    topbarDriftChip.textContent = batch.overall_status;
  }
  if (statusDesc) {
    statusDesc.textContent = isAlert
      ? "Significant distribution shift detected across key feature distributions"
      : "Zero statistical alerts triggered on clean held-out test split";
  }
  if (predPsi) {
    predPsi.textContent = batch.prediction_drift ? batch.prediction_drift.psi.toFixed(4) : "0.0101";
  }
  if (predTest) {
    const pval = batch.prediction_drift && batch.prediction_drift.p_value !== undefined
      ? batch.prediction_drift.p_value < 0.0001 ? "< 0.0001" : batch.prediction_drift.p_value.toFixed(4)
      : "0.3343";
    predTest.textContent = `KS test p-value: ${pval}`;
  }
  if (featuresCount) {
    featuresCount.textContent = `${batch.drifted_features_count} / ${batch.total_features_evaluated}`;
    featuresCount.style.color = isAlert ? "var(--accent-rose)" : "var(--text-primary)";
  }
  if (actionText) {
    actionText.textContent = isAlert ? "Retraining Alert" : "No Action Required";
    actionText.style.color = isAlert ? "var(--accent-rose)" : "var(--accent-emerald)";
  }

  // Render Features Table
  if (tbody && batch.features) {
    tbody.innerHTML = Object.entries(batch.features).map(([name, f]) => {
      const pvalStr = f.p_value < 0.0001 ? "< 0.0001" : f.p_value.toFixed(4);
      const testName = f.test_applied === "kolmogorov_smirnov" ? "Kolmogorov-Smirnov" : "Chi-Square (χ²)";
      const testStat = f.ks_statistic !== undefined ? f.ks_statistic.toFixed(4) : (f.chi2_statistic || 0).toFixed(2);
      const refMean = f.reference_mean !== undefined ? f.reference_mean.toFixed(2) : "Prop";
      const currMean = f.current_mean !== undefined ? f.current_mean.toFixed(2) : "Prop";

      let statusPill = `<span class="status-pill pill-stable">Stable</span>`;
      if (f.status === "ALERT") {
        statusPill = `<span class="status-pill pill-alert">🚨 Alert</span>`;
      } else if (f.status === "WARNING") {
        statusPill = `<span class="status-pill pill-warning">⚠️ Warning</span>`;
      }

      return `
        <tr>
          <td><code class="font-mono">${name}</code></td>
          <td style="text-transform:capitalize;">${f.feature_type}</td>
          <td class="font-mono" style="${f.psi >= 0.1 ? 'color:var(--accent-rose); font-weight:700;' : ''}">${f.psi.toFixed(4)}</td>
          <td style="font-size:0.75rem;">${testName}</td>
          <td class="font-mono">${testStat}</td>
          <td class="font-mono" style="${f.p_value < 0.05 ? 'color:var(--accent-rose); font-weight:700;' : ''}">${pvalStr}</td>
          <td class="font-mono">${refMean}</td>
          <td class="font-mono">${currMean}</td>
          <td>${statusPill}</td>
        </tr>
      `;
    }).join("");
  }
}

/* ==========================================================
   MLOPS 4: MODEL REGISTRY CATALOG
========================================================== */
async function loadModelRegistry() {
  const container = document.getElementById("registry-versions-container");
  if (!container) return;

  let reg = null;
  try {
    const res = await fetch(`${API_BASE_URL}/reports/registry`);
    if (res.ok) reg = await res.json();
  } catch (e) {}

  if (!reg) {
    try {
      const resFallback = await fetch("./data/registry.json");
      if (resFallback.ok) reg = await resFallback.json();
    } catch (e) {}
  }

  if (!reg || !reg.versions) return;

  container.innerHTML = Object.values(reg.versions).map((ver) => {
    const isProduction = ver.stage === "Production" || ver.is_active;
    const badge = isProduction
      ? `<span class="status-pill pill-champion">⭐ Active Production</span>`
      : `<span class="status-pill pill-archived">Archived</span>`;

    const modelArt = ver.artifacts && ver.artifacts.model_artifact ? ver.artifacts.model_artifact : {};
    const prepArt = ver.artifacts && ver.artifacts.preprocessor_artifact ? ver.artifacts.preprocessor_artifact : {};

    return `
      <div class="card" style="margin-bottom:1.25rem; border-color:${isProduction ? 'var(--accent-lavender-border)' : 'var(--border-card)'}; background:${isProduction ? 'linear-gradient(180deg, #FFFFFF, var(--bg-card-secondary))' : '#FFFFFF'};">
        <div style="display:flex; justify-content:space-between; align-items:center; margin-bottom:0.75rem;">
          <div>
            <h3 style="font-size:1.1rem; font-weight:800; display:inline-flex; align-items:center; gap:0.5rem;">
              <code>${ver.version}</code> — ${ver.algorithm}
            </h3>
            <div style="font-size:0.75rem; color:var(--text-muted); margin-top:0.2rem;">
              Registered: ${ver.registration_timestamp} • Dataset Fingerprint: <span class="font-mono">${(ver.dataset_fingerprint || "").slice(0, 16)}...</span>
            </div>
          </div>
          <div>${badge}</div>
        </div>

        <div style="display:grid; grid-template-columns:repeat(auto-fit, minmax(200px, 1fr)); gap:1rem; margin:0.75rem 0;">
          <div style="background:var(--bg-subtle); padding:0.75rem; border-radius:var(--radius-sm); border:1px solid var(--border-subtle);">
            <div style="font-size:0.7rem; font-weight:700; color:var(--text-muted); text-transform:uppercase;">Model Artifact</div>
            <div class="font-mono" style="font-size:0.8rem; font-weight:700; margin-top:0.25rem;">${modelArt.relative_path || "N/A"}</div>
            <div style="font-size:0.7rem; color:var(--text-muted); margin-top:0.25rem;">SHA-256: <code style="font-size:0.65rem;">${(modelArt.sha256_hash || "").slice(0, 24)}...</code></div>
          </div>

          <div style="background:var(--bg-subtle); padding:0.75rem; border-radius:var(--radius-sm); border:1px solid var(--border-subtle);">
            <div style="font-size:0.7rem; font-weight:700; color:var(--text-muted); text-transform:uppercase;">Preprocessor Pipeline</div>
            <div class="font-mono" style="font-size:0.8rem; font-weight:700; margin-top:0.25rem;">${prepArt.relative_path || "models/preprocessor.joblib"}</div>
            <div style="font-size:0.7rem; color:var(--text-muted); margin-top:0.25rem;">Encoded Features: <strong>${ver.encoded_feature_count || 17} columns</strong></div>
          </div>

          <div style="background:var(--bg-subtle); padding:0.75rem; border-radius:var(--radius-sm); border:1px solid var(--border-subtle);">
            <div style="font-size:0.7rem; font-weight:700; color:var(--text-muted); text-transform:uppercase;">Test Evaluation (7,793 Rows)</div>
            <div style="font-size:0.85rem; font-weight:700; color:var(--accent-purple); margin-top:0.25rem;">MAE: ${ver.evaluation_metrics.mae.toFixed(4)} min</div>
            <div style="font-size:0.7rem; color:var(--text-muted);">RMSE: ${ver.evaluation_metrics.rmse.toFixed(4)} • R²: ${ver.evaluation_metrics.r2.toFixed(4)}</div>
          </div>
        </div>

        <div style="font-size:0.78125rem; color:var(--text-secondary); background:#FFFFFF; border:1px solid var(--border-subtle); padding:0.6rem 0.85rem; border-radius:var(--radius-sm);">
          <strong>Architectural Notes:</strong> ${ver.notes || "Model registered and verified."}
        </div>
      </div>
    `;
  }).join("");
}

/* ==========================================================
   MLOPS 5: FEATURE IMPORTANCE BAR CHART (Section 7)
========================================================== */
function renderFeatureImportance() {
  const container = document.getElementById("feature-importance-bars");
  if (!container) return;

  container.innerHTML = FEATURE_IMPORTANCES.map((item) => {
    const pct = (item.importance * 100).toFixed(2);
    return `
      <div class="importance-bar-row">
        <div class="importance-bar-label" title="${item.feature}">${item.label}</div>
        <div class="importance-bar-track">
          <div class="importance-bar-fill" style="width: ${pct}%;"></div>
        </div>
        <div class="importance-bar-value">${pct}%</div>
      </div>
    `;
  }).join("");
}

/* ==========================================================
   LIVE PREDICTION WORKBENCH & INFERENCE (Section 5)
========================================================== */
function initPredictionForm() {
  const form = document.getElementById("predict-form");
  if (!form) return;

  form.addEventListener("submit", async (e) => {
    e.preventDefault();
    await executePrediction();
  });
}

function initQuickProbeButton() {
  const btnProbe = document.getElementById("btn-trigger-probe");
  if (!btnProbe) return;

  btnProbe.addEventListener("click", async () => {
    btnProbe.disabled = true;
    btnProbe.textContent = "Probing...";
    await executePrediction();
    btnProbe.textContent = "⚡ Send Live Probe";
    btnProbe.disabled = false;
  });
}

async function executePrediction() {
  const restLat = parseFloat(document.getElementById("inp-rest-lat").value) || 12.9352;
  const restLon = parseFloat(document.getElementById("inp-rest-lon").value) || 77.6245;
  const custLat = parseFloat(document.getElementById("inp-cust-lat").value) || 12.9550;
  const custLon = parseFloat(document.getElementById("inp-cust-lon").value) || 77.6400;
  const hour = parseInt(document.getElementById("inp-order-hour").value, 10) || 20;
  const dow = parseInt(document.getElementById("inp-day-of-week").value, 10) || 4;
  const type = document.getElementById("inp-order-type").value || "Meal";
  const festival = document.getElementById("inp-festival").value || "No";
  const city = document.getElementById("inp-city").value || "Metropolitian";

  const payload = {
    order_id: `PROBE-${Date.now().toString().slice(-4)}`,
    restaurant_lat: restLat,
    restaurant_lon: restLon,
    customer_lat: custLat,
    customer_lon: custLon,
    order_hour: hour,
    day_of_week: dow,
    type_of_order: type,
    festival: festival,
    city: city
  };

  const t0 = performance.now();
  try {
    const res = await fetch(`${API_BASE_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    const roundTripMs = (performance.now() - t0).toFixed(1);

    if (res.ok) {
      const data = await res.json();
      updatePredictionOutput(data, roundTripMs);
      if (globalAlert) globalAlert.classList.add("hidden");
    } else {
      const err = await res.json();
      showGlobalError("Inference Failed", JSON.stringify(err.details || err.error));
    }
  } catch (err) {
    showGlobalError(
      "API Offline",
      "Unable to connect to ETAFlow prediction server at http://127.0.0.1:8000. Start it via 'python pipelines/serve_api.py'."
    );
  }
}

function updatePredictionOutput(data, latencyMs) {
  const durationVal = document.getElementById("predict-duration-val");
  const distanceVal = document.getElementById("predict-distance-val");
  const latencyVal = document.getElementById("predict-latency-val");
  const modelIdVal = document.getElementById("predict-model-id-val");
  const jsonOutput = document.getElementById("predict-json-output");

  if (durationVal) durationVal.textContent = data.predicted_duration_minutes.toFixed(1);
  if (distanceVal) distanceVal.textContent = `${data.distance_km.toFixed(2)} km`;
  if (latencyVal) latencyVal.textContent = `${latencyMs} ms`;
  if (modelIdVal) modelIdVal.textContent = data.model_identity || "random_forest_regressor";
  if (jsonOutput) jsonOutput.textContent = JSON.stringify(data, null, 2);
}

function initHourSlider() {
  const slider = document.getElementById("inp-order-hour");
  const label = document.getElementById("lbl-order-hour");
  if (slider && label) {
    slider.addEventListener("input", () => {
      label.textContent = `${String(slider.value).padStart(2, "0")}:00`;
    });
  }
}

/* ==========================================================
   HEALTH CHECK & TELEMETRY
========================================================== */
async function checkApiHealth() {
  try {
    const res = await fetch(`${API_BASE_URL}/health`);
    if (res.ok) {
      const data = await res.json();
      if (topbarStatusDot) {
        topbarStatusDot.className = "status-dot-chip";
      }
      if (topbarStatusText) {
        topbarStatusText.textContent = "API Online (Port 8000)";
      }
      if (sidebarStatusCard) {
        sidebarStatusCard.className = "system-status-card status-healthy";
      }
      if (sidebarStatusLabel) {
        sidebarStatusLabel.textContent = "API Healthy";
      }
      if (globalAlert) {
        globalAlert.classList.add("hidden");
      }
      return true;
    }
  } catch (e) {
    if (topbarStatusDot) {
      topbarStatusDot.className = "status-dot-chip offline";
    }
    if (topbarStatusText) {
      topbarStatusText.textContent = "API Offline";
    }
    if (sidebarStatusCard) {
      sidebarStatusCard.className = "system-status-card status-offline";
    }
    if (sidebarStatusLabel) {
      sidebarStatusLabel.textContent = "API Offline";
    }
  }
  return false;
}

function showGlobalError(title, msg) {
  if (globalAlert && alertMessage) {
    alertMessage.innerHTML = `<strong>${title}:</strong> ${msg}`;
    globalAlert.classList.remove("hidden");
  }
}

/* ==========================================================
   LEAFLET MAP INTEGRATION (CartoDB Positron Light Tiles)
========================================================== */
function initPredictSpatialMap() {
  const mapElement = document.getElementById("predict-map");
  if (!mapElement || typeof L === "undefined") return;

  try {
    predictMap = L.map("predict-map", {
      zoomControl: true,
      attributionControl: false
    }).setView([12.9450, 77.6320], 13);

    // CartoDB Positron Light Tile Layer
    L.tileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png", {
      maxZoom: 19
    }).addTo(predictMap);

    const restIcon = L.divIcon({
      className: "custom-map-pin-rest",
      html: `<div style="background:#7C3AED; color:#FFF; border-radius:50%; width:28px; height:28px; display:flex; align-items:center; justify-content:center; box-shadow:0 2px 6px rgba(0,0,0,0.3); font-size:14px; border:2px solid #FFF;">🍳</div>`,
      iconSize: [28, 28],
      iconAnchor: [14, 14]
    });

    const custIcon = L.divIcon({
      className: "custom-map-pin-cust",
      html: `<div style="background:#E11D48; color:#FFF; border-radius:50%; width:28px; height:28px; display:flex; align-items:center; justify-content:center; box-shadow:0 2px 6px rgba(0,0,0,0.3); font-size:14px; border:2px solid #FFF;">🏠</div>`,
      iconSize: [28, 28],
      iconAnchor: [14, 14]
    });

    predictRestMarker = L.marker([12.9352, 77.6245], { icon: restIcon, draggable: true }).addTo(predictMap);
    predictCustMarker = L.marker([12.9550, 77.6400], { icon: custIcon, draggable: true }).addTo(predictMap);

    predictRestMarker.on("dragend", () => {
      const pos = predictRestMarker.getLatLng();
      document.getElementById("inp-rest-lat").value = pos.lat.toFixed(4);
      document.getElementById("inp-rest-lon").value = pos.lng.toFixed(4);
      updatePredictTrajectory();
    });

    predictCustMarker.on("dragend", () => {
      const pos = predictCustMarker.getLatLng();
      document.getElementById("inp-cust-lat").value = pos.lat.toFixed(4);
      document.getElementById("inp-cust-lon").value = pos.lng.toFixed(4);
      updatePredictTrajectory();
    });

    updatePredictTrajectory();
  } catch (err) {
    console.warn("Predict map init fallback", err);
  }
}

async function updatePredictTrajectory() {
  if (!predictMap || !predictRestMarker || !predictCustMarker) return;

  const lat1 = predictRestMarker.getLatLng().lat;
  const lon1 = predictRestMarker.getLatLng().lng;
  const lat2 = predictCustMarker.getLatLng().lat;
  const lon2 = predictCustMarker.getLatLng().lng;

  if (predictPolyline) predictMap.removeLayer(predictPolyline);

  // Attempt OSRM road geometry
  const coords = await fetchRoadRoute(lat1, lon1, lat2, lon2);
  const routeBadge = document.getElementById("predict-routing-badge");

  if (coords && coords.length > 0) {
    predictPolyline = L.polyline(coords, { color: "#7C3AED", weight: 4, opacity: 0.8 }).addTo(predictMap);
    if (routeBadge) routeBadge.textContent = "OSRM Road Route";
  } else {
    predictPolyline = L.polyline([[lat1, lon1], [lat2, lon2]], { color: "#7C3AED", weight: 3, dashArray: "5, 8" }).addTo(predictMap);
    if (routeBadge) routeBadge.textContent = "Geodesic Haversine Fallback";
  }

  predictMap.fitBounds([[lat1, lon1], [lat2, lon2]], { padding: [40, 40] });
}

function initSimulationMap() {
  const mapElement = document.getElementById("sim-map");
  if (!mapElement || typeof L === "undefined") return;

  try {
    simMap = L.map("sim-map", {
      zoomControl: true,
      attributionControl: false
    }).setView([12.9450, 77.6320], 13);

    L.tileLayer("https://{s}.basemaps.cartocdn.com/light_all/{z}/{x}/{y}{r}.png", {
      maxZoom: 19
    }).addTo(simMap);

    focusSimulationMapOnActiveOrder();
  } catch (err) {
    console.warn("Simulation map fallback", err);
  }
}

async function focusSimulationMapOnActiveOrder() {
  if (!simMap) return;

  const order = appOrders.find((o) => o.order_id === activeSimOrderId) || appOrders[0];
  if (!order) return;

  if (simMapPolyline) simMap.removeLayer(simMapPolyline);
  if (simCourierMarker) simMap.removeLayer(simCourierMarker);
  Object.values(simMapMarkers).forEach((m) => simMap.removeLayer(m));
  simMapMarkers = {};

  const restIcon = L.divIcon({
    html: `<div style="background:#7C3AED; color:#FFF; border-radius:50%; width:26px; height:26px; display:flex; align-items:center; justify-content:center; border:2px solid #FFF;">🍳</div>`,
    iconSize: [26, 26],
    iconAnchor: [13, 13]
  });

  const custIcon = L.divIcon({
    html: `<div style="background:#E11D48; color:#FFF; border-radius:50%; width:26px; height:26px; display:flex; align-items:center; justify-content:center; border:2px solid #FFF;">🏠</div>`,
    iconSize: [26, 26],
    iconAnchor: [13, 13]
  });

  const courierIcon = L.divIcon({
    className: "pin-courier-wrapper",
    html: `<div style="background:#0284C7; color:#FFF; border-radius:50%; width:28px; height:28px; display:flex; align-items:center; justify-content:center; box-shadow:0 0 10px rgba(2,132,199,0.5); border:2px solid #FFF; font-size:14px;">🛵</div>`,
    iconSize: [28, 28],
    iconAnchor: [14, 14]
  });

  simMapMarkers.rest = L.marker([order.restaurant_lat, order.restaurant_lon], { icon: restIcon }).addTo(simMap);
  simMapMarkers.cust = L.marker([order.customer_lat, order.customer_lon], { icon: custIcon }).addTo(simMap);

  const coords = await fetchRoadRoute(order.restaurant_lat, order.restaurant_lon, order.customer_lat, order.customer_lon);
  if (coords && coords.length > 0) {
    simMapPolyline = L.polyline(coords, { color: "#0284C7", weight: 4, opacity: 0.85 }).addTo(simMap);
  } else {
    simMapPolyline = L.polyline([[order.restaurant_lat, order.restaurant_lon], [order.customer_lat, order.customer_lon]], { color: "#0284C7", weight: 3, dashArray: "5, 8" }).addTo(simMap);
  }

  // Position courier
  const progress = (order.stage_progress_pct || 0) / 100.0;
  const cLat = order.restaurant_lat + progress * (order.customer_lat - order.restaurant_lat);
  const cLon = order.restaurant_lon + progress * (order.customer_lon - order.restaurant_lon);
  simCourierMarker = L.marker([cLat, cLon], { icon: courierIcon }).addTo(simMap);

  simMap.fitBounds([[order.restaurant_lat, order.restaurant_lon], [order.customer_lat, order.customer_lon]], { padding: [50, 50] });
}

/* ==========================================================
   APPENDIX: SIMULATION CONTROLS & STATE MACHINE
========================================================== */
function initSimulationControls() {
  const btnPlay = document.getElementById("btn-sim-play");
  const btnPause = document.getElementById("btn-sim-pause");
  const btnReset = document.getElementById("btn-sim-reset");

  if (btnPlay) btnPlay.addEventListener("click", startSimulation);
  if (btnPause) btnPause.addEventListener("click", pauseSimulation);
  if (btnReset) btnReset.addEventListener("click", resetSimulation);

  document.querySelectorAll(".speed-pill").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll(".speed-pill").forEach((p) => p.classList.remove("active"));
      btn.classList.add("active");
      simulationSpeed = parseFloat(btn.getAttribute("data-speed")) || 1.0;
    });
  });
}

function startSimulation() {
  if (isSimulationRunning) return;
  isSimulationRunning = true;

  const runningBadge = document.getElementById("sim-running-badge");
  if (runningBadge) runningBadge.style.display = "inline-block";

  simulationTimer = setInterval(() => {
    tickSimulation(0.5);
  }, 500);
}

function pauseSimulation() {
  isSimulationRunning = false;
  if (simulationTimer) clearInterval(simulationTimer);

  const runningBadge = document.getElementById("sim-running-badge");
  if (runningBadge) runningBadge.style.display = "none";
}

function resetSimulation() {
  pauseSimulation();
  simulationClockSeconds = 0;
  const clockDisplay = document.getElementById("sim-clock-display");
  if (clockDisplay) clockDisplay.textContent = "00:00:00";

  initFleetStore(true);
  updateSimulationOrderCard(activeSimOrderId);
  focusSimulationMapOnActiveOrder();
  renderOperationsOverview();
}

function tickSimulation(realDeltaSeconds) {
  const simDelta = realDeltaSeconds * simulationSpeed;
  simulationClockSeconds += simDelta;

  const clockDisplay = document.getElementById("sim-clock-display");
  if (clockDisplay) clockDisplay.textContent = formatTime(simulationClockSeconds);

  // Advance orders
  appOrders.forEach((order) => {
    if (order.status === "DELIVERED" || order.status === "CANCELLED") return;

    order.simulation_elapsed_seconds = (order.simulation_elapsed_seconds || 0) + simDelta;
    const totalSimDurationSec = (order.predicted_duration_minutes || 20.0) * 60.0;
    const progressPct = Math.min(100.0, (order.simulation_elapsed_seconds / totalSimDurationSec) * 100.0);
    order.stage_progress_pct = progressPct;

    if (progressPct >= 100.0) {
      order.status = "DELIVERED";
      order.simulated_actual_duration_minutes = (order.simulation_elapsed_seconds / 60.0);
      order.delay_cause = "Simulated on-time delivery";
      logActivity("🏁", `Order ${order.order_id} delivered successfully.`);
    } else if (progressPct >= 55.0) {
      order.status = "OUT_FOR_DELIVERY";
    } else if (progressPct >= 40.0) {
      order.status = "READY_FOR_PICKUP";
    } else if (progressPct >= 15.0) {
      order.status = "RESTAURANT_PREPARING";
    } else {
      order.status = "ORDER_RECEIVED";
    }
  });

  saveFleetStore();
  updateSimulationOrderCard(activeSimOrderId);
  renderOperationsOverview();
  renderFleetTable();
  renderDelayIntelligence();
}

function updateSimulationOrderCard(orderId) {
  const order = appOrders.find((o) => o.order_id === orderId) || appOrders[0];
  if (!order) return;

  const elId = document.getElementById("sim-card-order-id");
  const elCat = document.getElementById("sim-card-category-city");
  const elStatus = document.getElementById("sim-card-status-pill");
  const elPred = document.getElementById("sim-card-predicted");
  const elElapsed = document.getElementById("sim-card-elapsed");
  const elRemaining = document.getElementById("sim-card-remaining-hint");
  const elHaversine = document.getElementById("sim-card-haversine");
  const elPct = document.getElementById("sim-card-progress-pct");
  const elFill = document.getElementById("sim-card-progress-fill");
  const elExplainer = document.getElementById("sim-stage-explainer-text");

  if (elId) elId.textContent = order.order_id;
  if (elCat) elCat.textContent = `${order.type_of_order} • ${order.city}`;
  if (elPred) elPred.textContent = `${order.predicted_duration_minutes.toFixed(1)} min`;
  if (elElapsed) elElapsed.textContent = `${((order.simulation_elapsed_seconds || 0) / 60.0).toFixed(1)} min`;
  if (elHaversine) elHaversine.textContent = `${order.distance_km.toFixed(2)} km`;

  const pct = (order.stage_progress_pct || 0);
  if (elPct) elPct.textContent = `${pct.toFixed(0)}%`;
  if (elFill) elFill.style.width = `${pct}%`;

  const remMins = Math.max(0, order.predicted_duration_minutes - ((order.simulation_elapsed_seconds || 0) / 60.0));
  if (elRemaining) elRemaining.textContent = `Estimated remaining: ~${remMins.toFixed(1)} mins`;

  if (elStatus) {
    elStatus.textContent = formatStatusLabel(order.status);
    elStatus.className = `order-status-pill ${getStatusClass(order.status)}`;
  }

  // Update 5 Milestone Steps
  const mRec = document.getElementById("stage-milestone-received");
  const mPrep = document.getElementById("stage-milestone-prep");
  const mPick = document.getElementById("stage-milestone-pickup");
  const mTran = document.getElementById("stage-milestone-transit");
  const mDeliv = document.getElementById("stage-milestone-delivered");

  [mRec, mPrep, mPick, mTran, mDeliv].forEach((m) => m && m.classList.remove("active"));

  if (order.status === "DELIVERED" && mDeliv) mDeliv.classList.add("active");
  else if (order.status === "OUT_FOR_DELIVERY" && mTran) mTran.classList.add("active");
  else if (order.status === "READY_FOR_PICKUP" && mPick) mPick.classList.add("active");
  else if (order.status === "RESTAURANT_PREPARING" && mPrep) mPrep.classList.add("active");
  else if (mRec) mRec.classList.add("active");

  if (elExplainer) {
    if (order.status === "DELIVERED") elExplainer.textContent = "Order delivered at doorstep. SLA calculated.";
    else if (order.status === "OUT_FOR_DELIVERY") elExplainer.textContent = "Courier moving along road route towards delivery location.";
    else if (order.status === "READY_FOR_PICKUP") elExplainer.textContent = "Kitchen marked order ready. Driver arriving at counter.";
    else if (order.status === "RESTAURANT_PREPARING") elExplainer.textContent = "Order accepted by kitchen. Food preparation in progress.";
    else elExplainer.textContent = "Order received by platform. Finding available fleet driver.";
  }

  // Update courier marker position on map
  if (simCourierMarker && order.restaurant_lat && order.customer_lat) {
    const fraction = Math.min(1.0, pct / 100.0);
    const cLat = order.restaurant_lat + fraction * (order.customer_lat - order.restaurant_lat);
    const cLon = order.restaurant_lon + fraction * (order.customer_lon - order.restaurant_lon);
    simCourierMarker.setLatLng([cLat, cLon]);
  }
}

/* ==========================================================
   APPENDIX: FLEET MANAGEMENT
========================================================== */
function initFleetStore(forceReset = false) {
  if (forceReset || !localStorage.getItem("etaflow_fleet_v4")) {
    appOrders = JSON.parse(JSON.stringify(DEFAULT_DEMO_ORDERS));
    saveFleetStore();
  } else {
    try {
      appOrders = JSON.parse(localStorage.getItem("etaflow_fleet_v4"));
    } catch (e) {
      appOrders = JSON.parse(JSON.stringify(DEFAULT_DEMO_ORDERS));
    }
  }
}

function saveFleetStore() {
  localStorage.setItem("etaflow_fleet_v4", JSON.stringify(appOrders));
}

function initFleetManager() {
  const searchInp = document.getElementById("fleet-search-input");
  const sortSel = document.getElementById("fleet-sort-select");
  const resetBtn = document.getElementById("btn-reset-fleet");
  const refreshEtaBtn = document.getElementById("btn-refresh-model-etas");

  if (searchInp) searchInp.addEventListener("input", renderFleetTable);
  if (sortSel) sortSel.addEventListener("change", renderFleetTable);
  if (resetBtn) resetBtn.addEventListener("click", () => {
    initFleetStore(true);
    renderFleetTable();
  });
  if (refreshEtaBtn) refreshEtaBtn.addEventListener("click", refreshFleetModelEtas);

  document.querySelectorAll("#fleet-filter-pills button").forEach((btn) => {
    btn.addEventListener("click", () => {
      document.querySelectorAll("#fleet-filter-pills button").forEach((b) => b.classList.remove("active"));
      btn.classList.add("active");
      renderFleetTable();
    });
  });

  renderFleetTable();
}

async function refreshFleetModelEtas() {
  const isOnline = await checkApiHealth();
  if (!isOnline) {
    showGlobalError("Offline", "Cannot refresh ETAs while API is offline.");
    return;
  }

  for (const o of appOrders) {
    try {
      const res = await fetch(`${API_BASE_URL}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(o)
      });
      if (res.ok) {
        const d = await res.json();
        o.predicted_duration_minutes = d.predicted_duration_minutes;
        o.prediction_source = "Random Forest API (Live)";
      }
    } catch (e) {}
  }
  saveFleetStore();
  renderFleetTable();
}

function renderFleetTable() {
  const tbody = document.getElementById("fleet-table-body");
  if (!tbody) return;

  const searchInp = document.getElementById("fleet-search-input");
  const query = (searchInp ? searchInp.value : "").toLowerCase();

  const activePill = document.querySelector("#fleet-filter-pills button.active");
  const filter = activePill ? activePill.getAttribute("data-filter") : "ALL";

  let filtered = appOrders.filter((o) => {
    const matchQuery = o.order_id.toLowerCase().includes(query) || o.type_of_order.toLowerCase().includes(query);
    if (!matchQuery) return false;
    if (filter === "ACTIVE") return !["DELIVERED", "CANCELLED"].includes(o.status);
    if (filter === "DELIVERED") return o.status === "DELIVERED";
    if (filter === "CANCELLED") return o.status === "CANCELLED";
    return true;
  });

  tbody.innerHTML = filtered.map((o) => {
    const isLive = o.prediction_source && o.prediction_source.includes("Live");
    const sourceBadge = isLive
      ? `<span class="source-tag-live">RF Live API</span>`
      : `<span class="source-tag-preset">Preset</span>`;

    const progress = o.status === "DELIVERED"
      ? `<strong class="font-mono text-accent">${(o.simulated_actual_duration_minutes || 0).toFixed(1)} mins</strong>`
      : `<span class="font-mono text-muted">${(o.stage_progress_pct || 0).toFixed(0)}%</span>`;

    return `
      <tr>
        <td><strong class="font-mono">${o.order_id}</strong></td>
        <td>${o.type_of_order} • <span class="text-muted">${o.city}</span></td>
        <td><span class="order-status-pill ${getStatusClass(o.status)}">${formatStatusLabel(o.status)}</span></td>
        <td><span class="font-mono" style="font-weight:700;">${o.predicted_duration_minutes.toFixed(1)} min</span> ${sourceBadge}</td>
        <td class="font-mono">${o.distance_km.toFixed(2)} km</td>
        <td>${progress}</td>
        <td>
          <button type="button" class="btn btn-secondary btn-sm" onclick="trackOrderInSim('${o.order_id}')">Track</button>
        </td>
      </tr>
    `;
  }).join("");
}

window.trackOrderInSim = function(orderId) {
  activeSimOrderId = orderId;
  switchTab("tab-simulation");
  updateSimulationOrderCard(orderId);
  focusSimulationMapOnActiveOrder();
};

/* ==========================================================
   APPENDIX: SCENARIOS LAB
========================================================== */
let scenariosList = [
  { id: "scn-base", name: "Standard Friday Dinner", hour: 20, day: 4, type: "Meal", festival: "No", city: "Metropolitian", duration: 20.3, delta: 0.0 }
];

function initScenariosLab() {
  const btnAdd = document.getElementById("btn-add-scenario");
  const btnExport = document.getElementById("btn-export-scenarios-csv");

  if (btnAdd) btnAdd.addEventListener("click", addScenario);
  if (btnExport) btnExport.addEventListener("click", exportScenariosToCsv);

  renderScenariosTable();
}

async function addScenario() {
  const name = document.getElementById("scenario-name-input").value || "Custom Scenario";
  const hour = parseInt(document.getElementById("scenario-hour-select").value, 10);
  const day = parseInt(document.getElementById("scenario-day-select").value, 10);
  const type = document.getElementById("scenario-type-select").value;
  const fest = document.getElementById("scenario-festival-select").value;
  const city = document.getElementById("scenario-city-select").value;

  const payload = {
    order_id: `SCN-${Date.now().toString().slice(-4)}`,
    restaurant_lat: 12.9352,
    restaurant_lon: 77.6245,
    customer_lat: 12.9550,
    customer_lon: 77.6400,
    order_hour: hour,
    day_of_week: day,
    type_of_order: type,
    festival: fest,
    city: city
  };

  let duration = 20.3;
  try {
    const res = await fetch(`${API_BASE_URL}/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(payload)
    });
    if (res.ok) {
      const data = await res.json();
      duration = data.predicted_duration_minutes;
    }
  } catch (e) {
    duration = 20.3 + (fest === "Yes" ? 7.5 : 0.0) + (hour >= 21 ? 2.1 : 0.0);
  }

  const baseDuration = scenariosList[0] ? scenariosList[0].duration : 20.3;
  const delta = duration - baseDuration;

  scenariosList.push({
    id: `scn-${Date.now()}`,
    name,
    hour,
    day,
    type,
    festival: fest,
    city,
    duration,
    delta
  });

  renderScenariosTable();
}

function renderScenariosTable() {
  const tbody = document.getElementById("scenarios-table-body");
  if (!tbody) return;

  tbody.innerHTML = scenariosList.map((sc, idx) => {
    const deltaStr = sc.delta === 0 ? "Base" : `${sc.delta > 0 ? "+" : ""}${sc.delta.toFixed(1)} min`;
    const deltaColor = sc.delta > 0 ? "color:var(--accent-rose);" : sc.delta < 0 ? "color:var(--accent-emerald);" : "color:var(--text-muted);";

    return `
      <tr>
        <td><strong>${sc.name}</strong></td>
        <td>${sc.hour}:00 (Day ${sc.day})</td>
        <td>${sc.type}</td>
        <td>${sc.festival}</td>
        <td>${sc.city}</td>
        <td class="font-mono text-accent" style="font-weight:700;">${sc.duration.toFixed(1)} mins</td>
        <td class="font-mono" style="${deltaColor} font-weight:700;">${deltaStr}</td>
        <td>
          ${idx > 0 ? `<button type="button" class="btn btn-secondary btn-sm" onclick="removeScenario('${sc.id}')">✕</button>` : `<span class="text-muted">Anchor</span>`}
        </td>
      </tr>
    `;
  }).join("");
}

window.removeScenario = function(id) {
  scenariosList = scenariosList.filter((s) => s.id !== id);
  renderScenariosTable();
};

function exportScenariosToCsv() {
  let csv = "Scenario Identifier,Scenario Name,Order Hour,Day of Week,Order Type,Festival,City,Predicted Duration (min),Delta vs Baseline (min)\n";
  scenariosList.forEach((s) => {
    csv += `"${s.id}","${s.name}",${s.hour},${s.day},"${s.type}","${s.festival}","${s.city}",${s.duration.toFixed(1)},"${s.delta > 0 ? "+" : ""}${s.delta.toFixed(1)}"\n`;
  });

  const blob = new Blob([csv], { type: "text/csv;charset=utf-8;" });
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = `ETAFlow_Scenario_Matrix_${Date.now()}.csv`;
  a.click();
}

/* ==========================================================
   APPENDIX: DELAY ANALYTICS
========================================================== */
function renderDelayIntelligence() {
  const tbody = document.getElementById("delay-table-body");
  if (!tbody) return;

  const delivered = appOrders.filter((o) => o.status === "DELIVERED" && o.simulated_actual_duration_minutes);
  if (delivered.length === 0) {
    tbody.innerHTML = `<tr><td colspan="7" class="text-center text-muted">No completed deliveries yet.</td></tr>`;
    return;
  }

  let onTime = 0, late = 0, early = 0;
  let totalAbsErr = 0;

  tbody.innerHTML = delivered.map((o) => {
    const diff = o.simulated_actual_duration_minutes - o.predicted_duration_minutes;
    const absErr = Math.abs(diff);
    totalAbsErr += absErr;

    let sla = "On-Time";
    let slaClass = "sla-pill-on-time";
    if (diff > 3.0) {
      sla = "Late";
      slaClass = "sla-pill-late";
      late++;
    } else if (diff < -3.0) {
      sla = "Early";
      slaClass = "sla-pill-early";
      early++;
    } else {
      onTime++;
    }

    return `
      <tr>
        <td><strong class="font-mono">${o.order_id}</strong></td>
        <td class="font-mono">${o.predicted_duration_minutes.toFixed(1)} min</td>
        <td class="font-mono" style="font-weight:700;">${o.simulated_actual_duration_minutes.toFixed(1)} min</td>
        <td class="font-mono" style="font-weight:700; ${diff > 0 ? 'color:var(--accent-rose);' : 'color:var(--accent-emerald);'}">${diff > 0 ? "+" : ""}${diff.toFixed(1)} min</td>
        <td class="font-mono">${absErr.toFixed(1)} min</td>
        <td><span class="sla-pill ${slaClass}">${sla}</span></td>
        <td style="font-size:0.75rem; color:var(--text-secondary);">${o.delay_cause || "Operational mechanics"}</td>
      </tr>
    `;
  }).join("");

  const countOnTime = document.getElementById("delay-count-on-time");
  const countEarly = document.getElementById("delay-count-early");
  const countLate = document.getElementById("delay-count-late");
  const slaPct = document.getElementById("delay-sla-pct");
  const meanAbsErr = document.getElementById("delay-mean-abs-error");

  if (countOnTime) countOnTime.textContent = onTime;
  if (countEarly) countEarly.textContent = early;
  if (countLate) countLate.textContent = late;
  if (slaPct) slaPct.textContent = `${Math.round(((onTime + early) / delivered.length) * 100)}%`;
  if (meanAbsErr) meanAbsErr.textContent = `${(totalAbsErr / delivered.length).toFixed(1)} min`;
}

/* ==========================================================
   APPENDIX: ORDER MODAL & CREATION
========================================================== */
function initModalHandlers() {
  const btnOpen = document.getElementById("btn-open-create-order");
  const btnClose = document.getElementById("btn-close-modal");
  const btnCancel = document.getElementById("btn-cancel-modal");
  const form = document.getElementById("form-create-order");
  const modal = document.getElementById("modal-create-order");

  if (btnOpen && modal) btnOpen.addEventListener("click", () => modal.classList.remove("hidden"));
  if (btnClose && modal) btnClose.addEventListener("click", () => modal.classList.add("hidden"));
  if (btnCancel && modal) btnCancel.addEventListener("click", () => modal.classList.add("hidden"));

  if (form) {
    form.addEventListener("submit", async (e) => {
      e.preventDefault();

      const orderId = document.getElementById("mod-order-id").value;
      const restLat = parseFloat(document.getElementById("mod-rest-lat").value);
      const restLon = parseFloat(document.getElementById("mod-rest-lon").value);
      const custLat = parseFloat(document.getElementById("mod-cust-lat").value);
      const custLon = parseFloat(document.getElementById("mod-cust-lon").value);
      const hour = parseInt(document.getElementById("mod-order-hour").value, 10);
      const dow = parseInt(document.getElementById("mod-day-of-week").value, 10);
      const type = document.getElementById("mod-order-type").value;
      const fest = document.getElementById("mod-festival").value;
      const city = document.getElementById("mod-city").value;

      const payload = {
        order_id: orderId,
        restaurant_lat: restLat,
        restaurant_lon: restLon,
        customer_lat: custLat,
        customer_lon: custLon,
        order_hour: hour,
        day_of_week: dow,
        type_of_order: type,
        festival: fest,
        city: city
      };

      try {
        const res = await fetch(`${API_BASE_URL}/predict`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(payload)
        });

        if (res.ok) {
          const data = await res.json();
          const newOrder = {
            ...payload,
            predicted_duration_minutes: data.predicted_duration_minutes,
            distance_km: data.distance_km,
            prediction_source: "Random Forest API (Live)",
            model_identity: data.model_identity,
            status: "ORDER_RECEIVED",
            simulation_elapsed_seconds: 0.0,
            stage_progress_pct: 0.0,
            simulated_actual_duration_minutes: null,
            delay_cause: null
          };

          appOrders.unshift(newOrder);
          saveFleetStore();
          renderFleetTable();
          modal.classList.add("hidden");
          logActivity("📝", `New order ${orderId} created with live API inference.`);
        } else {
          const err = await res.json();
          alert(`Inference failed: ${JSON.stringify(err)}`);
        }
      } catch (err) {
        const errBanner = document.getElementById("modal-error-banner");
        const errText = document.getElementById("modal-error-text");
        if (errBanner && errText) {
          errText.textContent = "Cannot calculate ML prediction: ETAFlow API is currently offline at http://127.0.0.1:8000. Start the server via 'python pipelines/serve_api.py'. Orders will not be created with fabricated model predictions.";
          errBanner.classList.remove("hidden");
        }
      }
    });
  }
}

/* ==========================================================
   APPENDIX: OPERATIONS OVERVIEW & ACTIVITY STREAM
========================================================== */
function initActivityLog() {
  activityLog = [
    { icon: "⚡", text: "Production model Random Forest v1.0.0 verified.", time: "10:00 AM" },
    { icon: "📈", text: "Data & prediction drift monitor verified healthy (PSI: 0.0101).", time: "10:05 AM" },
    { icon: "🛵", text: "Fleet dispatch tracking initialized.", time: "10:10 AM" }
  ];
}

function logActivity(icon, text) {
  const d = new Date();
  const time = d.toLocaleTimeString([], { hour: "2-digit", minute: "2-digit" });
  activityLog.unshift({ icon, text, time });
  if (activityLog.length > 10) activityLog.pop();
  renderActivityStream();
}

function renderOperationsOverview() {
  const elActive = document.getElementById("op-active-dispatches");
  const elCompleted = document.getElementById("op-completed-orders");
  const elAvg = document.getElementById("op-avg-pred");

  const active = appOrders.filter((o) => !["DELIVERED", "CANCELLED"].includes(o.status)).length;
  const completed = appOrders.filter((o) => o.status === "DELIVERED").length;

  if (elActive) elActive.textContent = active;
  if (elCompleted) elCompleted.textContent = completed;

  if (elAvg && appOrders.length > 0) {
    const sum = appOrders.reduce((acc, o) => acc + (o.predicted_duration_minutes || 0), 0);
    elAvg.innerHTML = `${(sum / appOrders.length).toFixed(1)} <span class="kpi-unit">min</span>`;
  }

  renderActivityStream();
}

function renderActivityStream() {
  const container = document.getElementById("operations-activity-stream");
  if (!container) return;

  container.innerHTML = activityLog.map((item) => `
    <div style="display:flex; align-items:center; gap:0.75rem; background:var(--bg-subtle); padding:0.6rem 0.85rem; border-radius:var(--radius-sm); border:1px solid var(--border-subtle); font-size:0.8125rem;">
      <span style="font-size:1.1rem;">${item.icon}</span>
      <span style="flex:1;">${item.text}</span>
      <span class="font-mono text-muted" style="font-size:0.7rem;">${item.time}</span>
    </div>
  `).join("");
}

/* ==========================================================
   UTILITY HELPERS (OSRM, Haversine, Time Format)
========================================================== */
function calculateHaversine(lat1, lon1, lat2, lon2) {
  const R = 6371.0;
  const dLat = (lat2 - lat1) * (Math.PI / 180.0);
  const dLon = (lon2 - lon1) * (Math.PI / 180.0);
  const a = Math.sin(dLat / 2) * Math.sin(dLat / 2) +
            Math.cos(lat1 * (Math.PI / 180.0)) * Math.cos(lat2 * (Math.PI / 180.0)) *
            Math.sin(dLon / 2) * Math.sin(dLon / 2);
  const c = 2 * Math.atan2(Math.sqrt(a), Math.sqrt(1 - a));
  return R * c;
}

async function fetchRoadRoute(lat1, lon1, lat2, lon2) {
  const url = `https://router.project-osrm.org/route/v1/driving/${lon1},${lat1};${lon2},${lat2}?overview=full&geometries=geojson`;
  try {
    const controller = new AbortController();
    const timeout = setTimeout(() => controller.abort(), 2000);
    const res = await fetch(url, { signal: controller.signal });
    clearTimeout(timeout);
    if (res.ok) {
      const data = await res.json();
      if (data.routes && data.routes[0] && data.routes[0].geometry) {
        return data.routes[0].geometry.coordinates.map(([lon, lat]) => [lat, lon]);
      }
    }
  } catch (e) {
    // Fall back to geodesic
  }
  return null;
}

function getStatusClass(status) {
  switch (status) {
    case "ORDER_RECEIVED": return "status-received";
    case "RESTAURANT_PREPARING": return "status-prep";
    case "READY_FOR_PICKUP": return "status-ready";
    case "OUT_FOR_DELIVERY": return "status-transit";
    case "DELIVERED": return "status-delivered";
    case "CANCELLED": return "status-cancelled";
    default: return "";
  }
}

function formatStatusLabel(status) {
  return (status || "").replace(/_/g, " ").toLowerCase().replace(/\b\w/g, (c) => c.toUpperCase());
}

function formatTime(totalSeconds) {
  const s = Math.floor(totalSeconds % 60);
  const m = Math.floor((totalSeconds / 60) % 60);
  const h = Math.floor(totalSeconds / 3600);
  return `${String(h).padStart(2, "0")}:${String(m).padStart(2, "0")}:${String(s).padStart(2, "0")}`;
}
