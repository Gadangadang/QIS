# Tactical Asset Allocation (TAA) Framework Implementation Plan

## 1. Executive Summary
This document outlines the roadmap for integrating a Machine Learning-based Tactical Asset Allocation (TAA) framework into the `QuantTrading` repository. The goal is to build a system that provides weekly/monthly over- and underweight recommendations for sectors and industries, respecting a long-only constraint. The system will leverage XGBoost models trained on macro, fundamental, and technical features to forecast relative returns across multiple time horizons (1-week, 4-week, 12-week).

## 2. Architecture & Integration

### 2.1. Directory Structure Alignment
To maintain consistency with the existing `QuantTrading` structure while accommodating the complexity of the new TAA module, we propose the following integration path. Instead of creating a separate `src/` root, we will integrate into the existing `core/`, `signals/`, and `data/` structure, with a new dedicated `taa/` module for the specific logic.

```text
QuantTrading/
├── config/                     # NEW: Centralized configuration
│   ├── taa_model_params.yaml
│   ├── taa_constraints.yaml
│   └── data_sources.yaml
├── core/
│   ├── taa/                    # NEW: Core TAA logic
│   │   ├── __init__.py
│   │   ├── feature_pipeline.py # Feature engineering orchestration
│   │   ├── model_engine.py     # XGBoost/Ensemble wrappers
│   │   ├── optimizer.py        # Portfolio optimization (Mean-Variance)
│   │   └── constraints.py      # Long-only, tracking error logic
│   ├── data/                   # ENHANCED: Unified data layer
│   │   ├── collectors/         # Fetchers (Yahoo, FRED, Bloomberg)
│   │   ├── processors/         # Cleaning & alignment
│   │   └── storage/            # Parquet management
│   └── ... (existing core files)
├── signals/
│   ├── taa_signals.py          # Adapter to expose TAA as a SignalModel
│   └── ... (existing signals)
├── notebooks/
│   ├── taa/                    # NEW: TAA-specific notebooks
│   │   ├── 01_data_audit.ipynb
│   │   ├── 02_feature_dev.ipynb
│   │   └── ...
│   └── ...
├── reports/
│   ├── taa/                    # NEW: TAA reports
│   └── ...
└── scripts/
    ├── taa_weekly_run.py       # Production script
    └── ...
```

### 2.2. Integration with Existing Core
*   **Asset Registry**: We will expand `core/asset_registry.py` to include Sector ETFs (XLK, XLF, etc.) and Indices.
*   **Backtest Orchestrator**: The TAA module can function as a "Strategy" within the `BacktestOrchestrator`, or we can build a specialized `TAAOrchestrator` if the portfolio constraints (tracking error, sector neutrality) are too complex for the existing engine.
*   **Signal Interface**: We will create a `TAASignal` class that adheres to the `SignalModel` interface, allowing it to be backtested using the existing infrastructure where possible.

## 3. Detailed Implementation Phases

### Phase 1: Data Infrastructure (Weeks 1-2)
*   **Objective**: Establish a robust pipeline for weekly/monthly data.
*   **Tasks**:
    *   Audit existing data in `Dataset/`.
    *   Implement `core/data/collectors` for Yahoo Finance (yfinance) and FRED (pandas_datareader).
    *   Create `core/data/processors` to align data to weekly timestamps.
    *   **Deliverable**: A clean Parquet-based dataset containing 15+ years of Sector Prices, Macro Indicators, and Sentiment proxies.

### Phase 2: Feature Engineering (Weeks 3-4)
*   **Objective**: Transform raw data into predictive features.
*   **Tasks**:
    *   Implement `core/taa/feature_pipeline.py`.
    *   Create Momentum features (1w, 4w, 12w, 52w).
    *   Create Macro features (Yield Curve, Credit Spreads, VIX).
    *   Create Relative Value features (Sector vs SPX).
    *   **Deliverable**: A feature matrix with ~100-150 features per sector, handled for stationarity and outliers.

### Phase 3: Modeling & Validation (Weeks 5-8)
*   **Objective**: Train and validate the ML models.
*   **Tasks**:
    *   Implement `core/taa/model_engine.py` using XGBoost.
    *   Set up the Multi-Horizon target variables (1w, 4w, 12w forward returns).
    *   Implement Walk-Forward Validation (expanding window) to prevent look-ahead bias.
    *   **Deliverable**: Trained models with performance metrics (IR, Hit Rate) on out-of-sample data.

### Phase 4: Portfolio Construction (Weeks 9-10)
*   **Objective**: Convert forecasts into portfolio weights.
*   **Tasks**:
    *   Implement `core/taa/optimizer.py` using `cvxpy` or `scipy.optimize`.
    *   Define constraints: Long-only (0 <= w <= 1), Tracking Error limits, Max Sector deviation.
    *   Implement Transaction Cost models.
    *   **Deliverable**: An optimizer that takes forecasts and outputs target weights.

### Phase 5: Reporting & Dashboard (Weeks 11-12)
*   **Objective**: Create tools for PM interaction.
*   **Tasks**:
    *   Develop a weekly HTML report generator.
    *   (Optional) Build a Streamlit dashboard for "What-If" analysis.
    *   **Deliverable**: Automated weekly reporting pipeline.

## 4. Suggestions & Improvements

### 4.1. Data Quality & Governance (Crucial Addition)
The original plan assumes data is clean. We suggest adding a **Data Validation Layer** using `pandera` or `Great Expectations` within `core/data/processors`.
*   **Why**: Financial data is noisy. Tickers change, splits happen, API calls fail.
*   **Action**: Add checks for nulls, outliers (e.g., >20% daily move), and missing timestamps before any training occurs.

### 4.2. MLOps & Experiment Tracking
Instead of just saving `.pkl` files, we suggest integrating **MLflow** or **Weights & Biases**.
*   **Why**: You will run hundreds of experiments (features sets, hyperparameters). Tracking them manually in notebooks is error-prone.
*   **Action**: Add a simple logging wrapper in `core/taa/model_engine.py` to log params and metrics.

### 4.3. Explainability Dashboard
The PM needs to trust the "Black Box".
*   **Suggestion**: Include **SHAP (SHapley Additive exPlanations)** plots in the weekly report.
*   **Why**: It shows exactly *why* the model likes "Tech" this week (e.g., "High Momentum" + "Low VIX").

### 4.4. Dynamic Cost Models
The plan uses fixed bps for costs.
*   **Suggestion**: Use a volatility-adjusted cost model.
*   **Why**: Trading during high volatility (VIX > 30) is more expensive due to wider spreads. This prevents the model from over-trading in crises.

### 4.5. Benchmark Construction
*   **Clarification Needed**: Are we tracking the S&P 500 (market cap weighted) or an Equal Weight version? The "Sector Neutrality" constraint implies we need a precise definition of the benchmark weights history.
*   **Action**: Ensure we have historical benchmark constituent weights if possible, or approximate using Sector Market Caps.

## 5. Immediate Next Steps
1.  **Confirm Directory Structure**: Do you agree with the `core/taa` integration vs a separate `src/` folder?
2.  **Data Audit**: Run a script to check what data is currently available in `Dataset/` vs what needs to be fetched.
3.  **Environment Setup**: Install `xgboost`, `cvxpy`, `shap`, `yfinance`.

---
**Status**: Plan created. Awaiting approval to proceed with Phase 1.
