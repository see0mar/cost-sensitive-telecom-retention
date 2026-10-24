# Cost-Sensitive Machine Learning for Value Compression & Revenue Preservation in Telecom

**Python 3.10+** · **LightGBM** · **1-million-record synthetic benchmark** · **MIT License**

An enterprise-oriented predictive analytics pipeline that detects **value compression** — a reduction in subscriber spend of ≥ 25 % — and maximises campaign ROI by replacing the default 0.50 probability threshold with a **cost-derived decision boundary**.

> **Benchmark note:** This repository uses a 1-million-record synthetic dataset. All EGP figures are simulated estimates derived from the stated cost assumptions and are not claims of realised production revenue.

---

## 📌 Executive Summary

In subscription-based telecom markets, a dissatisfied subscriber may quietly reduce their spend — downgrading packages, dropping data add-ons — while their line **stays active**. Standard CRM triggers miss this cohort entirely because no line termination event fires.

This project frames the problem correctly: **value compression is a regression signal disguised as a classification problem**, where the cost of a missed at-risk account (500 EGP) is ten times the cost of a wasted retention offer (50 EGP). The break-even threshold is therefore:

```
θ* = C / V = 50 / 500 = 0.10
```

Shifting from the default θ = 0.50 to the cost-derived θ = 0.10 on the same trained model produces a **22× increase in net campaign value** (+373,950 EGP) — with no retraining, no additional data, and no change to the model architecture.

---

## 🧠 Three Analytical Hypotheses

| # | Hypothesis | Feature |
|---|------------|---------|
| **H1** | Recurring support-call frequency is a **leading indicator** of spend restriction — subscribers exceeding ≈ 3.89 calls/month show a non-linear risk inflection | `customer_service_calls_mean` |
| **H2** | Short-term revenue velocity (ΔR ≤ −0.25) is a stronger predictive signal than long-term static billing averages | `monthly_overage_fee_mean` |
| **H3** | A cost-sensitive decision boundary produces greater simulated campaign value than any symmetric classification metric | threshold optimisation |

---

## 🏗️ Repository Structure

```
cost-sensitive-telecom-retention/
├── data/                                          # gitignored — generated locally
├── notebooks/
│   └── 01_pipeline_demo.ipynb                    # Full walkthrough notebook
├── scripts/
│   ├── generate_data.py                          # Synthetic benchmark-data generator
│   └── build_notebook.py                         # Notebook source (developer use)
├── src/
│   ├── __init__.py
│   ├── data_processor.py                         # TelemetryDataProcessor
│   ├── model_trainer.py                          # RetentionModelTrainer
│   └── cost_evaluator.py                         # CostSensitiveOptimizer
├── .gitignore
├── LICENSE
├── README.md
└── requirements.txt
```

---

## 🚀 Quickstart

### 1 · Clone and set up the environment

```bash
git clone https://github.com/see0mar/cost-sensitive-telecom-retention.git
cd cost-sensitive-telecom-retention
python -m venv .venv
```

**Windows (PowerShell)**
```powershell
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

**macOS / Linux**
```bash
source .venv/bin/activate
pip install -r requirements.txt
```

### 2 · Generate the synthetic benchmark dataset

```bash
python scripts/generate_data.py
```

Writes `data/synthetic_telecom_retention_1m.parquet` (~30 MB compressed).  
Optional flags: `--n-samples 500000 --seed 0 --output data/custom.parquet`

### 3 · Run the notebook

```bash
jupyter notebook notebooks/01_pipeline_demo.ipynb
```

Or run the full pipeline programmatically:

```python
from src.data_processor  import TelemetryDataProcessor
from src.model_trainer   import RetentionModelTrainer
from src.cost_evaluator  import CostSensitiveOptimizer

# 1 · Load and split
processor = TelemetryDataProcessor("data/synthetic_telecom_retention_1m.parquet")
X_train, X_test, y_train, y_test = processor.prepare_splits(test_size=0.30)

# 2 · Train champion model (scale_pos_weight calculated dynamically)
trainer   = RetentionModelTrainer()
lgb_model = trainer.train_lightgbm(X_train, y_train)
y_probs   = trainer.predict_proba(lgb_model, X_test)

# 3 · Optimise the decision threshold
optimizer = CostSensitiveOptimizer(value_per_save_egp=500.0, cost_per_offer_egp=50.0)
best_threshold, results_df = optimizer.optimize_threshold(y_test.values, y_probs)

print(optimizer.financial_summary_table(results_df, best_threshold))
```

---

## 📊 Financial Simulation Ledger

Results from the LightGBM champion model on the 300,000-record held-out test partition.

| Financial Metric | Baseline (θ = 0.50) | Optimised (θ = 0.10) | Net Lift |
|---|---:|---:|---:|
| **Accounts targeted** | 25,574 | **105,350** | +79,776 |
| **Subscribers retained (TP)** | 18,340 | **35,242** | +16,902 |
| **Gross value preserved** | 9,170,000 EGP | **17,621,000 EGP** | +8,451,000 EGP |
| **Retention campaign cost** | 1,278,700 EGP | **5,267,500 EGP** | +3,988,800 EGP spend |
| **Net Campaign Value** | **7,891,300 EGP** | **12,353,500 EGP** | **+4,462,200 EGP (×1.6)** |

---

## 🛡️ Engineering Safeguards

1. **Temporal leakage controls** — retrospective aggregate trend metrics (`change_rev`, `avg3rev`, etc.) are excluded at load time by `TelemetryDataProcessor`.
2. **Dynamic imbalance weighting** — `scale_pos_weight` is computed from the training partition at runtime rather than hardcoded, ensuring correct weighting on any dataset.
3. **Stratified splitting** — both train and test partitions preserve the natural 4 : 1 class ratio.
4. **Sentinel encoding** — missing income-bracket values use −1.0 so the model can learn patterns associated with structural missingness.
5. **Memory optimisation** — feature vectors downcast to `int16` / `float32`; 1 M records fit in < 80 MB RAM.

---

## 🔬 Notebook Walkthrough

| Section | Content |
|---------|---------|
| 0 · Setup | Imports, visual config, path constants |
| 1 · Data | Generate / load synthetic dataset; class balance |
| 2 · EDA | Class distribution; support-call KDE (H1); feature correlation matrix |
| 3 · Split | Stratified 70/30 split via `TelemetryDataProcessor` |
| 4 · Benchmark | Train all three models; AUC comparison table |
| 5 · Diagnostics | Confusion matrices (all 3 models); ROC curve; classification report |
| 6 · Optimisation | NCV sweep; cost-curve plot; before/after financial table |
| 7 · SHAP | Global bar; directional beeswarm; local waterfall (per account) |
| 8 · Summary | Results table and key findings |

---

## 📜 License & Citation

Distributed under the MIT License. See [`LICENSE`](LICENSE) for details.

```bibtex
@software{ElTokhy2026CostSensitiveTelecom,
  author       = {Omar ElTokhy},
  title        = {Cost-Sensitive Machine Learning for Value Compression
                  and Revenue Preservation in Telecom},
  year         = {2026},
  publisher    = {GitHub},
  howpublished = {\url{https://github.com/see0mar/cost-sensitive-telecom-retention}}
}
```
