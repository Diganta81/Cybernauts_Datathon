# FictiPay Customer Churn Prediction

End-to-end churn prediction pipeline for **FictiPay**, a synthetic mobile
wallet provider, built for the **Cybernauts 2026 Datathon (NSU CEC)**.
The pipeline ingests 200M+ transaction records and ~360M day-end balance
records via Dask, engineers 137 account-level features, and trains a
LightGBM + CatBoost ensemble to predict the probability that a customer
churns in April 2024.

**Final result:** Blended out-of-fold AUC-ROC **0.9846**, Precision@10%
**0.910**, Recall@10% **0.718**.

---

## Table of Contents

- [Problem Overview](#problem-overview)
- [Repository Structure](#repository-structure)
- [Pipeline Overview](#pipeline-overview)
- [Feature Engineering](#feature-engineering)
- [Modeling](#modeling)
- [Results](#results)
- [Business Recommendations](#business-recommendations)
- [How to Run](#how-to-run)
- [Limitations & Future Work](#limitations--future-work)

---

## Problem Overview

FictiPay offers three account types — **Customer**, **Merchant**, and
**Biller** — and provides transaction history (Jan–Mar 2024), day-end
balances, and KYC/demographic data for ~850,000 customer accounts. The
task is to predict, for each test account, the **probability of churn**
in the unobserved April 2024 window.

| | |
|---|---|
| Observation window | 2024-01-01 → 2024-03-31 (strictly `< 2024-04-01`) |
| Prediction window | 2024-04-01 → 2024-04-30 (label, not provided) |
| Evaluation metric | AUC-ROC on a float `CHURN_PROB ∈ [0, 1]` (Avg. Precision, Precision@10%, Recall@10% also reported) |
| Train set | 595,000 accounts, 12.678% churn rate (75,435 / 595,000) |
| Test set | 255,000 accounts |
| Raw data | 200M+ transaction rows, ~360M balance rows, ~1.96 GB (parquet) |

---

## Repository Structure

```
.
├── notebook.ipynb            # End-to-end pipeline (feature engineering -> training -> submission)
├── predictions.csv           # Final CHURN_PROB submission (float, [0,1])
├── features.md               # Catalogue of all 137 engineered features, with rationale
├── report.pdf / report.md    # Model performance report (data handling, feature quality,
│                              #   class imbalance, CV results, hyperparameter tuning)
├── presentation.pdf/.pptx    # 5-slide business summary (decision rule, interventions, impact)
├── explainability/
│   ├── insights.md           # SHAP analysis, top features, leakage checklist
│   ├── shap_summary.png
│   ├── shap_importance_bar.png
│   └── shap_dependence_top.png
└── README.md                 # This file
```

---

## Pipeline Overview

All ingestion and row-level aggregation is done with **Dask** to keep peak
memory bounded by the *output* size (~850K accounts), not the raw input
size (200M+ / 360M rows):

1. **Ingest** — `dask.dataframe.read_parquet` over partitioned transaction
   and balance files. A temporal filter (`TRX_DATETIME` / `DATE` <
   `CUTOFF = 2024-04-01`) is applied immediately, before any `.compute()`,
   so Dask can push the predicate down to the partition level.
2. **Feature engineering** — every feature is produced via a
   `groupby(...).agg(...).compute()` (or `.size()` / `.nunique()` /
   `.min()`), collapsing to one row per account before leaving Dask. The
   weekly-binned activity block consolidates what would otherwise be three
   separate 7d/30d/90d passes into a **single** groupby/pivot.
3. **Matrix construction** — two-tier missing-value strategy (see below),
   `inf` → `NaN` conversion before imputation.
4. **Cross-validation** — `StratifiedKFold(n_splits=5, random_state=42)`,
   with **out-of-fold REGION target encoding** computed independently per
   fold (no leakage).
5. **Modeling** — LightGBM + CatBoost trained per fold, blended via a
   weighted average searched over OOF predictions.
6. **Explainability** — SHAP summary/importance/dependence plots on a
   5,000-row sample.
7. **Submission** — blended test predictions, clipped to `[0, 1]`,
   reordered to match `sample_submission.csv`.

---

## Feature Engineering

**137 features** across 18 categories (full rationale per group in
[`features.md`](./features.md)):

| Category | # Features |
|---|---|
| Transaction window aggregates (7d/30d/90d) | 15 |
| Fine-grained recent activity (1d/2d/3d/5d) | 4 |
| Overall recency | 1 |
| Transaction-type mix (90d) | 10 |
| Network breadth | 2 |
| P2P receipt activity | 1 |
| Per-type, per-direction recency | 10 |
| Weekly-binned activity & trend | 34 |
| Bill payment regularity | 5 |
| Balance level & distribution | 8 |
| Balance trend (full + 14-day) | 2 |
| Balance drain timing | 1 |
| Zero-balance activity | 1 |
| Recent vs. early balance | 2 |
| Trailing dormancy streaks | 2 |
| KYC / demographic | 3 |
| Interaction / ratio features | 16 |
| Missing-indicator flags | 19 |

**Missing-value strategy (two-tier):**
- **99 count-like columns** (counts, sums, recencies, flags) → `-999`
  sentinel ("missing = no recorded activity," a meaningful state for tree
  splits).
- **19 ratio/slope/momentum columns** → median-imputed (train-only
  median) + paired `_isnan` flag. `inf`/`-inf` from zero-denominator
  divisions are converted to `NaN` first.
- **`REGION`** → out-of-fold target encoding (per-fold churn rate by
  region, global-mean fallback for unseen categories).

---

## Modeling

| | LightGBM | CatBoost |
|---|---|---|
| Type | `gbdt`, leaf-wise | symmetric-tree, ordered boosting |
| learning_rate | 0.02 | 0.03 |
| n_estimators / iterations | 6000 | 4000 |
| num_leaves / depth | 255 | 8 |
| min_child_samples | 30 | — |
| reg_alpha / l2_leaf_reg | 0.1 | 3.0 |
| scale_pos_weight | 6.888 | 6.888 |
| early_stopping_rounds | 200 | 200 |

**Class imbalance** (12.678% churn, ~7:1) is handled via
`scale_pos_weight ≈ 6.888` rather than over/under-sampling — this keeps
the output probabilities calibrated to the true base rate (important
since the evaluation metric is a float `CHURN_PROB`) and avoids creating
duplicate account-level feature vectors.

**Cross-validation:** `StratifiedKFold(n_splits=5, random_state=42)`. Each
fold computes its own REGION target encoding from training-fold labels
only, applied to that fold's validation set and the test set.

**Ensembling:** A weighted blend `w · oof_lgb + (1-w) · oof_cb` is
searched over `w ∈ [0, 1]` on OOF predictions; the best weight is applied
to test predictions.

---

## Results

### Per-fold AUC

| Fold | LightGBM AUC | CatBoost AUC |
|---|---|---|
| 1 | 0.98424 | 0.98423 |
| 2 | 0.98435 | 0.98437 |
| 3 | 0.98502 | 0.98496 |
| 4 | 0.98460 | 0.98464 |
| 5 | 0.98456 | 0.98459 |

### Blended out-of-fold metrics

| Metric | Value |
|---|---|
| LightGBM OOF AUC | 0.98444 |
| CatBoost OOF AUC | 0.98454 |
| **Blended OOF AUC** (w = 0.45 LGB) | **0.98460** |
| Blended OOF Average Precision | 0.91982 |
| Precision @ Top 10% | 0.90975 |
| Recall @ Top 10% | 0.71757 |

### Top SHAP features (by mean \|SHAP\|)

| Feature | Mean \|SHAP\| |
|---|---|
| `trx_count_7d` | 0.945 |
| `trx_count_30d_v2` | 0.759 |
| `days_since_last_trx` | 0.462 |
| `trx_count_30d` | 0.376 |
| `trx_count_5d` | 0.228 |
| `trx_count_90d` | 0.189 |
| `activity_momentum` | 0.092 |
| `bal_slope_14d` | 0.063 |
| `weeks_since_active` | 0.060 |

**Takeaway:** churn risk is dominated by **recent transaction
volume/recency** (7–30 day activity collapse, days since last
transaction), with **14-day balance trend** and **weeks of inactivity**
as secondary confirming signals. Full SHAP plots and a leakage checklist
are in [`explainability/insights.md`](./explainability/insights.md).

---

## Business Recommendations

**Decision rule:** flag the top 10% of accounts by predicted
`CHURN_PROB` for retention outreach.

- **Precision@10% = 0.910** — 9 in 10 flagged accounts genuinely churn.
- **Recall@10% = 0.718** — this captures ~72% of all churners.
- **Lift = Precision@10% / base rate ≈ 0.910 / 0.1268 ≈ 7.2x** — the
  flagged list is ~7x more concentrated with churners than a random
  sample of the same size.

**Targeted interventions:**

| Risk Signal | Recommended Action |
|---|---|
| Sharp drop in 7–30 day transaction count | Personalized re-engagement push + cashback/fee-waiver on next transaction |
| Long stretch since last transaction | Time-limited reactivation offer (bonus on next CashIn/P2P) |
| Negative 14-day balance trend | Proactive incentive (savings nudge, cashback) tied to maintaining balance |
| Missed expected bill payment | Reminder notification + auto-pay enrollment prompt |

See [`presentation.pdf`](./presentation.pdf) for the full 5-slide
business summary.

---

## How to Run

The pipeline is designed for the Kaggle competition environment (dual-T4,
`/kaggle/input/competitions/bkash-presents-nsucec-datathon/public/`).

```bash
pip install dask pandas numpy scipy scikit-learn lightgbm catboost shap matplotlib
```

Open `notebook.ipynb` and run all cells top to bottom. Expected outputs:

- `predictions.csv` — final submission (float `CHURN_PROB`, one row per
  test account, ordered to match `sample_submission.csv`)
- `explainability/shap_summary.png`, `shap_importance_bar.png`,
  `shap_dependence_top.png`

**Note on AUC-ROC submissions:** `CHURN_PROB` must remain a **continuous
score** in `[0, 1]` — AUC-ROC depends only on the *relative ranking* of
scores across all test accounts. Rounding/binarizing predictions to 0/1
collapses the ranking to a single threshold and will substantially reduce
the AUC (the smooth ROC curve degenerates to one operating point). If a
binary flag list is needed for a separate deliverable (e.g. the top-10%
decision rule above), derive it from `predictions.csv` without modifying
the submitted probabilities themselves.

Runtime: ingestion + feature engineering is the dominant cost (single
pass per aggregation group over the filtered Dask frames); CV training is
5 folds × 2 models with early stopping (typically 180–500 boosting
rounds per fold).

---

## Limitations & Future Work

- **Synthetic dataset** — results should be re-validated against
  production data before deployment.
- **Performance plateau** — OOF AUC has converged around 0.985 across
  several feature-engineering iterations (additional recency/dormancy
  features yielded gains of ≤0.001 each), suggesting this aggregate
  feature representation is near its practical ceiling.
- **Hard cases** (from error analysis): accounts that are *active with
  rising balances* yet still churn, and accounts that go *quiet with
  falling balances* yet don't — candidates for **last-transaction
  anomaly** features (e.g., an unusually large final withdrawal relative
  to the account's own history).
- **Next steps:** sequence-based models over raw transaction order
  (e.g., transformer/RNN embeddings), last-transaction anomaly features,
  and threshold/calibration analysis for production deployment.

---

## Acknowledgements

Built for the **Cybernauts 2026 Datathon**, hosted by **NSU CEC** in
partnership with **bKash**.
