"""
Stage 4 – Class Imbalance: Empirical Comparison
================================================
Compares three class-imbalance handling strategies on a common feature
matrix extracted from the competition data:

    (A) scale_pos_weight  — class-weighted loss (production choice)
    (B) RandomUnderSampling — randomly discard majority-class rows to 1:1
    (C) SMOTE             — synthetic minority oversampling (5-NN)

All three use identical LightGBM hyperparameters, 5-fold StratifiedKFold,
and OOF metric reporting for a fair comparison.

    python report/imbalance_comparison/compare_methods.py

Outputs (written to report/imbalance_comparison/):
    imbalance_comparison.png  – grouped bar chart: OOF AUC, AP, Prec@10%, Recall@10%
    comparison_table.csv      – full numeric results table
"""

import os
import sys
import time
import warnings

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import lightgbm as lgb
import matplotlib.pyplot as plt
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold

warnings.filterwarnings("ignore")

# Add repo root so we can import the shared utility
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "report"))
from build_compact_features import build_compact_features

try:
    from imblearn.over_sampling import SMOTE

    HAS_IMBLEARN = True
except ImportError:
    HAS_IMBLEARN = False
    print(
        "WARNING: imbalanced-learn not installed. Install with: pip install imbalanced-learn"
    )

OUT_DIR = THIS_DIR
N_SPLITS = 5
SEED = 42
SCALE_POS_WEIGHT = 519_565 / 75_435  # ≈ 6.888

# ── Build compact features ────────────────────────────────────────────────────
X, y, FEAT_COLS = build_compact_features(ROOT)

# ── Shared LightGBM params ────────────────────────────────────────────────────
BASE_PARAMS = dict(
    objective="binary",
    metric="auc",
    boosting_type="gbdt",
    num_leaves=127,
    learning_rate=0.05,
    n_estimators=3000,
    feature_fraction=0.8,
    bagging_fraction=0.8,
    bagging_freq=5,
    min_child_samples=30,
    reg_alpha=0.1,
    reg_lambda=1.0,
    random_state=SEED,
    n_jobs=-1,
    verbose=-1,
)


def prec_at_k(yt, yp, k=0.10):
    n = int(np.ceil(len(yp) * k))
    return yt[np.argsort(yp)[::-1][:n]].mean()


def rec_at_k(yt, yp, k=0.10):
    n = int(np.ceil(len(yp) * k))
    return yt[np.argsort(yp)[::-1][:n]].sum() / max(yt.sum(), 1)


def run_cv(X, y, params, label):
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof = np.zeros(len(y))
    fold_aucs = []
    t0 = time.time()
    for fold, (tr, val) in enumerate(skf.split(X, y), 1):
        m = lgb.LGBMClassifier(**params)
        m.fit(
            X[tr],
            y[tr],
            eval_set=[(X[val], y[val])],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)],
        )
        oof[val] = m.predict_proba(X[val])[:, 1]
        auc = roc_auc_score(y[val], oof[val])
        fold_aucs.append(auc)
        print(f"  {label} fold {fold}: AUC={auc:.5f}  iter={m.best_iteration_}")
    oa = roc_auc_score(y, oof)
    ap = average_precision_score(y, oof)
    p10 = prec_at_k(y, oof)
    r10 = rec_at_k(y, oof)
    print(
        f"  {label}  OOF AUC={oa:.5f}  AP={ap:.5f}  Prec@10%={p10:.5f}  "
        f"Recall@10%={r10:.5f}  ({time.time() - t0:.0f}s)"
    )
    return dict(
        method=label,
        oof_auc=oa,
        avg_precision=ap,
        precision_at_10=p10,
        recall_at_10=r10,
        fold_auc_std=np.std(fold_aucs),
    )


results = []

# ── A: scale_pos_weight ───────────────────────────────────────────────────────
print("\n── Method A: scale_pos_weight ──────────────────────────────────────────")
results.append(
    run_cv(
        X, y, {**BASE_PARAMS, "scale_pos_weight": SCALE_POS_WEIGHT}, "scale_pos_weight"
    )
)

# ── B: Random Undersampling ───────────────────────────────────────────────────
print("\n── Method B: Random Undersampling ──────────────────────────────────────")
skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
oof_us = np.zeros(len(y))
fold_aucs_us = []
t0 = time.time()
for fold, (tr, val) in enumerate(skf.split(X, y), 1):
    maj_idx = np.where(y[tr] == 0)[0]
    min_idx = np.where(y[tr] == 1)[0]
    rng = np.random.RandomState(SEED + fold)
    sampled = rng.choice(maj_idx, size=len(min_idx), replace=False)
    us_idx = np.concatenate([min_idx, sampled])
    rng.shuffle(us_idx)
    m = lgb.LGBMClassifier(**BASE_PARAMS)
    m.fit(
        X[tr][us_idx],
        y[tr][us_idx],
        eval_set=[(X[val], y[val])],
        eval_metric="auc",
        callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)],
    )
    oof_us[val] = m.predict_proba(X[val])[:, 1]
    auc = roc_auc_score(y[val], oof_us[val])
    fold_aucs_us.append(auc)
    print(
        f"  Undersampling fold {fold}: AUC={auc:.5f}  iter={m.best_iteration_}  "
        f"train={len(us_idx):,} (from {len(tr):,})"
    )

oa = roc_auc_score(y, oof_us)
ap = average_precision_score(y, oof_us)
print(
    f"  Undersampling  OOF AUC={oa:.5f}  AP={ap:.5f}  "
    f"Prec@10%={prec_at_k(y, oof_us):.5f}  Recall@10%={rec_at_k(y, oof_us):.5f}  "
    f"({time.time() - t0:.0f}s)"
)
results.append(
    dict(
        method="random_undersampling",
        oof_auc=oa,
        avg_precision=ap,
        precision_at_10=prec_at_k(y, oof_us),
        recall_at_10=rec_at_k(y, oof_us),
        fold_auc_std=np.std(fold_aucs_us),
    )
)

# ── C: SMOTE ──────────────────────────────────────────────────────────────────
if HAS_IMBLEARN:
    print("\n── Method C: SMOTE ─────────────────────────────────────────────────────")
    skf = StratifiedKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    oof_sm = np.zeros(len(y))
    fold_aucs_sm = []
    t0 = time.time()
    for fold, (tr, val) in enumerate(skf.split(X, y), 1):
        X_res, y_res = SMOTE(random_state=SEED + fold, k_neighbors=5).fit_resample(
            X[tr], y[tr]
        )
        m = lgb.LGBMClassifier(**BASE_PARAMS)
        m.fit(
            X_res,
            y_res,
            eval_set=[(X[val], y[val])],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(100, verbose=False), lgb.log_evaluation(-1)],
        )
        oof_sm[val] = m.predict_proba(X[val])[:, 1]
        auc = roc_auc_score(y[val], oof_sm[val])
        fold_aucs_sm.append(auc)
        print(
            f"  SMOTE fold {fold}: AUC={auc:.5f}  iter={m.best_iteration_}  "
            f"train={len(X_res):,} (from {len(tr):,})"
        )
    oa = roc_auc_score(y, oof_sm)
    ap = average_precision_score(y, oof_sm)
    print(
        f"  SMOTE  OOF AUC={oa:.5f}  AP={ap:.5f}  "
        f"Prec@10%={prec_at_k(y, oof_sm):.5f}  Recall@10%={rec_at_k(y, oof_sm):.5f}  "
        f"({time.time() - t0:.0f}s)"
    )
    results.append(
        dict(
            method="SMOTE",
            oof_auc=oa,
            avg_precision=ap,
            precision_at_10=prec_at_k(y, oof_sm),
            recall_at_10=rec_at_k(y, oof_sm),
            fold_auc_std=np.std(fold_aucs_sm),
        )
    )
else:
    print("\nSMOTE skipped (imbalanced-learn not installed).")

# ── Summary table & chart ─────────────────────────────────────────────────────
res_df = pd.DataFrame(results)
res_df.to_csv(os.path.join(OUT_DIR, "comparison_table.csv"), index=False)
print(f"\n{'=' * 70}")
print("COMPARISON RESULTS")
print(res_df.to_string(index=False))

metrics = ["oof_auc", "avg_precision", "precision_at_10", "recall_at_10"]
labels = ["OOF AUC-ROC", "Avg Precision", "Precision@10%", "Recall@10%"]
n_m = len(res_df)
x = np.arange(len(metrics))
w = 0.8 / n_m

fig, ax = plt.subplots(figsize=(13, 6))
colors = ["#2ECC71", "#E74C3C", "#3498DB"]
for i, (_, row) in enumerate(res_df.iterrows()):
    vals = [row[m] for m in metrics]
    bars = ax.bar(
        x + i * w, vals, w, label=row["method"], color=colors[i % 3], alpha=0.85
    )
    for bar, val in zip(bars, vals):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.003,
            f"{val:.4f}",
            ha="center",
            va="bottom",
            fontsize=8,
            fontweight="bold",
        )

ax.set_xticks(x + w * (n_m - 1) / 2)
ax.set_xticklabels(labels, fontsize=11)
ax.set_ylabel("Score", fontsize=11)
ax.set_title(
    "Class-Imbalance Strategy Comparison\n"
    "18-feature compact matrix · 5-fold StratifiedKFold · identical LightGBM params",
    fontsize=11,
    fontweight="bold",
)
ax.legend(fontsize=10)
ax.set_ylim(0.7, 1.05)

# Draw winner annotation
winner = res_df.loc[res_df["oof_auc"].idxmax()]
ax.annotate(
    f"Best: {winner['method']}\n(AUC {winner['oof_auc']:.5f})",
    xy=(x[0] + winner.name * w, winner["oof_auc"]),
    xytext=(x[0] + winner.name * w - 0.5, winner["oof_auc"] + 0.03),
    fontsize=8,
    color="darkgreen",
    arrowprops=dict(arrowstyle="->", color="darkgreen"),
)

plt.tight_layout()
plt.savefig(
    os.path.join(OUT_DIR, "imbalance_comparison.png"), dpi=150, bbox_inches="tight"
)
plt.close()
print(f"\nChart saved: {OUT_DIR}/imbalance_comparison.png")
print("\n✅  Stage 4 comparison complete.")
