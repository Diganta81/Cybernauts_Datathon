"""
Stage 6 – Hyperparameter Tuning with Optuna (TPE sampler)
==========================================================
Runs a 50-trial Bayesian optimisation study over LightGBM hyperparameters.

Search space:
    num_leaves        [63, 511]    log-uniform int
    learning_rate     [0.005, 0.1] log-uniform float
    feature_fraction  [0.5, 1.0]  uniform float
    bagging_fraction  [0.5, 1.0]  uniform float
    min_child_samples [10, 100]   log-uniform int
    reg_alpha         [1e-3, 10]  log-uniform float
    reg_lambda        [1e-3, 10]  log-uniform float

Each trial is evaluated with 3-fold StratifiedKFold on the compact
18-feature matrix so trials are fast (~20s each).

    pip install optuna
    python report/tuning/optuna_search.py

Outputs (written to report/tuning/):
    optuna_convergence.png  – per-trial AUC + best-so-far + default baseline
    optuna_importance.png   – fANOVA hyperparameter importance
    optuna_results.csv      – all 50 trials: params + AUC
    best_params.txt         – best configuration found
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
import optuna
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold

optuna.logging.set_verbosity(optuna.logging.WARNING)

warnings.filterwarnings("ignore")

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
sys.path.insert(0, os.path.join(ROOT, "report"))
from build_compact_features import build_compact_features

OUT_DIR = THIS_DIR
N_CV = 3
N_TRIALS = 50
SEED = 42
SCALE_POS_WEIGHT = 519_565 / 75_435

# ── Build compact features ────────────────────────────────────────────────────
X, y, FEAT_COLS = build_compact_features(ROOT)


def eval_params(params, n_splits=N_CV):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=SEED)
    aucs = []
    for tr, val in skf.split(X, y):
        m = lgb.LGBMClassifier(**params)
        m.fit(
            X[tr],
            y[tr],
            eval_set=[(X[val], y[val])],
            eval_metric="auc",
            callbacks=[lgb.early_stopping(50, verbose=False), lgb.log_evaluation(-1)],
        )
        aucs.append(roc_auc_score(y[val], m.predict_proba(X[val])[:, 1]))
    return float(np.mean(aucs))


# ── Default baseline ──────────────────────────────────────────────────────────
DEFAULT_PARAMS = dict(
    objective="binary",
    metric="auc",
    boosting_type="gbdt",
    num_leaves=31,
    learning_rate=0.1,
    n_estimators=100,
    feature_fraction=1.0,
    bagging_fraction=1.0,
    bagging_freq=0,
    min_child_samples=20,
    reg_alpha=0.0,
    reg_lambda=0.0,
    scale_pos_weight=SCALE_POS_WEIGHT,
    random_state=SEED,
    n_jobs=-1,
    verbose=-1,
)
print("\nEvaluating sklearn-default LightGBM params (5-fold) …")
default_auc = eval_params(DEFAULT_PARAMS, n_splits=5)
print(f"  Default 5-fold AUC: {default_auc:.5f}")


# ── Optuna objective ──────────────────────────────────────────────────────────
def objective(trial):
    params = dict(
        objective="binary",
        metric="auc",
        boosting_type="gbdt",
        num_leaves=trial.suggest_int("num_leaves", 63, 511, log=True),
        learning_rate=trial.suggest_float("learning_rate", 0.005, 0.1, log=True),
        n_estimators=2000,
        feature_fraction=trial.suggest_float("feature_fraction", 0.5, 1.0),
        bagging_fraction=trial.suggest_float("bagging_fraction", 0.5, 1.0),
        bagging_freq=5,
        min_child_samples=trial.suggest_int("min_child_samples", 10, 100, log=True),
        reg_alpha=trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
        reg_lambda=trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
        scale_pos_weight=SCALE_POS_WEIGHT,
        random_state=SEED,
        n_jobs=-1,
        verbose=-1,
    )
    return eval_params(params)


print(f"\nRunning Optuna TPE search: {N_TRIALS} trials × {N_CV}-fold CV …")
t0 = time.time()
study = optuna.create_study(
    direction="maximize", sampler=optuna.samplers.TPESampler(seed=SEED)
)
study.optimize(objective, n_trials=N_TRIALS, show_progress_bar=True)
print(f"Search complete in {(time.time() - t0) / 60:.1f} min")

best_params = study.best_params
best_auc_cv = study.best_value
print(f"\nBest trial ({N_CV}-fold) AUC: {best_auc_cv:.5f}")
for k, v in best_params.items():
    print(f"  {k:25s} = {v}")

# Re-evaluate best params with 5-fold
best_params_full = dict(
    objective="binary",
    metric="auc",
    boosting_type="gbdt",
    n_estimators=6000,
    scale_pos_weight=SCALE_POS_WEIGHT,
    bagging_freq=5,
    random_state=SEED,
    n_jobs=-1,
    verbose=-1,
    **best_params,
)
print(f"\nRe-evaluating best params (5-fold) …")
best_auc_5fold = eval_params(best_params_full, n_splits=5)
print(f"  Best Optuna 5-fold AUC: {best_auc_5fold:.5f}")
print(f"  Improvement over default: {best_auc_5fold - default_auc:+.5f}")

# ── Save results ──────────────────────────────────────────────────────────────
with open(os.path.join(OUT_DIR, "best_params.txt"), "w") as f:
    f.write(f"Default LightGBM 5-fold OOF AUC   : {default_auc:.5f}\n")
    f.write(f"Optuna best trial ({N_CV}-fold) AUC   : {best_auc_cv:.5f}\n")
    f.write(f"Optuna best params 5-fold AUC     : {best_auc_5fold:.5f}\n")
    f.write(
        f"Improvement over default          : {best_auc_5fold - default_auc:+.5f}\n\n"
    )
    f.write("Best hyperparameters:\n")
    for k, v in best_params.items():
        f.write(f"  {k:25s} = {v}\n")

trials_df = study.trials_dataframe()
param_cols = [c for c in trials_df.columns if c.startswith("params_")]
out_cols = ["number", "value"] + param_cols
trials_df[out_cols].rename(columns={"number": "trial", "value": "auc"}).to_csv(
    os.path.join(OUT_DIR, "optuna_results.csv"), index=False
)

# ── Plot 1: Convergence curve ─────────────────────────────────────────────────
tnums = [t.number + 1 for t in study.trials]
taucs = [t.value for t in study.trials]
best_so_far = pd.Series(taucs).cummax().tolist()

fig, ax = plt.subplots(figsize=(13, 5))
ax.scatter(
    tnums,
    taucs,
    alpha=0.55,
    s=25,
    color="#3498DB",
    label=f"Trial AUC ({N_CV}-fold)",
    zorder=3,
)
ax.plot(
    tnums,
    best_so_far,
    color="#E74C3C",
    linewidth=2.2,
    label="Best-so-far AUC",
    zorder=4,
)
ax.axhline(
    default_auc,
    color="#7F8C8D",
    linestyle="--",
    linewidth=1.5,
    label=f"sklearn-default AUC = {default_auc:.5f}",
)
ax.axhline(
    best_auc_cv,
    color="#2ECC71",
    linestyle=":",
    linewidth=1.5,
    label=f"Optuna best (3-fold) = {best_auc_cv:.5f}",
)

# Annotate improvement
mid_trial = tnums[-1] * 0.6
ax.annotate(
    f"+{best_auc_cv - default_auc:.4f} vs default",
    xy=(tnums[best_so_far.index(best_auc_cv)], best_auc_cv),
    xytext=(mid_trial, (default_auc + best_auc_cv) / 2 + 0.0005),
    fontsize=9,
    color="darkred",
    fontweight="bold",
    arrowprops=dict(arrowstyle="->", color="darkred"),
)

ax.set_xlabel("Optuna Trial Number", fontsize=11)
ax.set_ylabel(f"OOF AUC-ROC ({N_CV}-fold CV per trial)", fontsize=11)
ax.set_title(
    f"Optuna TPE Hyperparameter Search — LightGBM  ({N_TRIALS} trials)\n"
    "Search space: num_leaves [63–511], lr [0.005–0.1], "
    "feature_frac, bagging_frac, min_child_samples, reg_α/λ",
    fontsize=10,
    fontweight="bold",
)
ax.legend(fontsize=9)
ax.set_xlim(0, max(tnums) + 1)
plt.tight_layout()
plt.savefig(
    os.path.join(OUT_DIR, "optuna_convergence.png"), dpi=150, bbox_inches="tight"
)
plt.close()
print(f"\nConvergence plot saved.")

# ── Plot 2: Hyperparameter importance (fANOVA) ───────────────────────────────
try:
    importance = optuna.importance.get_param_importances(study)
    imp_df = pd.DataFrame(
        list(importance.items()), columns=["param", "importance"]
    ).sort_values("importance")
    fig, ax = plt.subplots(figsize=(9, 5))
    bars = ax.barh(imp_df["param"], imp_df["importance"], color="#3498DB", alpha=0.8)
    for bar, val in zip(bars, imp_df["importance"]):
        ax.text(
            bar.get_width() + 0.003,
            bar.get_y() + bar.get_height() / 2,
            f"{val:.3f}",
            va="center",
            fontsize=9,
        )
    ax.set_xlabel(
        "Hyperparameter Importance (fANOVA — fraction of AUC variance explained)",
        fontsize=9,
    )
    ax.set_title(
        "Optuna Hyperparameter Importance\n"
        "Which parameters had the most impact on AUC during the 50-trial search?",
        fontsize=10,
        fontweight="bold",
    )
    plt.tight_layout()
    plt.savefig(
        os.path.join(OUT_DIR, "optuna_importance.png"), dpi=150, bbox_inches="tight"
    )
    plt.close()
    print("Importance plot saved.")
except Exception as e:
    print(f"Importance plot skipped: {e}")

print(f"\n{'=' * 60}")
print(f"Default LightGBM 5-fold AUC  : {default_auc:.5f}")
print(f"Optuna best 5-fold AUC       : {best_auc_5fold:.5f}")
print(f"Improvement                  : {best_auc_5fold - default_auc:+.5f}")
print("\n✅  Stage 6 Optuna tuning complete.")
