"""
Stage 3 – Feature Quality & Pareto Analysis
=============================================
Generates distribution plots, skewness table, Pareto-80/20 chart,
and log-transform comparison for key features.

Outputs (written to report/pareto_analysis/):
    feature_distributions.png    – histograms for 12 key features (active accts only)
    log_transform_comparison.png – raw vs log1p for heavy-tailed count/sum features
    pareto_80_20.png             – Pareto chart: cumulative % of transactions by account rank
    zero_inflation.png           – zero / inactive fraction per feature
    skewness_table.csv           – skewness, kurtosis, p99 for every feature
"""

import os
import warnings

import matplotlib
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
from scipy import stats

warnings.filterwarnings("ignore")

THIS_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(THIS_DIR, "..", ".."))
TRX_DIR = os.path.join(ROOT, "transactions")
BAL_DIR = os.path.join(ROOT, "dayend_balance")
TRAIN_CSV = os.path.join(ROOT, "train_labels.csv")
OUT_DIR = THIS_DIR

CUTOFF = pd.Timestamp("2024-04-01")
REF_DATE = CUTOFF - pd.Timedelta(days=1)

train = pd.read_csv(TRAIN_CSV)
TRAIN_SET = set(train["ACCOUNT_ID"])
print(f"Train accounts: {len(TRAIN_SET):,}  |  Churn rate: {train['CHURN'].mean():.3%}")

# ── Read transactions (read only needed columns, filter early) ────────────────
print("Reading transactions (pyarrow, columns subset) …")
TRX_FILES = sorted(
    [os.path.join(TRX_DIR, f) for f in os.listdir(TRX_DIR) if f.endswith(".parquet")]
)
BAL_FILES = sorted(
    [os.path.join(BAL_DIR, f) for f in os.listdir(BAL_DIR) if f.endswith(".parquet")]
)

trx_chunks = []
for fpath in TRX_FILES:
    print(f"  {os.path.basename(fpath)} …", end=" ", flush=True)
    df_raw = pd.read_parquet(
        fpath,
        columns=["TRX_DATETIME", "SRC_ACCOUNT", "DST_ACCOUNT", "TRX_AMT", "TRX_TYPE"],
    )
    df_raw["TRX_DATETIME"] = pd.to_datetime(df_raw["TRX_DATETIME"])
    df_raw = df_raw[df_raw["TRX_DATETIME"] < CUTOFF]
    df_raw["days_ago"] = (REF_DATE - df_raw["TRX_DATETIME"]).dt.days
    df_raw = df_raw[df_raw["days_ago"] >= 0]
    # Filter to train accounts only
    df_raw = df_raw[df_raw["SRC_ACCOUNT"].isin(TRAIN_SET)]
    trx_chunks.append(df_raw)
    print(f"{len(df_raw):,} rows (train src)")

trx = pd.concat(trx_chunks, ignore_index=True)
print(f"Total transaction rows (train SRC): {len(trx):,}")

# ── Compute transaction features ─────────────────────────────────────────────
print("\nComputing transaction features …")


def window_agg(df, days, suffix):
    sub = df[df["days_ago"] <= days]
    g = (
        sub.groupby("SRC_ACCOUNT")["TRX_AMT"]
        .agg(["count", "sum", "mean", "std"])
        .reset_index()
    )
    g.columns = [
        "ACCOUNT_ID",
        f"trx_count_{suffix}",
        f"trx_amt_sum_{suffix}",
        f"trx_amt_mean_{suffix}",
        f"trx_amt_std_{suffix}",
    ]
    return g


f90 = window_agg(trx, 90, "90d")
f30 = window_agg(trx, 30, "30d")
f7 = window_agg(trx, 7, "7d")

recency = (
    trx.groupby("SRC_ACCOUNT")["days_ago"]
    .min()
    .reset_index()
    .rename(columns={"SRC_ACCOUNT": "ACCOUNT_ID", "days_ago": "days_since_last_trx"})
)

cp30 = (
    trx[trx["days_ago"] <= 30]
    .groupby("SRC_ACCOUNT")["DST_ACCOUNT"]
    .nunique()
    .reset_index()
    .rename(
        columns={
            "SRC_ACCOUNT": "ACCOUNT_ID",
            "DST_ACCOUNT": "unique_counterparties_30d",
        }
    )
)

trx_per_acct_90 = (
    trx[trx["days_ago"] <= 90]
    .groupby("SRC_ACCOUNT")
    .size()
    .reset_index(name="total_trx_90d")
    .rename(columns={"SRC_ACCOUNT": "ACCOUNT_ID"})
)

# ── Read balance features ─────────────────────────────────────────────────────
print("Reading balance data …")
bal_chunks = []
for fpath in BAL_FILES:
    print(f"  {os.path.basename(fpath)} …", end=" ", flush=True)
    b = pd.read_parquet(fpath, columns=["ACCOUNT_ID", "DATE", "AVAILABLE_BALANCE"])
    b["DATE"] = pd.to_datetime(b["DATE"])
    b = b[b["DATE"] < CUTOFF]
    b = b[b["ACCOUNT_ID"].isin(TRAIN_SET)]
    bal_chunks.append(b)
    print(f"{len(b):,} rows")

bal = pd.concat(bal_chunks, ignore_index=True)
print(f"Total balance rows (train): {len(bal):,}")

bst = (
    bal.groupby("ACCOUNT_ID")["AVAILABLE_BALANCE"]
    .agg(["mean", "std", "min", "max"])
    .reset_index()
    .rename(
        columns={
            "mean": "bal_mean",
            "std": "bal_std",
            "min": "bal_min",
            "max": "bal_max",
        }
    )
)
bst["bal_cv"] = bst["bal_std"] / (bst["bal_mean"].abs() + 1e-9)

# ── Merge ─────────────────────────────────────────────────────────────────────
df = train[["ACCOUNT_ID", "CHURN"]].copy()
for fdf in [f90, f30, f7, recency, cp30, bst, trx_per_acct_90]:
    df = df.merge(fdf, on="ACCOUNT_ID", how="left")
print(f"Feature matrix: {df.shape}")

FEAT_COLS = [
    "trx_count_30d",
    "trx_count_7d",
    "trx_count_90d",
    "trx_amt_sum_90d",
    "trx_amt_mean_90d",
    "trx_amt_std_90d",
    "days_since_last_trx",
    "unique_counterparties_30d",
    "bal_mean",
    "bal_std",
    "bal_cv",
    "bal_min",
]

# ── PLOT 1: Feature distributions ────────────────────────────────────────────
print("\nGenerating distribution plots …")
fig, axes = plt.subplots(3, 4, figsize=(22, 15))
axes = axes.flatten()
skew_rows = []

for i, feat in enumerate(FEAT_COLS):
    ax = axes[i]
    vals = df[feat].dropna().values
    active = vals[np.isfinite(vals)]
    inactive_pct = df[feat].isna().mean() * 100

    if len(active) == 0:
        ax.set_visible(False)
        continue

    sk = float(stats.skew(active))
    ku = float(stats.kurtosis(active))
    p99 = float(np.percentile(active, 99))
    med = float(np.median(active))

    skew_rows.append(
        {
            "feature": feat,
            "n_active": len(active),
            "inactive_pct": round(inactive_pct, 1),
            "mean": round(float(np.mean(active)), 2),
            "median": round(med, 2),
            "skewness": round(sk, 2),
            "excess_kurtosis": round(ku, 2),
            "p99": round(p99, 2),
            "max": round(float(active.max()), 2),
        }
    )

    clipped = active[active <= p99]
    color = "#E74C3C" if abs(sk) > 3 else "#3498DB"
    ax.hist(clipped, bins=60, color=color, alpha=0.75, edgecolor="none")
    ax.axvline(
        med, color="black", linestyle="--", linewidth=1.2, label=f"Median={med:.1f}"
    )
    ax.set_title(
        f"{feat}\nskew={sk:.2f}  inactive={inactive_pct:.1f}%",
        fontsize=8.5,
        fontweight="bold",
    )
    ax.legend(fontsize=7)
    ax.set_xlabel("Value (clipped at p99)", fontsize=7)
    ax.set_ylabel("Count", fontsize=7)
    ax.tick_params(labelsize=7)
    if abs(sk) > 3:
        ax.set_facecolor("#FFF3F3")

plt.suptitle(
    "Feature Distributions — Active Accounts Only  (NaN = no activity in window)\n"
    "Red background = |skewness| > 3 (heavy-tailed Pareto-like distribution)",
    fontsize=12,
    y=1.01,
    fontweight="bold",
)
plt.tight_layout()
out1 = os.path.join(OUT_DIR, "feature_distributions.png")
plt.savefig(out1, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {out1}")

# ── PLOT 2: Log-transform comparison ────────────────────────────────────────
LOG_FEATS = [
    "trx_count_30d",
    "trx_amt_sum_90d",
    "unique_counterparties_30d",
    "trx_amt_std_90d",
]
fig, axes = plt.subplots(2, 4, figsize=(22, 9))

for col, feat in enumerate(LOG_FEATS):
    vals = df[feat].dropna().values
    active = vals[vals > 0]
    sk_raw = float(stats.skew(active))
    sk_log = float(stats.skew(np.log1p(active)))
    p99 = np.percentile(active, 99)

    ax_raw = axes[0, col]
    ax_raw.hist(
        active[active <= p99], bins=60, color="#E74C3C", alpha=0.75, edgecolor="none"
    )
    ax_raw.set_title(f"{feat}\nRaw  skew={sk_raw:.2f}", fontsize=8.5, fontweight="bold")
    ax_raw.set_xlabel("Raw value (p99 clip)", fontsize=8)
    ax_raw.set_ylabel("Count", fontsize=8)
    ax_raw.tick_params(labelsize=7)

    ax_log = axes[1, col]
    ax_log.hist(
        np.log1p(active), bins=60, color="#2ECC71", alpha=0.75, edgecolor="none"
    )
    ax_log.set_title(
        f"log1p({feat})\nskew={sk_log:.2f}  (Δ={abs(sk_raw - sk_log):.1f}↓)",
        fontsize=8.5,
        fontweight="bold",
    )
    ax_log.set_xlabel("log1p(value)", fontsize=8)
    ax_log.set_ylabel("Count", fontsize=8)
    ax_log.tick_params(labelsize=7)

plt.suptitle(
    "Raw vs log1p Distributions — Heavy-Tailed (Pareto) Features\n"
    "log1p reduces skewness dramatically; not applied to final features because "
    "LightGBM/CatBoost are invariant to monotone transforms,\n"
    "but would be required for logistic regression or neural network baselines.",
    fontsize=10,
    y=1.02,
)
plt.tight_layout()
out2 = os.path.join(OUT_DIR, "log_transform_comparison.png")
plt.savefig(out2, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {out2}")

# ── PLOT 3: Pareto 80/20 ─────────────────────────────────────────────────────
print("Generating Pareto 80/20 chart …")
per_acct = trx_per_acct_90.sort_values("total_trx_90d", ascending=False).reset_index(
    drop=True
)
grand_total = per_acct["total_trx_90d"].sum()
cum_pct = per_acct["total_trx_90d"].cumsum() / grand_total * 100
n = len(per_acct)
acct_pct = np.arange(1, n + 1) / n * 100

fig, ax1 = plt.subplots(figsize=(12, 6))
step = max(1, n // 2000)
ax1.bar(
    acct_pct[::step],
    per_acct["total_trx_90d"].values[::step],
    width=0.1,
    color="#3498DB",
    alpha=0.45,
    label="Txn count per account",
)
ax1.set_xlabel("Cumulative % of accounts (sorted highest→lowest activity)", fontsize=10)
ax1.set_ylabel("Transactions per account (90d)", fontsize=10)

ax2 = ax1.twinx()
ax2.plot(
    acct_pct,
    cum_pct.values,
    color="#E74C3C",
    linewidth=2.5,
    label="Cumulative % of all transactions",
)
ax2.set_ylabel("Cumulative % of total transactions", fontsize=10)
ax2.yaxis.set_major_formatter(mtick.PercentFormatter())

idx_80 = int(np.searchsorted(cum_pct.values, 80))
acct_pct_80 = acct_pct[min(idx_80, len(acct_pct) - 1)]
ax2.axhline(80, color="gray", linestyle="--", linewidth=1.2)
ax2.axvline(acct_pct_80, color="gray", linestyle="--", linewidth=1.2)
ax2.annotate(
    f"Top {acct_pct_80:.1f}% of accounts\n→ 80% of transactions",
    xy=(acct_pct_80, 80),
    xytext=(acct_pct_80 + 8, 55),
    fontsize=9,
    color="darkred",
    arrowprops=dict(arrowstyle="->", color="darkred"),
)

lines1, labels1 = ax1.get_legend_handles_labels()
lines2, labels2 = ax2.get_legend_handles_labels()
ax1.legend(lines1 + lines2, labels1 + labels2, loc="lower right", fontsize=9)
plt.title(
    "Pareto Distribution of Transaction Activity (90-day window)\n"
    "A small fraction of high-activity accounts drives the majority of transactions",
    fontsize=11,
    fontweight="bold",
)
plt.tight_layout()
out3 = os.path.join(OUT_DIR, "pareto_80_20.png")
plt.savefig(out3, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {out3}")

# ── PLOT 4: Zero-inflation / sparsity ────────────────────────────────────────
print("Generating zero-inflation chart …")
zero_pcts = [(feat, df[feat].isna().mean() * 100) for feat in FEAT_COLS]
zero_df = pd.DataFrame(zero_pcts, columns=["feature", "inactive_pct"]).sort_values(
    "inactive_pct"
)

fig, ax = plt.subplots(figsize=(11, 7))
colors = ["#E74C3C" if v > 50 else "#3498DB" for v in zero_df["inactive_pct"]]
bars = ax.barh(zero_df["feature"], zero_df["inactive_pct"], color=colors, alpha=0.82)
ax.axvline(50, color="gray", linestyle="--", linewidth=1, label="50% threshold")
for bar, val in zip(bars, zero_df["inactive_pct"]):
    ax.text(
        bar.get_width() + 0.4,
        bar.get_y() + bar.get_height() / 2,
        f"{val:.1f}%",
        va="center",
        fontsize=9,
    )
ax.set_xlabel(
    "% of train accounts with no activity in this window (zero-inflated)", fontsize=10
)
ax.set_title(
    "Zero-Inflation / Sparsity by Feature\n"
    "Red = >50% of accounts inactive (sentinel strategy: -999 for tree models, _isnan flag for ratios)",
    fontsize=10,
    fontweight="bold",
)
ax.legend(fontsize=9)
plt.tight_layout()
out4 = os.path.join(OUT_DIR, "zero_inflation.png")
plt.savefig(out4, dpi=150, bbox_inches="tight")
plt.close()
print(f"  Saved: {out4}")

# ── Skewness table ────────────────────────────────────────────────────────────
skew_df = pd.DataFrame(skew_rows).sort_values("skewness", ascending=False)
csv_out = os.path.join(OUT_DIR, "skewness_table.csv")
skew_df.to_csv(csv_out, index=False)
print(f"\nSkewness table:")
print(skew_df.to_string(index=False))

print(f"\n{'=' * 60}")
print(
    f"Pareto 80/20 breakpoint: top {acct_pct_80:.1f}% of accounts → 80% of transactions"
)
print(
    f"Most skewed feature: {skew_df.iloc[0]['feature']} (skew={skew_df.iloc[0]['skewness']})"
)
print(
    f"Most sparse feature: {zero_df.sort_values('inactive_pct', ascending=False).iloc[0]['feature']}"
)
print(f"\n✅  All Stage 3 artifacts saved to: {OUT_DIR}")
