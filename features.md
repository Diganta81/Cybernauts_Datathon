# FictiPay Churn Prediction — Feature Engineering Documentation

**Total features fed to the model: 137**
- 116 base engineered features (grouped below)
- 19 missing-indicator flags (`_isnan`) for ratio/slope-type features
- 1 out-of-fold target-encoded `REGION` feature (computed per CV fold)

All features are computed strictly from data with `days_ago >= 0`, i.e.
`TRX_DATETIME` / `DATE` < `CUTOFF` (2024-04-01). This filter is applied at
the Dask partition level immediately after ingestion, before any
aggregation, so no record from the prediction window (2024-04-01 to
2024-04-30) can enter any feature.

---

## 1. Transaction Window Aggregates (15 features)

`trx_count_{7d,30d,90d}`, `trx_amt_sum_{7d,30d,90d}`,
`trx_amt_mean_{7d,30d,90d}`, `trx_amt_std_{7d,30d,90d}`,
`trx_amt_max_{7d,30d,90d}`

**Rationale:** Multi-horizon volume and value statistics capture both the
current activity level and how it compares across recent vs. longer-term
windows — a customer whose 7-day volume has collapsed relative to their
90-day baseline is behaving very differently from one with stable activity.

## 2. Fine-Grained Recent Activity (4 features)

`trx_count_{1d,2d,3d,5d}`

**Rationale:** Aggregate windows like 7d can mask very recent silence;
day-level counts let the model detect activity stopping abruptly in the
final days before the prediction window — the period closest to the
churn-defining month.

## 3. Overall Recency (1 feature)

`days_since_last_trx`

**Rationale:** Single strongest univariate dormancy signal in the dataset
(univariate AUC ≈ 0.95) — the number of days since the customer's most
recent transaction of any type.

## 4. Transaction-Type Mix, 90d (10 features)

`trxtype_{P2P,MerchantPay,BillPay,CashIn,CashOut}_90d` (raw counts) and
`trxtype_{...}_90d_ratio` (share of the customer's own total)

**Rationale:** A shift in *which* channels a customer uses (e.g. dropping
MerchantPay while still doing CashIn) can precede full churn and is
invisible to total-activity counts alone.

## 5. Network Breadth (2 features)

`unique_counterparties_{30d,90d}`

**Rationale:** Shrinking transaction network breadth — fewer distinct
people/merchants the customer interacts with — is a common
disengagement precursor independent of raw transaction count.

## 6. P2P Receipt Activity (1 feature)

`p2p_recv_count_90d`

**Rationale:** Captures incoming social/peer activity (money received via
P2P), distinct from the customer's own outgoing behavior.

## 7. Per-Type, Per-Direction Recency (10 features)

`days_since_last_{TYPE}_sent` and `days_since_last_{TYPE}_recv`
for TYPE in {P2P, MerchantPay, BillPay, CashIn, CashOut}

**Rationale:** Aggregate recency hides *which specific channel* a customer
stopped using first; churn often manifests channel-by-channel (e.g. bill
payments stop weeks before P2P activity does) rather than all at once.

## 8. Weekly-Binned Activity & Trend (34 features)

`trx_count_w0..w12`, `trx_sum_w0..w12` (13 weekly bins covering the full
90-day window), plus derived `trx_count_{7d,30d,90d}_v2`,
`trx_sum_{7d,30d,90d}_v2`, `activity_decay_slope`, `weeks_since_active`

**Rationale:** A 13-point weekly time series lets the model learn a
*trend* (via `activity_decay_slope`, the linear-regression slope across
weekly counts) rather than only a static level. `weeks_since_active`
gives a discrete "how many weeks has this account been completely quiet"
signal. Both are computed from a single Dask groupby/pivot pass — the 7d/
30d/90d "_v2" aggregates are then derived from these bins in pandas with
no extra Dask compute.

## 9. Bill Payment Regularity (5 features)

`bill_m0`, `bill_m1`, `bill_m2`, `bill_missed_recent`,
`days_since_last_billpay`

**Rationale:** Recurring bill payments indicate ongoing use of the wallet
as a financial utility. `bill_missed_recent` flags accounts that paid a
bill in the prior 30-day period (`bill_m1`) but not in the most recent
30-day period (`bill_m0`) — a sharp behavioral break in an otherwise
routine pattern.

## 10. Balance Level & Distribution (8 features)

`bal_mean`, `bal_std`, `bal_min`, `bal_max`, `bal_first`, `bal_last`,
`bal_cv`, `bal_drain_ratio`

**Rationale:** Captures both the typical balance held over the window and
how volatile/depleted it has become relative to the start of the window
(`bal_drain_ratio = bal_last / bal_first`).

## 11. Balance Trend (2 features)

`bal_slope` (full 90-day window), `bal_slope_14d` (last 14 days only)

**Rationale:** A separate short-window slope is far more sensitive to
*recent* acceleration in balance changes than a 90-day slope, which can be
dominated by older, less relevant trend. This was consistently the
single highest-gain feature for LightGBM across all CV folds.

## 12. Balance Drain Timing (1 feature)

`days_since_drain` — days since the balance first fell to ≤10% of its own
historical maximum (-1 if it never did)

**Rationale:** Encodes *when* a drain event happened, not just whether
the current balance is low — a customer who drained their account three
months ago and has been stable since is different from one whose balance
collapsed last week.

## 13. Zero-Balance Activity (1 feature)

`zero_bal_days` — total number of days with balance ≤ 0 over the window

## 14. Recent vs. Early Balance Level (2 features)

`bal_30d_avg` (mean balance, last 30 days), `bal_early_avg` (mean balance,
days 60-90 ago)

**Rationale:** Direct comparison of the most recent month's average
balance against the account's earlier baseline.

## 15. Trailing Dormancy Streaks (2 features)

`trailing_zero_bal_days`, `trailing_near_zero_bal_days` (balance ≤1% of
the account's own historical max)

**Rationale:** A fully vectorized run-length encoding (via `cumcount` /
`cummax`, no per-group `.apply`) of *consecutive* dormancy ending at the
cutoff date. This distinguishes "currently in an ongoing dormant streak"
from "was dormant at some point but recovered" — the latter should not be
treated as high churn risk.

## 16. KYC / Demographic (3 features)

`account_age_days`, `GENDER_enc` (label-encoded), `REGION` (out-of-fold
target-encoded → appended as `REGION_target_enc`, 1 feature, inside each
CV fold)

**Rationale:** Tenure (`account_age_days`) contextualizes activity level —
the same transaction count means something different for a 6-month-old
account vs. a 5-year-old one. `REGION` target encoding captures regional
baseline churn-rate differences without high-cardinality one-hot
expansion; it is computed separately within each fold using only that
fold's training labels to avoid leakage.

## 17. Interaction / Ratio Features (16 features)

`activity_momentum`, `channel_shift_ratio`, `is_silent_{3d,7d,30d}`,
`bal_recent_vs_early`, `p2p_social_ratio`, `sum_shift_ratio_7_30`,
`sum_shift_ratio_30_90`, `mean_amt_shift_30_90`,
`counterparty_shift_30_90`, `activity_per_tenure`, `bal_last_vs_30davg`,
`dormant_combo`, `is_fully_dormant`, `is_fully_dormant_30d`

**Rationale:** Ratios normalize for account-level baselines — a
high-volume merchant-like customer and a low-volume customer can both be
"slowing down" but at very different absolute scales, and a raw count
difference wouldn't capture that equally for both. The combined dormancy
flags (`is_fully_dormant`, `is_fully_dormant_30d`) jointly require *both*
no recent transactions *and* a sustained near-zero/zero balance, reducing
false positives from accounts that are simply low-activity by nature but
still financially active.

## 18. Missing-Indicator Flags (19 features)

`{ratio/slope/momentum column}_isnan` for each of the 19 ratio-type
features identified by name pattern (`ratio`, `slope`, `momentum`, `_cv`,
`shift`, `_vs_`)

**Rationale:** Ratio features are undefined (originally `inf`/`NaN`) when
a denominator is zero — e.g. no transactions in the comparison window.
Rather than letting median imputation silently absorb this, an explicit
flag lets the model treat "ratio undefined because there's no baseline
activity at all" as its own signal, distinct from "ratio happens to equal
the median value." All `inf`/`-inf` values are first converted to `NaN`
before this flag and the median imputation are applied, so no infinite
values reach the model.

---

## Missing-Value Strategy Summary

| Column type | Count | Strategy |
|---|---|---|
| Count-like (counts, sums, recencies, flags) | 99 | `-999` sentinel — "no recorded activity" is itself informative for tree splits |
| Ratio/slope/momentum-type | 19 | Median imputation (computed on train only) + paired `_isnan` flag |
| `REGION` | 1 | Out-of-fold target encoding, global-mean fallback for unseen categories |

**Final feature matrix:** 137 columns × 595,000 train rows / 255,000 test
rows. Churn rate: 12.678% (75,435 / 595,000), matching the competition
README's stated 519,565 / 75,435 overall split.
