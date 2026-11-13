#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_panel_data.py
======================
Synthetic customer-month panel for a value-compression / churn DEMONSTRATION.

*** READ THIS FIRST ***********************************************************
Every relationship in this dataset is built into the generator. Results obtained
on it (AUC, financial lift, "hypothesis" plots) demonstrate that the PIPELINE
works end to end. They are NOT evidence that the hypotheses hold in any real
telecom base. The data is fictional and contains no information from any real
company.
*******************************************************************************

What is simulated
-----------------
Three latent customer groups (ground truth, stored only in customers.parquet
with the prefix `latent_` and NEVER to be used as model inputs):

  stable       never deteriorates (a small share show a temporary false-alarm
               episode that recovers, so the problem is not trivial)
  compressing  stays active, but revenue drops sharply in the last quarter
  churning     deteriorates, then leaves the network in the last quarter

Compressing and churning customers share ONE deterioration process: for 4-6
months before their event, support calls, dropped/blocked calls and overage
fees rise. Churners get a slightly stronger amplitude. This is the structure
the timeline comparison in the notebook is designed to examine.

Tables written
--------------
  panel_monthly.parquet  one row per active customer-month
                         (no rows after a customer churns)
  customers.parquet      one row per customer (static attributes)

Usage
-----
  python scripts/generate_panel_data.py
  python scripts/generate_panel_data.py --n-customers 50000 --validate
"""

import argparse
import os
import time

import numpy as np
import pandas as pd


def _sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


# ──────────────────────────────────────────────────────────────────────────────
# Generator
# ──────────────────────────────────────────────────────────────────────────────
def generate_panel(n_customers=200_000, n_months=12, seed=42):
    """Return (panel_df, customers_df). Events occur in the last 3 months."""
    if n_months < 10:
        raise ValueError("n_months must be >= 10 (needs a >=7-month observation window).")

    rng = np.random.default_rng(seed)
    N, T = int(n_customers), int(n_months)
    t = np.arange(1, T + 1)[None, :]                      # (1, T) month index

    # ── Static attributes ────────────────────────────────────────────────────
    tenure_start = rng.integers(1, 72, N)                 # tenure at month 1
    inc_raw = rng.choice([1., 2., 3., 4., 5., 6., np.nan], size=N,
                         p=[0.15, 0.20, 0.20, 0.15, 0.10, 0.05, 0.15])
    income = np.nan_to_num(inc_raw, nan=-1.0).astype(np.float32)   # -1.0 = missing sentinel
    inc_z = (np.where(np.isnan(inc_raw), 3.0, inc_raw) - 3.0) / 1.5

    # ── Latent group assignment (mild dependence on tenure / income) ─────────
    p_risk = _sigmoid(-1.10 - 0.012 * (tenure_start - 36) - 0.10 * inc_z)
    at_risk = rng.random(N) < p_risk
    is_churner = at_risk & (rng.random(N) < 0.48)
    is_comp = at_risk & ~is_churner
    segment = np.where(is_churner, "churning", np.where(is_comp, "compressing", "stable"))

    # Event month: churners leave in the last 3 months; compressors start in the
    # last 2 (so the full compression quarter is mostly inside the outcome window).
    event = np.zeros(N, dtype=np.int16)
    event[is_churner] = rng.choice([T - 2, T - 1, T], size=int(is_churner.sum()),
                                   p=[0.30, 0.35, 0.35])
    event[is_comp] = rng.choice([T - 2, T - 1], size=int(is_comp.sum()), p=[0.5, 0.5])
    lead = rng.integers(4, 7, N)                          # precursor length: 4..6 months

    # ── Shared deterioration process ────────────────────────────────────────
    amp = rng.lognormal(0.0, 0.25, N)
    amp = np.where(is_churner, amp * 1.15, amp)           # churners: slightly stronger
    quiet = at_risk & (rng.random(N) < 0.15)              # 15% have weak, hard-to-see signals
    amp = np.where(quiet, amp * 0.25, amp)

    d = np.zeros((N, T), dtype=np.float32)                # 0 = normal, 1 = peak deterioration
    ar = np.where(at_risk)[0]
    d[ar] = np.clip((t - (event[ar, None] - lead[ar, None])) / lead[ar, None], 0, 1)

    # false alarms: 12% of stable customers have a temporary 3-month blip that recovers
    stable_idx = np.where(~at_risk)[0]
    blip = stable_idx[rng.random(stable_idx.size) < 0.12]
    s = rng.integers(3, T - 3, blip.size)
    d[blip] = np.clip(1 - np.abs(t - (s[:, None] + 1)) / 2.0, 0, 1) * 0.6
    eff = d * amp[:, None]

    # ── Behavioural signals (customer-level base rates x deterioration) ──────
    q = rng.gamma(4, 1 / 4, N)                            # network-quality factor (higher = worse)
    u = rng.gamma(3, 1 / 3, N)                            # usage-intensity factor
    sup_base = 0.9 * np.sqrt(q) * rng.gamma(4, 1 / 4, N)
    drop_base = 3.0 * q
    ovr_base = 14.0 * u * rng.gamma(2, 1 / 2, N)
    dat_base = 8.0 * np.sqrt(u) * rng.gamma(2, 1 / 2, N)  # independent draw: NOT derived from overage

    support = rng.poisson(sup_base[:, None] * (1 + 2.2 * eff))
    dropped = rng.poisson(drop_base[:, None] * (1 + 1.0 * eff))
    ovr = (ovr_base[:, None] * (1 + 1.6 * eff) * rng.lognormal(0, 0.35, (N, T))
           * (rng.random((N, T)) < np.clip(0.60 + 0.25 * eff, 0, 0.95)))
    dat = (dat_base[:, None] * (1 + 1.2 * eff) * rng.lognormal(0, 0.40, (N, T))
           * (rng.random((N, T)) < np.clip(0.50 + 0.25 * eff, 0, 0.95)))

    # ── Revenue: plan revenue x seasonality x drop, plus overage charges ─────
    base_plan = 200.0 * np.exp(0.10 * inc_z + rng.normal(0, 0.35, N))
    season = 1 + 0.03 * np.sin(2 * np.pi * (t - 1) / 12)
    delta = np.where(is_comp, rng.uniform(0.40, 0.70, N),
                     np.where(is_churner, rng.uniform(0.10, 0.75, N), 0.0))
    start_drop = np.where(is_comp, event, event - rng.integers(1, 3, N))  # churners collapse 1-2 months before leaving
    f = 1 - delta[:, None] * (t >= start_drop[:, None])
    revenue = base_plan[:, None] * season * rng.lognormal(0, 0.05, (N, T)) * f + ovr + dat

    # ── Panel mask: no rows after a customer churns ─────────────────────────
    valid = np.ones((N, T), dtype=bool)
    valid[is_churner] = t < event[is_churner, None]

    cid = np.broadcast_to(np.arange(1, N + 1, dtype=np.int32)[:, None], (N, T))
    mth = np.broadcast_to(t, (N, T))
    ten = tenure_start[:, None] + (t - 1)

    panel = pd.DataFrame({
        "customer_id":            cid[valid],
        "month":                  mth[valid].astype(np.int8),
        "account_tenure_months":  ten[valid].astype(np.int16),
        "monthly_revenue":        np.round(revenue[valid], 2).astype(np.float32),
        "monthly_overage_fee":    np.round(ovr[valid], 2).astype(np.float32),
        "data_overage_fee":       np.round(dat[valid], 2).astype(np.float32),
        "dropped_blocked_calls":  dropped[valid].astype(np.int16),
        "customer_service_calls": support[valid].astype(np.int16),
    })

    customers = pd.DataFrame({
        "customer_id":              np.arange(1, N + 1, dtype=np.int32),
        "tenure_start_months":      tenure_start.astype(np.int16),
        "estimated_income_bracket": income,
        "churned_flag":             is_churner.astype(np.int8),
        "churn_month":              pd.array(np.where(is_churner, event, -1), dtype="int16"),  # -1 = did not churn
        # ground truth for validation / teaching only -- NEVER a model input
        "latent_segment":           pd.Categorical(segment),
        "latent_event_month":       pd.array(np.where(at_risk, event, -1), dtype="int16"),     # -1 = no event
    })
    return panel, customers


# ──────────────────────────────────────────────────────────────────────────────
# Validation report (reads the tables exactly as the user receives them)
# ──────────────────────────────────────────────────────────────────────────────
def validate(panel, customers, n_months=12):
    T = n_months
    N = len(customers)
    print("\n" + "=" * 72)
    print("VALIDATION REPORT")
    print("=" * 72)
    print(f"customers: {N:,} | panel rows: {len(panel):,} | months: {T}")
    print("\nlatent group shares:")
    print((customers["latent_segment"].value_counts(normalize=True) * 100).round(1).to_string())
    print("\nactive customers by month (should fall only in the last quarter):")
    print(panel.groupby("month")["customer_id"].nunique().to_string())

    # wide matrices
    pv = lambda c: panel.pivot(index="customer_id", columns="month", values=c).reindex(customers["customer_id"]).to_numpy()
    rev, sup, drp, ovr, dat = (pv(c) for c in ["monthly_revenue", "customer_service_calls",
                                                "dropped_blocked_calls", "monthly_overage_fee", "data_overage_fee"])

    # change_rev = last 3 observed months vs the 3 months before them
    last = np.where(customers["churned_flag"].to_numpy() == 1,
                    customers["churn_month"].to_numpy() - 1, T)           # last observed month (1-based)
    rows = np.arange(N)[:, None]
    cur = np.nanmean(rev[rows, (last[:, None] - np.array([1, 2, 3]))], axis=1)       # months last-0..last-2 (0-based idx = last-1..last-3)
    prv = np.nanmean(rev[rows, (last[:, None] - np.array([4, 5, 6]))], axis=1)
    change_rev = (cur - prv) / prv * 100
    comp = change_rev <= -25
    seg = customers["latent_segment"].astype(str).to_numpy()
    active = customers["churned_flag"].to_numpy() == 0
    print("\nshare with change_rev <= -25%  (label definition used in the notebook):")
    for sname in ["stable", "compressing", "churning"]:
        m = seg == sname
        print(f"  {sname:<12} {comp[m].mean() * 100:5.1f}%   (n={m.sum():,})")
    pos = comp[active].mean()
    print(f"  -> positive rate among ACTIVE customers: {pos:.1%}  (imbalance {(1 - pos) / pos:.1f} : 1)")

    # timeline alignment: months relative to the event
    print("\ntimeline check -- mean behaviour by months-to-event (events: churn month / compression onset;")
    print("stable customers get a random matched reference event):")
    rng = np.random.default_rng(0)
    ev = customers["latent_event_month"].to_numpy().copy()
    stable_m = seg == "stable"
    ev[stable_m] = rng.choice([T - 2, T - 1], size=int(stable_m.sum()))
    rel = np.arange(-6, 0)
    out = {}
    for name, mat in [("support_calls", sup), ("dropped_calls", drp), ("overage_fee", ovr), ("revenue", rev)]:
        prof = {}
        for sname in ["stable", "compressing", "churning"]:
            m = seg == sname
            prof[sname] = np.array([np.nanmean(mat[m, ev[m] + r - 1]) for r in rel])
        out[name] = prof
        r_cc = np.corrcoef(prof["compressing"], prof["churning"])[0, 1]
        r_cs = np.corrcoef(prof["compressing"], prof["stable"])[0, 1]
        print(f"  {name:<14} corr(compress,churn)={r_cc:+.2f}   corr(compress,stable)={r_cs:+.2f}   "
              f"month -1 level: stable {prof['stable'][-1]:.1f} | compress {prof['compressing'][-1]:.1f} | churn {prof['churning'][-1]:.1f}")

    # multicollinearity check on customer-level means (months 1-9)
    feats = pd.DataFrame({
        "overage":  np.nanmean(ovr[:, :9], axis=1), "data_overage": np.nanmean(dat[:, :9], axis=1),
        "dropped":  np.nanmean(drp[:, :9], axis=1), "support": np.nanmean(sup[:, :9], axis=1),
        "tenure":   customers["tenure_start_months"].to_numpy(), "income": customers["estimated_income_bracket"].to_numpy(),
    })
    c = feats.corr().where(~np.eye(6, dtype=bool)).abs().stack()
    a, b = c.idxmax()
    print(f"\nstrongest pairwise correlation among 9-month feature means: {a} <-> {b}  r = {feats.corr().loc[a, b]:+.2f}")
    print("=" * 72)


# ──────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser(description="Generate the synthetic customer-month panel.")
    ap.add_argument("--n-customers", type=int, default=200_000)
    ap.add_argument("--n-months", type=int, default=12)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out-dir", type=str, default="data")
    ap.add_argument("--validate", action="store_true", help="print the validation report")
    a = ap.parse_args()

    t0 = time.time()
    panel, customers = generate_panel(a.n_customers, a.n_months, a.seed)
    os.makedirs(a.out_dir, exist_ok=True)
    p1 = os.path.join(a.out_dir, "panel_monthly.parquet")
    p2 = os.path.join(a.out_dir, "customers.parquet")
    panel.to_parquet(p1, compression="snappy", index=False)
    customers.to_parquet(p2, compression="snappy", index=False)
    print(f"Wrote {len(panel):,} panel rows -> {p1} ({os.path.getsize(p1) / 1e6:.1f} MB)")
    print(f"Wrote {len(customers):,} customers -> {p2} ({os.path.getsize(p2) / 1e6:.1f} MB)  [{time.time() - t0:.1f}s]")
    if a.validate:
        validate(panel, customers, a.n_months)


if __name__ == "__main__":
    main()
