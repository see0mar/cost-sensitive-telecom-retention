#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
generate_data.py
Generates a 1-million-record synthetic telecom retention benchmark dataset.
Usage:
    python scripts/generate_data.py
    python scripts/generate_data.py --n-samples 500000 --output data/custom.parquet
"""

import argparse, os, time
import numpy as np
import pandas as pd


def generate_synthetic_telecom_dataset(
    n_samples: int = 1_000_000,
    random_seed: int = 42,
    output_path: str = "data/synthetic_telecom_retention_1m.parquet",
) -> pd.DataFrame:
    np.random.seed(random_seed)
    print(f"Generating {n_samples:,} synthetic records  |  seed={random_seed}")
    t0 = time.time()

    # ── Features ──────────────────────────────────────────────────────────
    account_tenure_months     = np.random.randint(1, 72, size=n_samples, dtype=np.int16)
    monthly_overage_fee_mean  = np.random.exponential(scale=14.2, size=n_samples).astype(np.float32)
    data_overage_fee_mean     = (monthly_overage_fee_mean * np.random.uniform(0.35, 0.85, n_samples)).astype(np.float32)
    dropped_blocked_calls_mean = np.random.poisson(lam=3.8, size=n_samples).astype(np.int16)

    # Realistic correlation: degraded network -> more support calls
    _noise = np.random.poisson(lam=0.8, size=n_samples).astype(np.float32)
    customer_service_calls_mean = (dropped_blocked_calls_mean * 0.25 + _noise).clip(0).astype(np.float32)

    income_raw = np.random.choice(
        [1., 2., 3., 4., 5., 6., np.nan], size=n_samples,
        p=[0.15, 0.20, 0.20, 0.15, 0.10, 0.05, 0.15],
    )
    estimated_income_bracket = np.nan_to_num(income_raw, nan=-1.0).astype(np.float32)

    # ── Latent non-linear tier signal (not exposed as a column) ───────────
    # LightGBM discovers the tier boundaries from raw overage; LR cannot.
    overage_tier = np.where(monthly_overage_fee_mean > 30.8, 3,
                   np.where(monthly_overage_fee_mean > 20.9, 2,
                   np.where(monthly_overage_fee_mean >  7.08, 1, 0))).astype(np.float32)

    # ── Target (logit) — ~20 % positive rate ──────────────────────────────
    logit = (
          0.055 * monthly_overage_fee_mean
        + 0.220 * customer_service_calls_mean
        + 0.100 * dropped_blocked_calls_mean
        - 0.030 * account_tenure_months
        + 0.750 * overage_tier          # H2: tier-based non-linearity
        - 3.80                          # intercept
    )
    prob  = 1.0 / (1.0 + np.exp(-logit))
    value_compressed_flag = np.random.binomial(1, prob).astype(np.int8)

    # ── Assemble & export ─────────────────────────────────────────────────
    df = pd.DataFrame({
        "customer_id":                  [f"CUST_{i:07d}" for i in range(1, n_samples + 1)],
        "account_tenure_months":         account_tenure_months,
        "monthly_overage_fee_mean":      np.round(monthly_overage_fee_mean, 2),
        "data_overage_fee_mean":         np.round(data_overage_fee_mean, 2),
        "dropped_blocked_calls_mean":    dropped_blocked_calls_mean,
        "customer_service_calls_mean":   customer_service_calls_mean.round(2),
        "estimated_income_bracket":      estimated_income_bracket,
        "value_compressed_flag":         value_compressed_flag,
    })

    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
    df.to_parquet(output_path, compression="snappy", index=False)

    elapsed  = time.time() - t0
    mem_mb   = df.memory_usage(deep=True).sum() / 1024**2
    pos_rate = value_compressed_flag.mean()
    print(f"  Done in {elapsed:.1f}s | RAM {mem_mb:.0f} MB | "
          f"Positive rate {pos_rate:.1%} | -> {output_path}")
    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--n-samples", type=int,  default=1_000_000)
    parser.add_argument("--seed",      type=int,  default=42)
    parser.add_argument("--output",    type=str,  default="data/synthetic_telecom_retention_1m.parquet")
    args = parser.parse_args()
    generate_synthetic_telecom_dataset(args.n_samples, args.seed, args.output)
