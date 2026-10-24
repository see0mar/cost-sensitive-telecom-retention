# -*- coding: utf-8 -*-
"""
src/cost_evaluator.py
---------------------
Asymmetric cost-benefit threshold optimiser.

Maximises the Net Campaign Value (NCV) function:

    NCV(θ) = TP(θ) × (V − C) − FP(θ) × C

where V is the gross value preserved per successful retention and C is
the cost of one proactive retention offer.

The optimal threshold θ* satisfies:
    θ* = C / V   (the break-even probability)
"""

from typing import Dict, Tuple

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np
import pandas as pd


class CostSensitiveOptimizer:
    """
    Scans the full probability space to find the threshold that maximises
    Net Campaign Value (NCV) under an asymmetric cost matrix.

    Parameters
    ----------
    value_per_save_egp : float
        Gross value preserved per correctly identified and retained subscriber.
    cost_per_offer_egp : float
        Direct cost of a single proactive retention offer.

    Examples
    --------
    >>> optimizer = CostSensitiveOptimizer(value_per_save_egp=500.0,
    ...                                     cost_per_offer_egp=50.0)
    >>> best_threshold, results_df = optimizer.optimize_threshold(y_test, y_probs)
    >>> optimizer.plot_ncv_curve(results_df, best_threshold)
    """

    def __init__(
        self,
        value_per_save_egp: float = 500.0,
        cost_per_offer_egp: float = 50.0,
    ):
        self.V = value_per_save_egp
        self.C = cost_per_offer_egp
        self.break_even_threshold = self.C / self.V  # theoretical θ*

    # ── Core calculation ────────────────────────────────────────────────────

    def evaluate_threshold(
        self,
        y_true: np.ndarray,
        y_probs: np.ndarray,
        threshold: float,
    ) -> Dict[str, float]:
        """
        Calculates exact business metrics at a specific decision threshold.

        Parameters
        ----------
        y_true  : Ground-truth binary labels.
        y_probs : Positive-class probability scores.
        threshold : Classification boundary in [0, 1].

        Returns
        -------
        dict with confusion matrix counts and all financial metrics.
        """
        preds = (y_probs >= threshold).astype(int)

        tp = int(np.sum((preds == 1) & (y_true == 1)))
        fp = int(np.sum((preds == 1) & (y_true == 0)))
        fn = int(np.sum((preds == 0) & (y_true == 1)))
        tn = int(np.sum((preds == 0) & (y_true == 0)))

        gross_saved     = tp * self.V
        campaign_cost   = (tp + fp) * self.C
        ncv             = gross_saved - campaign_cost
        uncaught_loss   = fn * self.V

        return {
            "threshold":            threshold,
            "targeted_accounts":    tp + fp,
            "true_positives":       tp,
            "false_positives":      fp,
            "false_negatives":      fn,
            "true_negatives":       tn,
            "gross_saved_egp":      gross_saved,
            "campaign_cost_egp":    campaign_cost,
            "net_campaign_value_egp": ncv,
            "uncaught_loss_egp":    uncaught_loss,
        }

    def optimize_threshold(
        self,
        y_true: np.ndarray,
        y_probs: np.ndarray,
        step: float = 0.01,
    ) -> Tuple[float, pd.DataFrame]:
        """
        Scans thresholds from 0.01 → 0.99 and returns the one with the
        highest Net Campaign Value.

        Parameters
        ----------
        y_true  : Ground-truth binary labels.
        y_probs : Positive-class probability scores.
        step    : Granularity of the threshold scan.

        Returns
        -------
        best_threshold : float — NCV-maximising threshold.
        results_df     : pd.DataFrame — one row per scanned threshold.
        """
        thresholds = np.arange(0.01, 1.00, step)
        results    = [self.evaluate_threshold(y_true, y_probs, t) for t in thresholds]
        results_df = pd.DataFrame(results)

        best_row       = results_df.loc[results_df["net_campaign_value_egp"].idxmax()]
        best_threshold = float(best_row["threshold"])

        print(
            f"\n  [Optimizer] ── Threshold Scan Results ─────────────────────────\n"
            f"  Break-even threshold (C/V = {self.C}/{self.V}): θ* = {self.break_even_threshold:.2f}\n"
            f"  Empirical NCV-maximising threshold:            θ  = {best_threshold:.2f}\n"
            f"  Max Net Campaign Value:    {best_row['net_campaign_value_egp']:>12,.0f} EGP\n"
            f"  Targeted accounts:         {best_row['targeted_accounts']:>12,.0f}\n"
            f"  True positives retained:   {best_row['true_positives']:>12,.0f}\n"
            f"  Campaign cost:             {best_row['campaign_cost_egp']:>12,.0f} EGP\n"
            f"  ──────────────────────────────────────────────────────────────"
        )
        return best_threshold, results_df

    # ── Visualisation ────────────────────────────────────────────────────────

    def plot_ncv_curve(
        self,
        results_df: pd.DataFrame,
        best_threshold: float,
        baseline_threshold: float = 0.50,
        ax: plt.Axes | None = None,
    ) -> plt.Axes:
        """
        Plots Net Campaign Value versus decision threshold, marking the
        default 0.50 baseline and the cost-optimised threshold.

        Parameters
        ----------
        results_df         : Output from optimize_threshold().
        best_threshold     : NCV-maximising threshold to annotate.
        baseline_threshold : Standard default (default: 0.50).
        ax                 : Optional existing Axes to draw on.

        Returns
        -------
        plt.Axes
        """
        if ax is None:
            _, ax = plt.subplots(figsize=(9, 4))

        ax.plot(
            results_df["threshold"],
            results_df["net_campaign_value_egp"],
            color="#15803D", linewidth=2.0, label="NCV(θ)"
        )
        ax.fill_between(
            results_df["threshold"],
            results_df["net_campaign_value_egp"],
            alpha=0.08, color="#15803D"
        )

        # Optimal threshold
        opt_row = results_df.loc[results_df["threshold"].sub(best_threshold).abs().idxmin()]
        ax.axvline(best_threshold, color="#15803D", linestyle="--", linewidth=1.5,
                   label=f"Optimised θ = {best_threshold:.2f}")
        ax.annotate(
            f"  {opt_row['net_campaign_value_egp']:,.0f} EGP",
            xy=(best_threshold, opt_row["net_campaign_value_egp"]),
            fontsize=9.5, color="#15803D", fontweight="bold",
        )

        # Baseline threshold
        base_row = results_df.loc[results_df["threshold"].sub(baseline_threshold).abs().idxmin()]
        ax.axvline(baseline_threshold, color="#6B7280", linestyle=":", linewidth=1.5,
                   label=f"Default θ = {baseline_threshold:.2f}")
        ax.annotate(
            f"  {base_row['net_campaign_value_egp']:,.0f} EGP",
            xy=(baseline_threshold, base_row["net_campaign_value_egp"]),
            fontsize=9.5, color="#6B7280",
        )

        ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda x, _: f"{x:,.0f}"))
        ax.set_xlabel("Decision Threshold (θ)", fontsize=11)
        ax.set_ylabel("Net Campaign Value (EGP)", fontsize=11)
        ax.set_title("Cost-Sensitive Threshold Optimisation", fontsize=13, fontweight="bold")
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.35)
        return ax

    def financial_summary_table(
        self,
        results_df: pd.DataFrame,
        best_threshold: float,
        baseline_threshold: float = 0.50,
    ) -> pd.DataFrame:
        """
        Returns a two-row comparison DataFrame (baseline vs. optimised).

        Parameters
        ----------
        results_df         : Output from optimize_threshold().
        best_threshold     : NCV-maximising threshold.
        baseline_threshold : Default standard threshold (0.50).

        Returns
        -------
        pd.DataFrame with one row per scenario.
        """
        def _row(label: str, threshold: float) -> dict:
            r = results_df.loc[
                results_df["threshold"].sub(threshold).abs().idxmin()
            ]
            return {
                "Scenario":                  label,
                "Threshold (θ)":             f"{threshold:.2f}",
                "Accounts Targeted":         f"{int(r['targeted_accounts']):,}",
                "True Positives (Retained)": f"{int(r['true_positives']):,}",
                "Gross Revenue Saved (EGP)": f"{r['gross_saved_egp']:,.0f}",
                "Campaign Cost (EGP)":       f"{r['campaign_cost_egp']:,.0f}",
                "Net Campaign Value (EGP)":  f"{r['net_campaign_value_egp']:,.0f}",
            }

        rows = [
            _row(f"Baseline   (θ = {baseline_threshold:.2f})", baseline_threshold),
            _row(f"Optimised  (θ = {best_threshold:.2f})",     best_threshold),
        ]
        return pd.DataFrame(rows).set_index("Scenario")
