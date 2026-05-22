"""Diagram 8 — Data fault detection comes for free in SMC.

Two panels stacked:
  (a) Daily observations y_t.  Most fit the trend; one day has an anomalous
      spike (a data fault — e.g. a duplicate report batch).
  (b) The per-step predictive log-likelihood  log p(y_t | y_{1:t-1})  the
      PF computes at every step.  The anomalous day drops far below the
      baseline — a natural surprise score.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent.parent / "assets" / "data_fault_detection.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(11)


def _style():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "font.size": 10,
        }
    )


def main():
    _style()

    T = 40
    t = np.arange(T)

    # Trend: smooth outbreak ascent + decline.
    mu = 20 + 25 * np.exp(-((t - 22) ** 2) / 100)
    y = np.maximum(np.round(mu + rng.normal(0, 3.0, T)), 1)
    # Inject a data fault on day 14: triple-counted report.
    fault_day = 14
    y[fault_day] = int(mu[fault_day] * 3.5)

    # Surrogate per-step predictive log-likelihood.  Smaller absolute value
    # = less surprising.  Use a Gaussian-ish baseline + sharp dip at fault.
    log_pred = -0.5 * ((y - mu) / 3.0) ** 2 - 2.4
    log_pred = log_pred + rng.normal(0, 0.1, T)

    fig, (ax_top, ax_bot) = plt.subplots(
        2, 1, figsize=(11, 5.0), sharex=True,
        gridspec_kw={"hspace": 0.15, "height_ratios": [1.0, 0.7]},
    )

    # --- Top: observations ---
    ax_top.plot(t, mu, color="#1565C0", lw=1.2, alpha=0.6,
                label="model expected $\\mu_y(t)$")
    ax_top.scatter(t, y, s=40, color="k", zorder=3, label="observed $y_t$")
    # Highlight the fault
    ax_top.scatter([fault_day], [y[fault_day]], s=120, color="#d62728",
                   zorder=4, edgecolor="white", lw=1.5,
                   label=f"day {fault_day}: anomalous report")
    ax_top.set_ylabel("daily cases")
    ax_top.set_title("Daily observations — one report is far above the trend",
                     fontsize=11.5)
    ax_top.legend(loc="upper left", frameon=False, fontsize=9)

    # --- Bottom: predictive log-likelihood ---
    ax_bot.plot(t, log_pred, "o-", color="#2e7d32", lw=1.0, ms=4.5,
                label="$\\log p(y_t \\mid y_{1:t-1})$")
    ax_bot.scatter([fault_day], [log_pred[fault_day]], s=120, color="#d62728",
                   zorder=4, edgecolor="white", lw=1.5)
    ax_bot.axhline(np.median(log_pred), color="#2e7d32", ls=":", lw=0.8,
                   alpha=0.5, label="baseline (median)")
    # Annotate the dip
    ax_bot.annotate(
        f"  surprise: {log_pred[fault_day]:.1f}\n  ≪ baseline",
        xy=(fault_day, log_pred[fault_day]),
        xytext=(fault_day + 4, log_pred[fault_day] + 0.3),
        fontsize=10, color="#d62728",
        arrowprops=dict(arrowstyle="-|>", color="#d62728", lw=0.8),
    )
    ax_bot.set_ylabel("log predictive density")
    ax_bot.set_xlabel("day $t$")
    ax_bot.set_title("Per-step predictive log-likelihood — automatically flags the anomaly",
                     fontsize=11.5)
    ax_bot.legend(loc="lower left", frameon=False, fontsize=9, ncol=2)

    fig.suptitle(
        "Data fault detection — for free, every step",
        y=1.00, fontsize=13,
    )
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved → {OUT}")


if __name__ == "__main__":
    main()
