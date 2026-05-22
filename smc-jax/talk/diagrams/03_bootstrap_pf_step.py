"""Diagram 3 — One step of the bootstrap particle filter.

Three side-by-side panels:
  (1) Particles at time t-1.
  (2) PROPOSE under the transition: arrows showing each particle's drift,
      proposed positions at time t.
  (3) WEIGHT: particle sizes scaled by the likelihood evaluated at y_t.
  (4) RESAMPLE: particles drawn in proportion to weight; some duplicated,
      some dropped.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

OUT = Path(__file__).parent.parent / "assets" / "bootstrap_pf_step.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(7)


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

    n = 18
    x_prev = rng.normal(0.0, 1.0, n)

    # Transition: drift right + diffusion.
    drift = 1.0
    sigma_q = 0.5
    x_prop = x_prev + drift + sigma_q * rng.normal(size=n)

    # Likelihood at y_t: a peak at x = 1.6 with sd 0.45.
    y_t = 1.6
    sigma_y = 0.45
    w = norm.pdf(x_prop, y_t, sigma_y)
    w = w / w.sum()

    # Resample: multinomial draw indices.
    idx = rng.choice(n, size=n, p=w)
    x_post = x_prop[idx]
    # Jitter visually slightly so duplicates show up.
    x_post_disp = x_post + 0.0  # leave them stacked vertically per-x value below

    fig, axes = plt.subplots(1, 4, figsize=(15, 3.0), sharey=True)

    # --- Panel 1: particles at t-1 ---
    ax = axes[0]
    ax.scatter(x_prev, np.zeros(n), s=80, color="#1f77b4",
               alpha=0.75, edgecolor="white", lw=1.0)
    ax.set_title("Particles at $t - 1$", fontsize=11)
    ax.set_xlim(-3, 5)
    ax.set_yticks([])
    ax.axhline(0, color="#bbb", lw=0.6)

    # --- Panel 2: propose under transition ---
    ax = axes[1]
    for xi, xj in zip(x_prev, x_prop):
        ax.annotate("", xy=(xj, 0.0), xytext=(xi, 0.0),
                    arrowprops=dict(arrowstyle="-|>",
                                    color="#1f77b4", alpha=0.55, lw=0.9,
                                    shrinkA=2, shrinkB=2))
    ax.scatter(x_prev, np.zeros(n), s=40, color="#1f77b4",
               alpha=0.3, edgecolor="white", lw=0.6)
    ax.scatter(x_prop, np.zeros(n), s=80, color="#1f77b4",
               alpha=0.85, edgecolor="white", lw=1.0)
    ax.set_title("PROPOSE under $p(x_t \\mid x_{t-1})$", fontsize=11)
    ax.set_xlim(-3, 5)
    ax.axhline(0, color="#bbb", lw=0.6)

    # --- Panel 3: weight by likelihood ---
    ax = axes[2]
    # Background: likelihood curve over x.
    xs = np.linspace(-3, 5, 300)
    lik = norm.pdf(xs, y_t, sigma_y) / norm.pdf(y_t, y_t, sigma_y)
    ax.fill_between(xs, 0, lik * 0.35, color="#d62728", alpha=0.12,
                    linewidth=0)
    ax.plot(xs, lik * 0.35, color="#d62728", lw=1.2, alpha=0.7,
            label="$p(y_t \\mid x_t)$")
    # Particles sized by normalised weight.
    sizes = 30 + 600 * w
    ax.scatter(x_prop, np.zeros(n), s=sizes, color="#d62728",
               alpha=0.6, edgecolor="white", lw=1.0)
    ax.axvline(y_t, color="#d62728", ls="--", lw=0.8, alpha=0.7)
    ax.text(y_t + 0.05, 0.45, "$y_t$", color="#d62728", fontsize=11)
    ax.set_title("WEIGHT by $p(y_t \\mid x_t)$", fontsize=11)
    ax.set_xlim(-3, 5)
    ax.set_ylim(-0.1, 0.5)
    ax.axhline(0, color="#bbb", lw=0.6)

    # --- Panel 4: resample ---
    ax = axes[3]
    # Stack duplicate draws vertically so the audience sees survivors and copies.
    seen = {}
    ys = []
    for xv in x_post:
        c = seen.get(round(xv, 3), 0)
        ys.append(c * 0.07)
        seen[round(xv, 3)] = c + 1
    ax.scatter(x_post, ys, s=80, color="#2e7d32",
               alpha=0.85, edgecolor="white", lw=1.0)
    # Faintly show the un-resampled cloud for comparison.
    ax.scatter(x_prop, np.zeros(n) - 0.15, s=30, color="#999",
               alpha=0.5, edgecolor="white", lw=0.6)
    ax.text(-2.8, -0.15, "post-weighting", color="#666", fontsize=8.5,
            va="center")
    ax.text(-2.8, 0.05, "post-resampling", color="#2e7d32", fontsize=9,
            va="center")
    ax.set_title("RESAMPLE in proportion to weight", fontsize=11)
    ax.set_xlim(-3, 5)
    ax.set_ylim(-0.3, 0.6)
    ax.axhline(0, color="#bbb", lw=0.6)
    ax.set_xlabel("latent state $x$", fontsize=10)

    for a in axes:
        a.spines["left"].set_visible(False)

    fig.suptitle(
        "One step of the bootstrap particle filter: propose, weight, resample",
        y=1.04, fontsize=12.5,
    )
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved → {OUT}")


if __name__ == "__main__":
    main()
