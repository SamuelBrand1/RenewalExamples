"""Diagram 6 — Liu-West shrink-jitter geometry.

Two-panel:
  (a) particle cloud with arrows pointing each particle toward the weighted
      mean — shrinkage step.
  (b) the shrunk cloud, jittered by adding ε ~ N(0, h²V).

Formula caption: θ_i' = a·θ_i + (1 − a)·m + ε_i,  ε_i ~ N(0, h²V),  a = √(1 − h²).

Shrinkage cancels the variance inflation of the jitter, preserving mean and
covariance of the cloud.  (Implementation: src/smc_renewal/pf/liu_west.py.)
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent.parent / "assets" / "liu_west_geometry.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(2)


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

    n = 40
    mean = np.array([1.0, 0.5])
    cov = np.array([[1.4, 0.6], [0.6, 0.9]])
    L = np.linalg.cholesky(cov)
    theta = mean + rng.normal(size=(n, 2)) @ L.T

    # Liu-West parameters.
    h = 0.25
    a = np.sqrt(1 - h ** 2)

    theta_shrunk = a * theta + (1 - a) * mean
    eps = rng.normal(size=(n, 2)) @ (h * L.T)
    theta_jittered = theta_shrunk + eps

    fig, axes = plt.subplots(1, 2, figsize=(12, 5.4), sharey=True)

    # --- Panel (a): shrink ---
    ax = axes[0]
    ax.scatter(theta[:, 0], theta[:, 1],
               s=70, color="#1f77b4", alpha=0.7,
               edgecolor="white", lw=0.8, label="θ-particle (pre)")
    # Mean point
    ax.scatter([mean[0]], [mean[1]], s=180, marker="X",
               color="#d62728", edgecolor="white", lw=1.0, zorder=4,
               label="weighted mean  m")
    # Arrows from each particle to the shrunk position.
    for (xs, ys), (xe, ye) in zip(theta, theta_shrunk):
        ax.annotate(
            "",
            xy=(xe, ye), xytext=(xs, ys),
            arrowprops=dict(arrowstyle="-|>", color="#1f77b4",
                            lw=0.8, alpha=0.45,
                            shrinkA=2, shrinkB=2),
        )
    ax.scatter(theta_shrunk[:, 0], theta_shrunk[:, 1],
               s=30, color="#1f77b4", alpha=0.6,
               edgecolor="white", lw=0.5)
    ax.set_title("(a) Shrink toward weighted mean:  $\\theta_i \\to a\\,\\theta_i + (1-a)\\,m$",
                 fontsize=11)
    ax.set_xlabel("$\\theta_1$")
    ax.set_ylabel("$\\theta_2$")
    ax.legend(loc="upper left", fontsize=9, frameon=False)

    # --- Panel (b): jitter ---
    ax = axes[1]
    ax.scatter(theta_shrunk[:, 0], theta_shrunk[:, 1],
               s=30, color="#1f77b4", alpha=0.4,
               edgecolor="white", lw=0.5, label="shrunk")
    ax.scatter([mean[0]], [mean[1]], s=180, marker="X",
               color="#d62728", edgecolor="white", lw=1.0, zorder=4)
    # Arrows from shrunk to jittered (Gaussian noise).
    for (xs, ys), (xe, ye) in zip(theta_shrunk, theta_jittered):
        ax.annotate(
            "",
            xy=(xe, ye), xytext=(xs, ys),
            arrowprops=dict(arrowstyle="-|>", color="#2e7d32",
                            lw=0.8, alpha=0.45,
                            shrinkA=2, shrinkB=2),
        )
    ax.scatter(theta_jittered[:, 0], theta_jittered[:, 1],
               s=70, color="#2e7d32", alpha=0.75,
               edgecolor="white", lw=0.8, label="θ-particle (post)")
    ax.set_title("(b) Jitter:  $+ \\, \\epsilon_i, \\quad \\epsilon_i \\sim \\mathcal{N}(0, h^2 V)$",
                 fontsize=11)
    ax.set_xlabel("$\\theta_1$")
    ax.legend(loc="upper left", fontsize=9, frameon=False)

    fig.suptitle(
        "Liu-West shrink-jitter:  drift the parameter cloud without collapsing it",
        y=1.04, fontsize=12.5,
    )
    # Reserve room below the panels for the formula box, then place it
    # inside the figure (not below it) so it isn't clipped by tight bbox.
    fig.subplots_adjust(bottom=0.22)
    fig.text(
        0.5, 0.04,
        "$\\theta_i' = a\\,\\theta_i + (1-a)\\,m + \\epsilon_i,  "
        "\\epsilon_i \\sim \\mathcal{N}(0, h^2 V),  "
        "a = \\sqrt{1 - h^2}$  "
        "→  preserves mean and covariance of the cloud asymptotically.",
        ha="center", va="center", fontsize=11,
        bbox=dict(facecolor="#FFF8E1", edgecolor="#E65100", lw=1, pad=6),
    )
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved → {OUT}")


if __name__ == "__main__":
    main()
