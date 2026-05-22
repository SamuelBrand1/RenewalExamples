"""Diagram 7 — Why NUTS can't differentiate through discrete latents.

Two-panel:
  (a) smooth log-posterior over a continuous latent — HMC trajectories slide
      down the gradient.
  (b) staircase log-posterior over an integer latent (e.g. cohort count) —
      flat between integers, undefined at the jumps.  No gradient to flow
      down.  HMC gets stuck or refuses.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent.parent / "assets" / "discrete_latent_gradient.png"
OUT.parent.mkdir(parents=True, exist_ok=True)


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

    fig, axes = plt.subplots(1, 2, figsize=(12.5, 4.6))

    # --- Panel (a): smooth log-posterior over continuous latent ---
    ax = axes[0]
    x = np.linspace(-4, 4, 400)
    log_p = -(x - 0.7) ** 2 / 2.4 + 0.3 * np.sin(x * 1.2) - 1.0
    ax.plot(x, log_p, color="#1565C0", lw=2.0)
    ax.fill_between(x, log_p.min() - 0.1, log_p, color="#1565C0", alpha=0.08)
    # Arrows showing gradient flow
    for x0 in [-2.5, -1.0, 2.0, 3.2]:
        idx = np.argmin(np.abs(x - x0))
        slope = (log_p[idx + 1] - log_p[idx - 1]) / (x[idx + 1] - x[idx - 1])
        ax.annotate(
            "",
            xy=(x0 + 0.4 * np.sign(slope) if abs(slope) > 0.01 else x0, log_p[idx]),
            xytext=(x0, log_p[idx]),
            arrowprops=dict(arrowstyle="-|>", color="#2e7d32", lw=1.4),
        )
    ax.set_title("(a) Continuous latent — smooth log-posterior",
                 fontsize=11.5)
    ax.set_xlabel("latent state  $x$ (continuous)", fontsize=10.5)
    ax.set_ylabel("log $p(x \\mid y)$", fontsize=10.5)
    # Caption placed inside the plot area, top-right corner — far from
    # the x-axis label and from the gradient arrows.
    ax.text(0.97, 0.05,
            "Gradient $\\nabla \\log p(x \\mid y)$ is well-defined.\n"
            "HMC / NUTS happily flow downhill.",
            transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9.5, color="#2e7d32",
            fontstyle="italic",
            bbox=dict(facecolor="white", edgecolor="#2e7d32",
                      lw=0.7, pad=3, alpha=0.92))

    # --- Panel (b): integer latent — staircase ---
    ax = axes[1]
    # Build a piecewise-constant log-posterior over integers.
    n_int = np.arange(0, 11)
    log_p_n = -((n_int - 4) ** 2) / 3.5 + 0.5 * np.sin(n_int * 0.8) - 0.5
    # Step plot: draw horizontal segments + vertical jumps.
    for i, n in enumerate(n_int):
        ax.hlines(log_p_n[i], n - 0.5, n + 0.5,
                  color="#c62828", lw=2.5)
        if i < len(n_int) - 1:
            ax.vlines(n + 0.5,
                      min(log_p_n[i], log_p_n[i + 1]),
                      max(log_p_n[i], log_p_n[i + 1]),
                      color="#c62828", lw=0.8, ls=":")
        # Dots at integer points
        ax.scatter([n], [log_p_n[i]], s=40, color="#c62828",
                   zorder=3, edgecolor="white", lw=0.8)

    # Try to draw a "stuck" HMC trajectory: bounces flat then fails.
    for x_h, dx in [(2.5, 0.6), (6.5, -0.6)]:
        ax.annotate(
            "",
            xy=(x_h + dx, log_p_n[int(round(x_h))]),
            xytext=(x_h, log_p_n[int(round(x_h))]),
            arrowprops=dict(arrowstyle="-|>", color="#999", lw=1.0,
                            linestyle="dotted"),
        )
    ax.set_title("(b) Integer latent — staircase log-posterior",
                 fontsize=11.5)
    ax.set_xlabel("latent count  $N$ (integer)", fontsize=10.5)
    ax.set_ylabel("log $p(N \\mid y)$", fontsize=10.5)
    ax.set_xticks(n_int)
    ax.set_xlim(-0.6, 10.6)
    # Caption placed inside the plot area, bottom-right corner — clear of
    # the staircase, the dotted-arrow HMC stalls, and the x-axis label.
    ax.text(0.97, 0.05,
            "Gradient is zero between integers,\n"
            "undefined at the jumps.\n"
            "HMC has nowhere to go.",
            transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9.5, color="#c62828",
            fontstyle="italic",
            bbox=dict(facecolor="white", edgecolor="#c62828",
                      lw=0.7, pad=3, alpha=0.92))

    fig.suptitle(
        "Why HMC / NUTS does not handle discrete latents — "
        "no gradient to follow",
        y=1.02, fontsize=12.5,
    )
    fig.tight_layout()
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved → {OUT}")


if __name__ == "__main__":
    main()
