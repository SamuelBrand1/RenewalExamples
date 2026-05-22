"""Diagram 1 — "SMC is about sequentially evolving a probability distribution".

Five panels showing a 1-D density at successive time steps, going through
predict → update → predict → update → ... .  The density starts wide (prior),
gets narrowed by an observation, drifts under a transition, gets narrowed
again, etc.  Cartoon — no real model here, just the *shape* of what
sequential Bayesian filtering does.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

OUT = Path(__file__).parent.parent / "assets" / "density_evolution.png"
OUT.parent.mkdir(parents=True, exist_ok=True)


def _style():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "font.size": 11,
        }
    )


def main():
    _style()
    x = np.linspace(-6, 6, 400)

    # Sequence: (mu, sigma) for the displayed density at each panel.
    # Even indices = after prediction (wider); odd indices = after update (narrower).
    sequence = [
        (-2.0, 1.6, "t = 0:  prior  p(x_0)"),
        (-1.4, 0.7, "t = 0:  posterior  p(x_0 | y_0)"),
        (-0.4, 1.2, "t = 1:  predict  p(x_1 | y_0)"),
        (0.3, 0.55, "t = 1:  posterior  p(x_1 | y_{0:1})"),
        (1.3, 1.0, "t = 2:  predict  p(x_2 | y_{0:1})"),
        (1.9, 0.5, "t = 2:  posterior  p(x_2 | y_{0:2})"),
    ]

    fig, axes = plt.subplots(1, 6, figsize=(15, 2.6), sharey=True)
    for ax, (mu, sd, title) in zip(axes, sequence):
        y = norm.pdf(x, mu, sd)
        # Fill under the curve to make it visually 'a distribution'.
        is_update = "posterior" in title
        col = "#1f77b4" if not is_update else "#d62728"
        ax.fill_between(x, 0, y, color=col, alpha=0.30, linewidth=0)
        ax.plot(x, y, color=col, lw=1.4)
        ax.set_title(title, fontsize=9.5)
        ax.set_xlim(-6, 6)
        ax.set_ylim(0, 0.85)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel("latent state $x$", fontsize=9)
        # Annotate the type of step
        kind = "update" if is_update else "predict"
        ax.text(
            0.04, 0.93, kind,
            transform=ax.transAxes,
            fontsize=9, color=col,
            ha="left", va="top",
            bbox=dict(facecolor="white", edgecolor=col, lw=0.7, pad=2.5),
        )

    fig.suptitle(
        "Sequential Bayesian filtering — the posterior is itself a stream",
        y=1.04,
        fontsize=12.5,
    )
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved → {OUT}")


if __name__ == "__main__":
    main()
