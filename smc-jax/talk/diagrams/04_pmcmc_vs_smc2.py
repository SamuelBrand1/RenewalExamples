"""Diagram 4 — pMCMC vs SMC²: two flowcharts side by side.

pMCMC (Particle Marginal Metropolis-Hastings):
    Outer:  MCMC loop over θ
    Inner:  PF computes  log p(y | θ)

SMC² (Chopin, Jacob, Papaspiliopoulos 2013):
    Outer:  SMC sampler over θ (tempered sequence)
    Inner:  PF computes  log p(y | θ)

Same inner block.  Different outer-loop strategy.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt

OUT = Path(__file__).parent.parent / "assets" / "pmcmc_vs_smc2.png"
OUT.parent.mkdir(parents=True, exist_ok=True)


def _style():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.spines.left": False,
            "axes.spines.bottom": False,
            "font.size": 10,
        }
    )


def _box(ax, xy, w, h, text, *, fc, ec, fontsize=10, bold=False):
    x, y = xy
    box = patches.FancyBboxPatch(
        (x - w / 2, y - h / 2), w, h,
        boxstyle="round,pad=0.04",
        linewidth=1.0, edgecolor=ec, facecolor=fc, alpha=0.85,
    )
    ax.add_patch(box)
    ax.text(x, y, text, ha="center", va="center",
            fontsize=fontsize,
            fontweight="bold" if bold else "normal",
            color=ec, wrap=True)


def _arrow(ax, src, dst, *, color="#444", label=None):
    ax.annotate(
        "",
        xy=dst, xytext=src,
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.0,
                        shrinkA=6, shrinkB=6),
    )
    if label is not None:
        mx = (src[0] + dst[0]) / 2
        my = (src[1] + dst[1]) / 2
        ax.text(mx + 0.15, my, label, fontsize=8.5, color=color,
                fontstyle="italic")


def _draw_loop(ax, x_centre, *, title, outer_label, color):
    """Draw one pMCMC- or SMC²-style outer/inner block centred at x_centre."""
    # Title at top
    ax.text(x_centre, 6.6, title, ha="center", fontsize=12.5,
            fontweight="bold", color=color)

    # Outer loop label (lhs annotation)
    _box(ax, (x_centre, 5.6), 4.4, 0.7, outer_label, fc="#ECEFF1", ec=color,
         fontsize=10.5, bold=True)

    # Inner block (PF)
    _box(ax, (x_centre, 4.0), 3.6, 0.7,
         "for each θ-proposal / θ-particle:",
         fc="#fafafa", ec="#666", fontsize=9.5)
    _box(ax, (x_centre, 3.0), 3.6, 0.7,
         "run a particle filter →  log p(y | θ)",
         fc="#FFF3E0", ec="#E65100", fontsize=10)
    _box(ax, (x_centre, 2.0), 3.6, 0.7,
         "(Andrieu–Doucet–Holenstein: unbiased)",
         fc="#fff", ec="#999", fontsize=8.5)

    # Arrows: outer down to inner
    _arrow(ax, (x_centre, 5.25), (x_centre, 4.35), color=color)
    _arrow(ax, (x_centre, 3.65), (x_centre, 3.35), color="#E65100")
    _arrow(ax, (x_centre, 2.65), (x_centre, 2.35), color="#999")
    # Loop back arrow on the right side — straight (no curve) so it
    # doesn't bow into the label.
    ax.annotate(
        "",
        xy=(x_centre + 2.6, 5.6),
        xytext=(x_centre + 2.6, 2.0),
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.2),
    )
    # Label placed FURTHER right of the arrow, so it isn't crossed.
    ax.text(x_centre + 2.9, 3.8,
            "update θ\n(accept/reject\nor reweight)",
            fontsize=8.5, color=color, ha="left", va="center",
            fontstyle="italic")


def main():
    _style()
    fig, ax = plt.subplots(figsize=(13, 7.0))
    ax.set_xlim(-0.5, 13.5)
    ax.set_ylim(-1.2, 7.2)
    ax.set_xticks([])
    ax.set_yticks([])

    # Left column: pMCMC
    _draw_loop(
        ax, x_centre=3.2,
        title="pMCMC / PMMH",
        outer_label="Outer:  Metropolis-Hastings over θ",
        color="#5e35b1",
    )

    # Right column: SMC²
    _draw_loop(
        ax, x_centre=9.8,
        title="SMC²",
        outer_label="Outer:  SMC sampler (tempered) over θ",
        color="#2e7d32",
    )

    # Bottom narrative — lowered to y = 1.0 (well clear of the box bottom
    # at y = 1.65) so the box edge doesn't clip the first text line.
    ax.text(3.2, 1.0,
            "θ is a single Markov chain.\nEmbarrassingly parallel only across chains.\n"
            "Asymptotically exact.",
            ha="center", va="top", fontsize=9.5, color="#5e35b1",
            fontstyle="italic")
    ax.text(9.8, 1.0,
            "θ is a population — adaptive tempering, parallel.\n"
            "Sequential-update friendly (extend by reweighting).\n"
            "Asymptotically exact.",
            ha="center", va="top", fontsize=9.5, color="#2e7d32",
            fontstyle="italic")

    # Same inner block reminder
    ax.text(6.5, -0.6,
            "Both share the inner block — a particle filter that returns an "
            "unbiased  log p(y | θ).\n"
            "They differ in how θ is explored.",
            ha="center", va="center", fontsize=10.5, color="#222",
            bbox=dict(facecolor="#FFF8E1", edgecolor="#E65100",
                      lw=1.0, pad=4))
    ax.set_ylim(-1.2, 7.2)

    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved → {OUT}")


if __name__ == "__main__":
    main()
