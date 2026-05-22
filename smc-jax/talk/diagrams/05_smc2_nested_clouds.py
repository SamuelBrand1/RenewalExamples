"""Diagram 5 — SMC² as nested clouds.

Outer cloud: parameter particles θ_i (large dots in 2-D θ-space).
For each θ_i: an "inner" cloud of state particles representing
log p(y_{1:T} | θ_i).

A right-hand arrow / annotation labels the rejuvenation step (resample +
move kernel) that the outer SMC sampler does between tempering steps.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt
import numpy as np

OUT = Path(__file__).parent.parent / "assets" / "smc2_nested_clouds.png"
OUT.parent.mkdir(parents=True, exist_ok=True)

rng = np.random.default_rng(13)


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
    fig = plt.figure(figsize=(13, 6.0))
    gs = fig.add_gridspec(1, 2, width_ratios=[1.3, 1.0], wspace=0.25)
    ax_left = fig.add_subplot(gs[0])
    ax_right = fig.add_subplot(gs[1])

    # --- Left: theta-cloud with inner state clouds ---
    n_outer = 10
    theta_cloud = rng.normal(0.0, 1.0, size=(n_outer, 2))
    # Bias one cluster
    theta_cloud[:6] += np.array([1.0, 0.5])
    theta_cloud[6:] += np.array([-0.8, -0.3])

    ax_left.set_title("Outer θ-cloud  +  inner state cloud per θ-particle",
                      fontsize=11.5)
    ax_left.set_xlabel("θ₁", fontsize=10)
    ax_left.set_ylabel("θ₂", fontsize=10)
    ax_left.set_xlim(-3, 4)
    ax_left.set_ylim(-3, 3)

    # Draw inner state clouds first (so outer dots sit on top)
    for tx, ty in theta_cloud:
        # 30 inner particles around each theta
        inner = rng.normal(0.0, 0.18, size=(30, 2)) + np.array([tx, ty])
        ax_left.scatter(inner[:, 0], inner[:, 1],
                        s=4, color="#FFA726", alpha=0.6)
    # Outer theta particles
    ax_left.scatter(theta_cloud[:, 0], theta_cloud[:, 1],
                    s=180, color="#1565C0", edgecolor="white", lw=1.2,
                    alpha=0.9, zorder=3, label="θ-particle")

    # Legend annotations
    ax_left.scatter([], [], s=180, color="#1565C0", alpha=0.9,
                    label="θ-particle  (parameter)")
    ax_left.scatter([], [], s=14, color="#FFA726",
                    label="inner state particle (latent)")
    ax_left.legend(loc="upper left", frameon=False, fontsize=9.5)

    # --- Right: schematic of an outer-loop iteration ---
    ax_right.set_title("Outer SMC step  (per tempering exponent β)", fontsize=11.5)
    ax_right.set_xticks([])
    ax_right.set_yticks([])
    ax_right.set_xlim(0, 10)
    ax_right.set_ylim(0, 10)
    ax_right.spines["left"].set_visible(False)
    ax_right.spines["bottom"].set_visible(False)

    boxes = [
        ((5, 9), "1.  For each θ_i,  run inner PF.\n      Get log p(y | θ_i).",
         "#FFF3E0", "#E65100"),
        ((5, 7), "2.  Update outer weights  w_i ∝ w_i · p(y | θ_i)^Δβ.",
         "#E3F2FD", "#1565C0"),
        ((5, 5), "3.  If ESS too low, resample θ-cloud.",
         "#E3F2FD", "#1565C0"),
        ((5, 3), "4.  Move kernel on θ  (random-walk MH within SMC).",
         "#E3F2FD", "#1565C0"),
        ((5, 1), "5.  Step β  →  β + Δβ  (adaptive).",
         "#E8F5E9", "#2E7D32"),
    ]
    for (cx, cy), text, fc, ec in boxes:
        box = patches.FancyBboxPatch(
            (cx - 4.3, cy - 0.55), 8.6, 1.1,
            boxstyle="round,pad=0.04",
            linewidth=1.0, edgecolor=ec, facecolor=fc, alpha=0.85,
        )
        ax_right.add_patch(box)
        ax_right.text(cx, cy, text, ha="center", va="center",
                      fontsize=9.5, color=ec)
    # Arrows
    for y1, y2 in [(9, 7), (7, 5), (5, 3), (3, 1)]:
        ax_right.annotate(
            "",
            xy=(5, y2 + 0.55), xytext=(5, y1 - 0.55),
            arrowprops=dict(arrowstyle="-|>", color="#666", lw=1.0),
        )

    fig.suptitle(
        "SMC² = outer SMC over θ × inner PF over latent state",
        y=1.02, fontsize=13,
    )
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved → {OUT}")


if __name__ == "__main__":
    main()
