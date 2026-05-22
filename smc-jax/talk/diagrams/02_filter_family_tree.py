"""Diagram 2 — Family tree of sequential-Bayesian filters.

Root: Kalman filter (linear, Gaussian).  Two main branches:

  Gaussian-approximate (still parametric):  EKF, UKF, EnKF
  Sample-based (non-parametric):            bootstrap PF, auxiliary PF, guided PF

One-line caption on each node describes the assumption made.  No graphviz
dependency — drawn with matplotlib boxes and lines.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib.patches as patches
import matplotlib.pyplot as plt

OUT = Path(__file__).parent.parent / "assets" / "filter_family_tree.png"
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


def _node(ax, xy, width, height, title, subtitle, *, fc, ec):
    x, y = xy
    box = patches.FancyBboxPatch(
        (x - width / 2, y - height / 2),
        width, height,
        boxstyle="round,pad=0.04",
        linewidth=1.1, edgecolor=ec, facecolor=fc, alpha=0.85,
    )
    ax.add_patch(box)
    ax.text(x, y + 0.13, title, ha="center", va="center",
            fontsize=11, fontweight="bold", color=ec)
    ax.text(x, y - 0.18, subtitle, ha="center", va="center",
            fontsize=8.5, color="#222", wrap=True)


def _arrow(ax, src, src_box_h, dst, dst_box_h, *, color="#444"):
    """Arrow from BOTTOM of src box to TOP of dst box, so it never crosses
    title text inside the boxes."""
    src_bottom = (src[0], src[1] - src_box_h / 2)
    dst_top = (dst[0], dst[1] + dst_box_h / 2)
    ax.annotate(
        "",
        xy=dst_top, xytext=src_bottom,
        arrowprops=dict(arrowstyle="-|>", color=color, lw=1.0,
                        shrinkA=2, shrinkB=2),
    )


def main():
    _style()
    fig, ax = plt.subplots(figsize=(13, 6.5))
    ax.set_xlim(-0.2, 13.2)
    ax.set_ylim(-0.2, 6.7)
    ax.set_xticks([])
    ax.set_yticks([])

    # Root
    root_xy = (6.5, 5.7)
    root_h = 1.0
    _node(ax, root_xy, 4.0, root_h, "Kalman filter",
          "Exact when transition & observation are linear and Gaussian",
          fc="#cfd8dc", ec="#37474f")

    # Gaussian-parametric children
    ekf_xy = (1.2, 2.6)
    ukf_xy = (3.0, 2.6)
    enkf_xy = (4.8, 2.6)
    child_h = 1.0
    _node(ax, ekf_xy, 1.6, child_h, "EKF",
          "Linearise about\ncurrent mean (Jacobian)",
          fc="#bbdefb", ec="#1565c0")
    _node(ax, ukf_xy, 1.6, child_h, "UKF",
          "Sigma-point\nquadrature",
          fc="#bbdefb", ec="#1565c0")
    _node(ax, enkf_xy, 1.6, child_h, "EnKF",
          "Monte-Carlo prediction,\nGaussian update",
          fc="#bbdefb", ec="#1565c0")
    for c in (ekf_xy, ukf_xy, enkf_xy):
        _arrow(ax, root_xy, root_h, c, child_h, color="#1565c0")

    # Sample-based children
    boot_xy = (8.0, 2.6)
    aux_xy = (10.0, 2.6)
    guided_xy = (12.0, 2.6)
    _node(ax, boot_xy, 1.7, child_h, "Bootstrap PF",
          "Propose from transition,\nweight by likelihood",
          fc="#ffcdd2", ec="#c62828")
    _node(ax, aux_xy, 1.7, child_h, "Auxiliary PF",
          "Look-ahead\nresampling",
          fc="#ffcdd2", ec="#c62828")
    _node(ax, guided_xy, 1.7, child_h, "Guided PF",
          "Custom proposal,\nIS correction",
          fc="#ffcdd2", ec="#c62828")
    for c in (boot_xy, aux_xy, guided_xy):
        _arrow(ax, root_xy, root_h, c, child_h, color="#c62828")

    # Branch labels — placed at the LEFT MARGIN of each half, NOT on the
    # arrow path between root and children.
    ax.text(0.2, 4.4, "Gaussian-parametric branch",
            fontsize=10.5, fontweight="bold", color="#1565c0",
            ha="left", va="center",
            bbox=dict(facecolor="white", edgecolor="#1565c0",
                      lw=0.8, pad=3))
    ax.text(12.8, 4.4, "Sample-based branch",
            fontsize=10.5, fontweight="bold", color="#c62828",
            ha="right", va="center",
            bbox=dict(facecolor="white", edgecolor="#c62828",
                      lw=0.8, pad=3))

    # Bottom-row narratives
    ax.text(3.0, 1.55,
            "Keep the posterior Gaussian.\n"
            "Cheap, batchable, jit-able.\n"
            "Blind to non-Gaussian structure.",
            fontsize=9.5, color="#1565c0", ha="center", va="top",
            style="italic")
    ax.text(10.0, 1.55,
            "Posterior = weighted particle cloud.\n"
            "Handles arbitrary dynamics / discreteness.\n"
            "Cost scales with N_particles.",
            fontsize=9.5, color="#c62828", ha="center", va="top",
            style="italic")

    ax.set_title("A family tree of sequential-Bayesian filters",
                 fontsize=13.5, pad=10)
    fig.savefig(OUT, dpi=200, bbox_inches="tight")
    plt.close(fig)
    print(f"saved → {OUT}")


if __name__ == "__main__":
    main()
