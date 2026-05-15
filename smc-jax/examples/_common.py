"""Small shared helpers for the example scripts."""

from __future__ import annotations

import os
from pathlib import Path

import jax

# Enable float64 globally for stable particle-filter numerics.
jax.config.update("jax_enable_x64", True)

import matplotlib  # noqa: E402

# Use a non-interactive backend so scripts work in headless CI / over SSH.
matplotlib.use("Agg")

import matplotlib.pyplot as plt  # noqa: E402

FIGS = Path(__file__).parent / "figures"
FIGS.mkdir(exist_ok=True)


def save(fig, name: str) -> Path:
    out = FIGS / name
    fig.savefig(out, dpi=120, bbox_inches="tight")
    plt.close(fig)
    return out


def set_clean_style():
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": True,
            "grid.alpha": 0.25,
            "grid.linewidth": 0.6,
            "font.size": 10,
        }
    )
