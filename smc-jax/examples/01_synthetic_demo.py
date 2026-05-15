"""01 — Synthetic data tour.

Draw one ground-truth trajectory from the nested-RW renewal model and visualise
the four latent random walks, the resulting infection curve, and the noisy
observed case counts.  Builds intuition for what the model is doing before
diving into inference.
"""

from __future__ import annotations

import _common  # noqa: F401  (must precede other imports — sets float64 + Agg)
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt

from smc_renewal.config import default_config
from smc_renewal.state import ParticleParams
from smc_renewal.synthetic import simulate


def main():
    _common.set_clean_style()
    cfg = default_config()
    T = 180  # ~6 months of daily data
    truth = ParticleParams(
        log_tau_R=jnp.asarray(-4.0),  # innovation-std std for log_Rt walk
        log_tau_F=jnp.asarray(-12.0),
        log_phi=jnp.asarray(2.5),
    )
    ds = simulate(jr.key(2), cfg, T, params=truth)

    print(
        f"T={T} log_Rt range: [{float(jnp.min(ds.log_Rt)):.2f}, {float(jnp.max(ds.log_Rt)):.2f}]\n"
        f"infections mean={float(jnp.mean(ds.infections)):.1f}, peak={float(jnp.max(ds.infections)):.1f}\n"
        f"observed y mean={float(jnp.mean(ds.y)):.1f}, peak={int(jnp.max(ds.y))}"
    )

    t = jnp.arange(T)
    fig, axes = plt.subplots(3, 2, figsize=(11, 7.5), sharex=True)

    axes[0, 0].plot(t, jnp.exp(ds.log_Rt), color="C0")
    axes[0, 0].axhline(1.0, color="k", lw=0.5, ls="--")
    axes[0, 0].set_ylabel("Rt")
    axes[0, 0].set_title("Effective reproduction number")

    axes[0, 1].plot(t, jnp.exp(ds.log_sigma_R), color="C1")
    axes[0, 1].set_ylabel("σ_R(t)")
    axes[0, 1].set_title("Innovation std of log Rt (random-walked)")

    axes[1, 0].plot(t, jnp.exp(ds.log_F), color="C2")
    axes[1, 0].set_ylabel("F(t)")
    axes[1, 0].set_title("Infection-feedback strength (F = exp(log_F))")
    axes[1, 0].set_yscale("log")

    axes[1, 1].plot(t, jnp.exp(ds.log_sigma_F), color="C3")
    axes[1, 1].set_ylabel("σ_F(t)")
    axes[1, 1].set_title("Innovation std of log F (random-walked)")

    axes[2, 0].plot(t, ds.infections, color="C4")
    axes[2, 0].set_ylabel("I(t)")
    axes[2, 0].set_title("Latent infections")
    axes[2, 0].set_xlabel("day")

    axes[2, 1].plot(t, ds.mu_y, color="C5", label="μ_y(t) (delay-conv. infections)")
    axes[2, 1].scatter(t, ds.y, s=10, color="k", alpha=0.6, label="y(t) (NegBin obs.)")
    axes[2, 1].set_ylabel("cases")
    axes[2, 1].set_title("Observed cases")
    axes[2, 1].set_xlabel("day")
    axes[2, 1].legend(loc="upper left", frameon=False)

    fig.suptitle("Nested-RW renewal model — one synthetic ground-truth trajectory", y=1.0)
    out = _common.save(fig, "01_synthetic_tour.png")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
