"""08 — Comparison: Liu-West on log_τ (Model A) vs Liu-West on log_σ (Model B).

Same synthetic data fitted with two different placements of the Liu-West
approximation:

- **Model A** (``pf/runner.py::run_liu_west``): doubly-stochastic structure
  with ``log σ_R`` and ``log σ_F`` as time-varying latent states walking
  with rates ``τ_R, τ_F``; Liu-West applied to the static ``(log τ_R,
  log τ_F, log φ)`` triple.
- **Model B** (``pf/runner_sigma.py::run_liu_west_sigma``): one fewer level
  of hierarchy.  ``log σ_R, log σ_F`` are themselves Liu-West parameters
  (slowly drifting via shrink-jitter).  No ``τ``; the Liu-West shrinkage
  ``h`` plays its role.

Hypothesis: Model B should be better-identified because Liu-West acts on
the strongly-identified ``log σ`` instead of the weakly-identified
``log τ`` (where the data signal is second-order and prior-dominated).
"""

from __future__ import annotations

import dataclasses

import _common  # noqa: F401
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np

from smc_renewal.config import default_config
from smc_renewal.pf.bootstrap import weighted_quantile
from smc_renewal.pf.runner import run_liu_west
from smc_renewal.pf.runner_sigma import run_liu_west_sigma
from smc_renewal.state import ParticleParams
from smc_renewal.synthetic import simulate


def _wmean(x, lw):
    w = jnp.exp(lw - jnp.max(lw))
    w = w / w.sum()
    return jnp.sum(w * x)


def main():
    _common.set_clean_style()

    # Modest-sized synthetic series so the comparison runs in <2 min.
    cfg = dataclasses.replace(
        default_config(),
        sigma_floor=1e-6,
        init_log_F_mean=-9.0,
        init_log_F_sd=1.0,
        init_log_sigma_F_mean=-7.0,
        init_log_sigma_F_sd=1.0,
        prior_log_tau_F_mean=-8.0,
        prior_log_tau_F_sd=1.0,
    )
    T = 360
    truth = ParticleParams(
        log_tau_R=jnp.asarray(-3.5),
        log_tau_F=jnp.asarray(-7.0),
        log_phi=jnp.asarray(2.5),
    )
    ds = simulate(jr.key(2), cfg, T, params=truth)
    assert bool(jnp.isfinite(ds.mu_y).all()), "synthetic exploded, retry seed"

    N = 4000
    h = 0.1
    L_lag = 21
    print(f"fitting both models: T={T}, N={N}, h={h}, fixed_lag_L={L_lag}")

    print("  Model A (Liu-West on log_τ)...")
    res_A = run_liu_west(jr.key(11), cfg, ds.y, n_particles=N, h=h, fixed_lag_L=L_lag)
    print("  Model B (Liu-West on log_σ)...")
    res_B = run_liu_west_sigma(jr.key(11), cfg, ds.y, n_particles=N, h=h, fixed_lag_L=L_lag)

    # --- log Rt coverage (in-sample, headline diagnostic) ---
    def coverage_logRt(result):
        lw = result.log_weights_history
        log_Rt_p = result.particles_history.log_Rt
        lo = jax.vmap(weighted_quantile, in_axes=(0, 0, None))(log_Rt_p, lw, jnp.asarray(0.05))
        hi = jax.vmap(weighted_quantile, in_axes=(0, 0, None))(log_Rt_p, lw, jnp.asarray(0.95))
        cov = float(jnp.mean((ds.log_Rt >= lo) & (ds.log_Rt <= hi)))
        return lo, hi, cov

    lo_A, hi_A, cov_A = coverage_logRt(res_A)
    lo_B, hi_B, cov_B = coverage_logRt(res_B)
    print(f"  log_Rt 90% filter coverage:  Model A {cov_A:.3f}   Model B {cov_B:.3f}")

    # Marginal log-likelihoods (Model B should be at least as good if it's
    # not strictly throwing away information).
    print(f"  log marginal lik:            Model A {float(res_A.log_lik[-1]):.2f}   Model B {float(res_B.log_lik[-1]):.2f}")

    # Posterior on σ_R: in Model A this is a latent-state trajectory; in
    # Model B it's a directly-Liu-West'd parameter.  Compare distributions.
    final_lw_A = res_A.log_weights_history[-1]
    final_lw_B = res_B.log_weights_history[-1]
    # Model A: σ_R(T) from state trajectory at last time-step.
    sigma_R_A = res_A.particles_history.log_sigma_R[-1]
    # Model B: log_σ_R from final-step Liu-West parameter cloud.
    sigma_R_B = res_B.params_history.log_sigma_R[-1]
    print(f"  final log_σ_R posterior mean:")
    print(f"    Model A (from state):     {float(_wmean(sigma_R_A, final_lw_A)):+.3f}")
    print(f"    Model B (from Liu-West):  {float(_wmean(sigma_R_B, final_lw_B)):+.3f}")

    # --- plot ---
    t = jnp.arange(T)
    fig, axes = plt.subplots(3, 2, figsize=(12, 9), sharex="col",
                             gridspec_kw={"hspace": 0.45, "wspace": 0.30})

    # log Rt bands, side by side.
    for col, (res, lo, hi, color) in enumerate([
        (res_A, lo_A, hi_A, "C0"),
        (res_B, lo_B, hi_B, "C3"),
    ]):
        ax = axes[0, col]
        ax.plot(t, ds.log_Rt, color="k", lw=1.4, label="truth")
        ax.fill_between(t, lo, hi, color=color, alpha=0.22, label="90% filter")
        median = jax.vmap(weighted_quantile, in_axes=(0, 0, None))(
            res.particles_history.log_Rt, res.log_weights_history, jnp.asarray(0.5)
        )
        ax.plot(t, median, color=color, lw=1.0, label="median")
        ax.set_ylabel("log Rt")
        if col == 0:
            ax.legend(loc="lower right", frameon=False, fontsize=9)

    axes[0, 0].set_title(f"Model A — LW on log_τ  ·  cov={cov_A:.2f}")
    axes[0, 1].set_title(f"Model B — LW on log_σ  ·  cov={cov_B:.2f}")

    # σ_R trajectories: Model A latent state, Model B Liu-West param cloud.
    for col, (arr, lw_hist, label, color) in enumerate([
        (res_A.particles_history.log_sigma_R, res_A.log_weights_history, "Model A: log σ_R (state)", "C0"),
        (res_B.params_history.log_sigma_R, res_B.log_weights_history, "Model B: log σ_R (LW param)", "C3"),
    ]):
        ax = axes[1, col]
        qs = (0.10, 0.50, 0.90)
        Q = jnp.stack([
            jax.vmap(weighted_quantile, in_axes=(0, 0, None))(arr, lw_hist, jnp.asarray(q))
            for q in qs
        ])
        ax.fill_between(t, Q[0], Q[2], color=color, alpha=0.22)
        ax.plot(t, Q[1], color=color, lw=1.0, label="median")
        ax.set_ylabel("log σ_R")
        ax.set_title(label)

    # Final-step posterior histograms on log_σ_R / log_σ_F / log_φ.
    # Model A's "log σ_R" is a state at time T; Model B's is the LW parameter.
    # Different objects but both represent "how variable is log Rt".
    name_pairs = [
        ("log σ_R", res_A.particles_history.log_sigma_R[-1], res_B.params_history.log_sigma_R[-1]),
        ("log φ",
         res_A.params_history.log_phi[-1],  # Model A: log_phi is a Liu-West param
         res_B.params_history.log_phi[-1]), # Model B: same
    ]
    truths_dict = {
        "log σ_R": None,  # truth doesn't have a single σ_R (it's RW'd)
        "log φ": float(truth.log_phi),
    }
    for col, (name, arr_A, arr_B) in enumerate(name_pairs):
        ax = axes[2, col]
        wA = jnp.exp(final_lw_A - jnp.max(final_lw_A)); wA = wA / wA.sum()
        wB = jnp.exp(final_lw_B - jnp.max(final_lw_B)); wB = wB / wB.sum()
        ax.hist(np.asarray(arr_A), bins=40, weights=np.asarray(wA), density=True,
                color="C0", alpha=0.55, label="Model A")
        ax.hist(np.asarray(arr_B), bins=40, weights=np.asarray(wB), density=True,
                color="C3", alpha=0.55, label="Model B")
        if truths_dict[name] is not None:
            ax.axvline(truths_dict[name], color="k", lw=1.2, ls="--", label="truth")
        ax.set_title(f"final-step posterior on {name}")
        ax.tick_params(axis="y", labelleft=False, left=False)
        if col == 0:
            ax.legend(loc="upper right", frameon=False, fontsize=8)

    axes[0, 0].set_xlabel("day")
    axes[0, 1].set_xlabel("day")
    axes[1, 0].set_xlabel("day")
    axes[1, 1].set_xlabel("day")

    fig.suptitle(
        "Liu-West placement comparison — log_τ (Model A) vs log_σ (Model B)",
        y=0.995,
    )
    out = _common.save(fig, "08_lw_on_sigma_vs_tau.png")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
