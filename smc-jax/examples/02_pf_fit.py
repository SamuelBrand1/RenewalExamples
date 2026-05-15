"""02 — Liu-West bootstrap PF, with filtering vs smoothing diagnostics.

Fit the model to a synthetic case time series and visualise:
    1. Filtering p(log Rt[t] | y_{1..t}) AND smoothing p(log Rt[t] | y_{1..T})
       credible bands vs ground truth.  Smoothing draws come from
       ``pfjax.particle_smooth`` (genealogy-tracing) — note the early-time
       degeneracy this can produce.
    2. Posterior band for the time-varying volatility σ_R[t].
    3. ESS over time.
    4. Marginal posteriors over (log_tau_R, log_tau_F, log_phi).
"""

from __future__ import annotations

import _common  # noqa: F401
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
from pfjax import particle_smooth

from smc_renewal.config import default_config
from smc_renewal.pf.bootstrap import weighted_quantile
from smc_renewal.pf.runner import run_liu_west
from smc_renewal.state import ParticleParams
from smc_renewal.synthetic import simulate


def main():
    _common.set_clean_style()
    cfg = default_config()
    T = 180
    N = 4000
    truth = ParticleParams(
        log_tau_R=jnp.asarray(-4.0),
        log_tau_F=jnp.asarray(-12.0),
        log_phi=jnp.asarray(2.5),
    )
    ds = simulate(jr.key(2), cfg, T, params=truth)
    print(f"running PF: T={T}, N={N}, h=0.1")

    result = run_liu_west(jr.key(11), cfg, ds.y, n_particles=N, h=0.1)

    lw = result.log_weights_history
    log_Rt_p = result.particles_history.log_Rt
    log_sigma_R_p = result.particles_history.log_sigma_R

    # Quantile bands.
    qs = (0.05, 0.5, 0.95)
    log_Rt_q = jnp.stack(
        [jax.vmap(weighted_quantile)(log_Rt_p, lw, jnp.full((T,), q)) for q in qs]
    )
    log_sigma_R_q = jnp.stack(
        [jax.vmap(weighted_quantile)(log_sigma_R_p, lw, jnp.full((T,), q)) for q in qs]
    )

    in_band = (ds.log_Rt >= log_Rt_q[0]) & (ds.log_Rt <= log_Rt_q[2])
    print(f"filtering log_Rt 90% band coverage: {float(jnp.mean(in_band)):.3f}")

    # --- smoothing via pfjax.particle_smooth (genealogy-tracing) ---
    # Draw n_smooth backward-traced trajectories from p(x_{0:T} | y_{0:T}).
    # Note: this is the simple "particle smoother" — it uses ancestor lineages
    # of the surviving particles, which means early-time variability is
    # bottlenecked through the small number of distinct ancestors that survived
    # all the way back.  This produces overconfident bands at early times.
    n_smooth = 800
    smooth_keys = jr.split(jr.key(99), n_smooth)
    smooth_paths = jax.vmap(
        lambda k: particle_smooth(k, lw[-1], log_Rt_p, result.ancestors)
    )(smooth_keys)  # (n_smooth, T)
    # Equal-weight smoothing quantiles.
    smooth_q = jnp.quantile(smooth_paths, jnp.array([0.05, 0.5, 0.95]), axis=0)
    smooth_in_band = (ds.log_Rt >= smooth_q[0]) & (ds.log_Rt <= smooth_q[2])
    print(f"smoothing log_Rt 90% band coverage: {float(jnp.mean(smooth_in_band)):.3f}")

    t = jnp.arange(T)
    final_lw = lw[-1]
    w = jnp.exp(final_lw - jnp.max(final_lw)); w = w / w.sum()
    names = ["log_tau_R", "log_tau_F", "log_phi"]
    truths = [-4.0, -12.0, 2.5]
    samples = [
        result.params_history.log_tau_R[-1],
        result.params_history.log_tau_F[-1],
        result.params_history.log_phi[-1],
    ]

    # 3-row layout: 2 trace panels on top, 1 ESS panel in middle, 3 histograms bottom.
    fig = plt.figure(figsize=(12, 9))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.4, 1.0, 1.0], hspace=0.55, wspace=0.30)

    ax = fig.add_subplot(gs[0, :2])
    ax.fill_between(t, log_Rt_q[0], log_Rt_q[2], color="C0", alpha=0.18, label="filter 90%")
    ax.plot(t, log_Rt_q[1], color="C0", lw=1.0, label="filter median")
    ax.fill_between(t, smooth_q[0], smooth_q[2], color="C4", alpha=0.20, label="smoother 90%")
    ax.plot(t, smooth_q[1], color="C4", lw=1.0, label="smoother median")
    ax.plot(t, ds.log_Rt, color="k", lw=1.4, label="truth")
    ax.set_title("log Rt(t) — filtering p(x_t | y_{1..t}) vs smoothing p(x_t | y_{1..T})")
    ax.set_xlabel("day"); ax.set_ylabel("log Rt")
    ax.legend(loc="lower left", frameon=False, fontsize=8, ncol=2)

    ax = fig.add_subplot(gs[0, 2])
    ax.fill_between(t, jnp.exp(log_sigma_R_q[0]), jnp.exp(log_sigma_R_q[2]), color="C1", alpha=0.25)
    ax.plot(t, jnp.exp(log_sigma_R_q[1]), color="C1", label="post. median")
    ax.plot(t, jnp.exp(ds.log_sigma_R), color="k", lw=1.2, label="truth")
    ax.set_title("σ_R(t)")
    ax.set_xlabel("day"); ax.set_ylabel("σ_R")
    ax.legend(loc="upper left", frameon=False, fontsize=8)

    ax = fig.add_subplot(gs[1, :])
    ax.plot(t, result.ess_history, color="C3")
    ax.axhline(N / 2.0, color="k", lw=0.5, ls="--", label=f"N/2 ({N//2})")
    ax.axhline(N / 10.0, color="r", lw=0.5, ls="--", label=f"N/10 ({N//10})")
    ax.set_title(f"Effective sample size (N={N})")
    ax.set_xlabel("day"); ax.set_ylabel("ESS")
    ax.legend(loc="upper right", frameon=False)

    for i, (name, tr, s) in enumerate(zip(names, truths, samples)):
        a = fig.add_subplot(gs[2, i])
        a.hist(s, bins=30, weights=w, color="C2", alpha=0.75)
        a.axvline(tr, color="k", lw=1.2, ls="--", label="truth")
        a.set_title(name)
        a.tick_params(axis="y", labelleft=False, left=False)
        if i == 0:
            a.legend(loc="upper right", frameon=False, fontsize=8)

    fig.suptitle("Liu-West bootstrap PF — diagnostics on synthetic data", y=0.995)
    out = _common.save(fig, "02_pf_fit.png")
    print(f"saved: {out}")

    # Headline metrics to stdout.
    for name, tr, s in zip(names, truths, samples):
        m = float(jnp.sum(w * s))
        print(f"  posterior mean {name} = {m:+.3f}  (truth {tr:+.3f})")


if __name__ == "__main__":
    main()
