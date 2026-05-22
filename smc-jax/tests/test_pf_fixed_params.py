"""Coverage test: pfjax-backed bootstrap PF on synthetic data with theta fixed at truth.

Verifies the pfjax integration produces a non-degenerate filtering distribution
that covers ground truth ≥65% of the time on the 80% credible band (averaged
over a few well-behaved synthetic seeds — bootstrap PFs are intrinsically noisy
on this kind of state space and we don't want a brittle test).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr

from smc_renewal.config import default_config
from smc_renewal.pf.bootstrap import run_pf, weighted_quantile
from smc_renewal.state import ParticleParams
from smc_renewal.synthetic import simulate


def _coverage_one_seed(cfg, T, N, params, sim_seed, pf_seed):
    ds = simulate(jr.key(sim_seed), cfg, T, params=params)
    if not bool(jnp.isfinite(ds.mu_y).all()):
        return None
    res = run_pf(jr.key(pf_seed), cfg, ds.y, params, n_particles=N)
    log_Rt_p = res.x_particles.log_Rt           # (T, N)
    lw = res.logw                                # (T, N)

    lo = jax.vmap(weighted_quantile, in_axes=(0, 0, None))(
        log_Rt_p, lw, jnp.asarray(0.10)
    )
    hi = jax.vmap(weighted_quantile, in_axes=(0, 0, None))(
        log_Rt_p, lw, jnp.asarray(0.90)
    )
    cov = float(jnp.mean((ds.log_Rt >= lo) & (ds.log_Rt <= hi)))
    return cov


def test_pf_rt_coverage_on_synthetic():
    cfg = default_config()
    T = 120
    N = 2000
    truth = ParticleParams(
        log_tau_R=jnp.asarray(-4.0),
        log_tau_F=jnp.asarray(-12.0),
        log_phi=jnp.asarray(2.5),
    )

    covs = []
    for sim_seed in range(8):
        c = _coverage_one_seed(cfg, T, N, truth, sim_seed, sim_seed + 100)
        if c is None:
            continue
        covs.append(c)
        if len(covs) >= 5:
            break

    assert len(covs) >= 3, f"too few well-behaved synthetic seeds ({len(covs)})"
    mean_cov = sum(covs) / len(covs)
    assert mean_cov >= 0.65, (
        f"mean 80% band coverage = {mean_cov:.3f} over seeds {covs} (< 0.65)"
    )
