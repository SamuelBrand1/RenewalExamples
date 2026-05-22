"""Sequential consistency: full fit-through-T ≈ fit-through-T/2 + sequential update.

The two cannot be IDENTICAL — they consume different random keys — but their
posterior summaries (e.g. mean log_Rt at time T) should agree within sampling
error.  Likewise the cumulative marginal log-likelihood should agree closely.
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr

from smc_renewal.config import default_config
from smc_renewal.pf.runner import run_liu_west
from smc_renewal.pf.update import _slice_final, extend_liu_west
from smc_renewal.state import ParticleParams
from smc_renewal.synthetic import simulate


def _wmean(x, lw):
    w = jnp.exp(lw - jnp.max(lw))
    w = w / w.sum()
    return jnp.sum(w * x)


def test_full_fit_matches_split_then_update():
    cfg = default_config()
    T_total = 90
    T_split = 45
    N = 1500
    truth = ParticleParams(
        log_tau_R=jnp.asarray(-4.0),
        log_tau_F=jnp.asarray(-12.0),
        log_phi=jnp.asarray(2.5),
    )
    ds = simulate(jr.key(2), cfg, T_total, params=truth)
    assert bool(jnp.isfinite(ds.mu_y).all()), "synthetic exploded; retry seed"

    # Path 1: full fit.
    full = run_liu_west(jr.key(11), cfg, ds.y, n_particles=N, h=0.1)

    # Path 2: split fit + sequential extend.
    first = run_liu_west(jr.key(11), cfg, ds.y[:T_split], n_particles=N, h=0.1)
    particles_T, params_T, lw_T = _slice_final(first)
    second = extend_liu_west(
        jr.key(22), cfg, ds.y[T_split:], particles_T, params_T, lw_T, h=0.1
    )

    # Final posterior mean of log_Rt at day T_total - 1.
    mu_full = _wmean(
        full.particles_history.log_Rt[-1], full.log_weights_history[-1]
    )
    mu_split = _wmean(
        second.particles_history.log_Rt[-1], second.log_weights_history[-1]
    )
    # Allow generous tolerance — different keys give different particle clouds.
    assert abs(float(mu_full) - float(mu_split)) < 0.6, (
        f"final-day posterior mean mismatch: full={float(mu_full):.3f} vs "
        f"split={float(mu_split):.3f}"
    )

    # Cumulative marginal log-likelihoods should agree (sum of increments
    # is path-independent up to MC noise).
    ll_full = float(full.log_lik[-1])
    ll_split = float(first.log_lik[-1] + second.log_lik[-1])
    rel = abs(ll_full - ll_split) / max(abs(ll_full), 1.0)
    assert rel < 0.15, f"log-lik mismatch full={ll_full:.1f} vs split={ll_split:.1f}"
