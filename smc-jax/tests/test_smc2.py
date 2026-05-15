"""Smoke test for the SMC^2-with-UKF runner.

We only assert that the algorithm terminates with a non-degenerate posterior on
``theta`` and that the marginal log-Z is finite.  Recovering the truth precisely
is not the goal of this project; see the rolling-origin forecast tests for
the actual deliverable.
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr

from smc_renewal.config import default_config
from smc_renewal.smc2.tempered import run_smc2
from smc_renewal.state import ParticleParams
from smc_renewal.synthetic import simulate


def test_smc2_runs_and_returns_finite_posterior():
    cfg = default_config()
    T = 80
    truth = ParticleParams(
        log_tau_R=jnp.asarray(-4.0),
        log_tau_F=jnp.asarray(-12.0),
        log_phi=jnp.asarray(2.5),
    )
    ds = simulate(jr.key(2), cfg, T, params=truth)
    assert bool(jnp.isfinite(ds.mu_y).all()), "synthetic exploded; retry seed"

    result = run_smc2(
        jr.key(11),
        cfg,
        ds.y,
        n_particles=64,
        target_ess_frac=0.5,
        num_mcmc_steps=3,
        rwmh_scale=0.4,
        max_outer_steps=20,
    )

    assert jnp.isfinite(result.log_marginal), result.log_marginal
    assert bool(jnp.isfinite(result.particles).all())
    # Variance across particles should be positive (non-collapsed).
    var = jnp.var(result.particles, axis=0)
    assert bool(jnp.all(var > 1e-6)), f"posterior cloud collapsed: var = {var}"
