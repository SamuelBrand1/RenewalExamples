"""Smoke tests for Model C (integrated-Brownian-motion log_Rt / log_F)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr

from smc_renewal.config import default_config
from smc_renewal.pf.bootstrap import weighted_quantile
from smc_renewal.pf.runner_trend import run_liu_west_trend
from smc_renewal.state import ParticleParams
from smc_renewal.synthetic import simulate


def test_model_trend_runs_and_covers_log_Rt():
    cfg = default_config()
    T = 120
    N = 2000
    truth = ParticleParams(
        log_tau_R=jnp.asarray(-4.0),
        log_tau_F=jnp.asarray(-12.0),
        log_phi=jnp.asarray(2.5),
    )
    ds = simulate(jr.key(2), cfg, T, params=truth)
    assert bool(jnp.isfinite(ds.mu_y).all())

    result = run_liu_west_trend(
        jr.key(11), cfg, ds.y, n_particles=N, h=0.1, fixed_lag_L=14
    )

    # No NaNs anywhere.
    assert bool(jnp.isfinite(result.log_lik).all())
    assert bool(jnp.isfinite(result.particles_history.log_Rt).all())
    assert bool(jnp.isfinite(result.particles_history.v_R).all())
    assert bool(jnp.isfinite(result.params_history.log_sigma_vR).all())

    # Coverage on log_Rt should be reasonable.
    log_Rt_p = result.particles_history.log_Rt
    lw = result.log_weights_history
    lo = jax.vmap(weighted_quantile, in_axes=(0, 0, None))(log_Rt_p, lw, jnp.asarray(0.05))
    hi = jax.vmap(weighted_quantile, in_axes=(0, 0, None))(log_Rt_p, lw, jnp.asarray(0.95))
    cov = float(jnp.mean((ds.log_Rt >= lo) & (ds.log_Rt <= hi)))
    assert cov >= 0.55, f"log_Rt 90% coverage = {cov:.3f}"


def test_model_trend_carries_velocity():
    """Sanity: the velocity state actually moves under the dynamics."""
    cfg = default_config()
    T = 60
    N = 1000
    truth = ParticleParams(
        log_tau_R=jnp.asarray(-3.0),
        log_tau_F=jnp.asarray(-7.0),
        log_phi=jnp.asarray(2.5),
    )
    ds = simulate(jr.key(2), cfg, T, params=truth)
    result = run_liu_west_trend(
        jr.key(11), cfg, ds.y, n_particles=N, h=0.1, fixed_lag_L=14
    )
    # v_R at final time should have non-trivial variance across particles.
    v_R_final = result.particles_history.v_R[-1]
    assert float(jnp.std(v_R_final)) > 1e-4, "v_R cloud collapsed"
