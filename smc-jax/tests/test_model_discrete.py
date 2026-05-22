"""Smoke tests for Model D (discrete Poisson renewal)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr

from smc_renewal.config import default_config
from smc_renewal.pf.bootstrap import weighted_quantile
from smc_renewal.pf.runner_discrete import run_liu_west_discrete
from smc_renewal.state import ParticleParams
from smc_renewal.synthetic import simulate


def test_model_discrete_runs_and_covers_log_Rt():
    cfg = default_config()
    T = 120
    N = 2000
    # Generate truth from Model A (deterministic renewal) — not Model D — to
    # check the inference machinery still runs and gives reasonable coverage
    # even when the model is mis-specified vs the data-generating process.
    truth = ParticleParams(
        log_tau_R=jnp.asarray(-4.0),
        log_tau_F=jnp.asarray(-12.0),
        log_phi=jnp.asarray(2.5),
    )
    ds = simulate(jr.key(2), cfg, T, params=truth)
    assert bool(jnp.isfinite(ds.mu_y).all())

    result = run_liu_west_discrete(
        jr.key(11), cfg, ds.y, n_particles=N, h=0.1, fixed_lag_L=14
    )

    # No NaNs.
    assert bool(jnp.isfinite(result.log_lik).all())
    assert bool(jnp.isfinite(result.particles_history.log_Rt).all())
    assert bool(jnp.isfinite(result.particles_history.v_R).all())
    assert bool(jnp.isfinite(result.params_history.log_sigma_vR).all())
    assert bool(jnp.isfinite(result.params_history.log_mu).all())

    # log_Rt 90% coverage should be reasonable.
    log_Rt_p = result.particles_history.log_Rt
    lw = result.log_weights_history
    lo = jax.vmap(weighted_quantile, in_axes=(0, 0, None))(log_Rt_p, lw, jnp.asarray(0.05))
    hi = jax.vmap(weighted_quantile, in_axes=(0, 0, None))(log_Rt_p, lw, jnp.asarray(0.95))
    cov = float(jnp.mean((ds.log_Rt >= lo) & (ds.log_Rt <= hi)))
    assert cov >= 0.55, f"log_Rt 90% coverage = {cov:.3f}"


def test_model_discrete_I_is_integer_valued():
    """The Poisson draw should produce integer-valued I_buf entries."""
    cfg = default_config()
    T = 30
    N = 200
    truth = ParticleParams(
        log_tau_R=jnp.asarray(-4.0),
        log_tau_F=jnp.asarray(-12.0),
        log_phi=jnp.asarray(2.5),
    )
    ds = simulate(jr.key(2), cfg, T, params=truth)
    result = run_liu_west_discrete(
        jr.key(11), cfg, ds.y, n_particles=N, h=0.1, fixed_lag_L=14
    )
    # I_buf[1:, :, 0] is the most-recent infection per particle, post-prior
    # (t=0 row comes from prior_sample with continuous I_buf init).  Every
    # subsequent row should be an integer-valued Poisson draw.
    I_recent = result.particles_history.I_buf[1:, :, 0]
    rounded = jnp.round(I_recent)
    assert bool(jnp.allclose(I_recent, rounded, atol=1e-6)), (
        "I_buf entries are not integer-valued — Poisson sampling broken"
    )
