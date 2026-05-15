"""Smoke tests for the synthetic-data generator."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr

from smc_renewal.config import default_config
from smc_renewal.state import ParticleParams
from smc_renewal.synthetic import simulate


def test_simulate_shapes_and_positivity():
    cfg = default_config()
    T = 90
    fixed_params = ParticleParams(
        log_tau_R=jnp.asarray(-3.0),
        log_tau_F=jnp.asarray(-3.5),
        log_phi=jnp.asarray(2.5),
    )
    ds = simulate(jr.key(0), cfg, T, params=fixed_params)

    assert ds.log_Rt.shape == (T,)
    assert ds.log_F.shape == (T,)
    assert ds.infections.shape == (T,)
    assert ds.mu_y.shape == (T,)
    assert ds.y.shape == (T,)

    assert bool(jnp.all(ds.infections > 0)), "infections should stay positive"
    assert bool(jnp.all(ds.mu_y > 0)), "mu_y should stay positive"
    assert bool(jnp.all(ds.y >= 0)), "NegBin samples are non-negative"

    # Sanity: with default priors, infections should not explode wildly within 90d.
    assert bool(jnp.max(ds.infections) < 1e8)


def test_simulate_reproducible_with_same_key():
    cfg = default_config()
    T = 30
    params = ParticleParams(
        log_tau_R=jnp.asarray(-3.0),
        log_tau_F=jnp.asarray(-3.5),
        log_phi=jnp.asarray(2.5),
    )
    ds1 = simulate(jr.key(42), cfg, T, params=params)
    ds2 = simulate(jr.key(42), cfg, T, params=params)
    assert bool(jnp.allclose(ds1.y, ds2.y))
    assert bool(jnp.allclose(ds1.log_Rt, ds2.log_Rt))
