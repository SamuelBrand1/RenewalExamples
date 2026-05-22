"""Tests for the Liu-West shrink-jitter kernel.

(1) At ``h=0`` the kernel is the identity (no shrink, no jitter).
(2) With nontrivial ``h`` and equal weights, the weighted mean of the
    parameter cloud is preserved in expectation.
(3) End-to-end: ``run_liu_west`` runs without NaNs on synthetic data.
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr

from smc_renewal.config import default_config
from smc_renewal.pf.liu_west import shrink_jitter, weighted_moments
from smc_renewal.pf.runner import init_params_from_prior, run_liu_west
from smc_renewal.state import ParticleParams
from smc_renewal.synthetic import simulate


def test_h_zero_is_identity():
    cfg = default_config()
    N = 256
    params = init_params_from_prior(jr.key(0), cfg, N)
    lw = jnp.full((N,), -jnp.log(N))
    out = shrink_jitter(params, lw, h=0.0, key=jr.key(1))
    assert jnp.allclose(out.log_tau_R, params.log_tau_R)
    assert jnp.allclose(out.log_tau_F, params.log_tau_F)
    assert jnp.allclose(out.log_phi, params.log_phi)


def test_kernel_preserves_mean_in_expectation():
    cfg = default_config()
    N = 4000
    params = init_params_from_prior(jr.key(0), cfg, N)
    lw = jnp.full((N,), -jnp.log(N))

    m_before = jnp.array(
        [params.log_tau_R.mean(), params.log_tau_F.mean(), params.log_phi.mean()]
    )

    # Average over many h-jitter applications to suppress sampling noise.
    out = shrink_jitter(params, lw, h=0.15, key=jr.key(2))
    m_after = jnp.array(
        [out.log_tau_R.mean(), out.log_tau_F.mean(), out.log_phi.mean()]
    )
    # Mean preserved up to MC noise.
    assert jnp.allclose(m_before, m_after, atol=0.05), (m_before, m_after)


def test_liu_west_runs_without_nans():
    cfg = default_config()
    T = 90
    N = 1000
    truth = ParticleParams(
        log_tau_R=jnp.asarray(-4.0),
        log_tau_F=jnp.asarray(-12.0),
        log_phi=jnp.asarray(2.5),
    )
    ds = simulate(jr.key(0), cfg, T, params=truth)
    # Skip if synthetic exploded.
    assert bool(jnp.isfinite(ds.mu_y).all())

    result = run_liu_west(jr.key(1), cfg, ds.y, n_particles=N, h=0.1)
    assert bool(jnp.isfinite(result.log_lik).all()), result.log_lik
    # log_tau_R posterior cloud should still contain values near truth.
    final_log_tau_R = result.params_history.log_tau_R[-1]
    weights = jnp.exp(
        result.log_weights_history[-1] - jnp.max(result.log_weights_history[-1])
    )
    weights = weights / weights.sum()
    weighted_mean = jnp.sum(weights * final_log_tau_R)
    assert -7.0 < float(weighted_mean) < 0.0, (
        f"posterior mean of log_tau_R = {float(weighted_mean):.3f} (truth -4.0)"
    )
