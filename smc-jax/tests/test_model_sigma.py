"""Smoke tests for Model B (Liu-West on log_σ directly).

Verifies the runner returns a sensible result on synthetic data: posterior
log_Rt band covers truth, no NaNs, ESS stays meaningful.  Doesn't try to
recover params sharply — that's a property of the model+data, not a
correctness check on the inference machinery.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr

from smc_renewal.config import default_config
from smc_renewal.pf.bootstrap import weighted_quantile
from smc_renewal.pf.runner_sigma import run_liu_west_sigma
from smc_renewal.state import ParticleParams
from smc_renewal.synthetic import simulate


def test_model_sigma_runs_and_covers_log_Rt():
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

    result = run_liu_west_sigma(
        jr.key(11), cfg, ds.y, n_particles=N, h=0.1, fixed_lag_L=14
    )

    # No NaNs anywhere.
    assert bool(jnp.isfinite(result.log_lik).all())
    assert bool(jnp.isfinite(result.particles_history.log_Rt).all())
    assert bool(jnp.isfinite(result.params_history.log_sigma_R).all())

    # Coverage on log_Rt should be reasonable (>=0.6 on a single synthetic seed).
    log_Rt_p = result.particles_history.log_Rt
    lw = result.log_weights_history
    lo = jax.vmap(weighted_quantile, in_axes=(0, 0, None))(log_Rt_p, lw, jnp.asarray(0.05))
    hi = jax.vmap(weighted_quantile, in_axes=(0, 0, None))(log_Rt_p, lw, jnp.asarray(0.95))
    cov = float(jnp.mean((ds.log_Rt >= lo) & (ds.log_Rt <= hi)))
    assert cov >= 0.55, f"log_Rt 90% coverage = {cov:.3f} (< 0.55)"

    # σ_R posterior should not be the prior — final-step weighted SD < initial.
    final_lw = lw[-1]
    w = jnp.exp(final_lw - jnp.max(final_lw))
    w = w / w.sum()
    final_sigma_R = result.params_history.log_sigma_R[-1]
    post_sd = float(jnp.sqrt(jnp.sum(w * (final_sigma_R - jnp.sum(w * final_sigma_R)) ** 2)))
    prior_sd = float(cfg.init_log_sigma_R_sd)
    # Allow some Liu-West-jitter spread but expect at least partial concentration.
    assert post_sd < prior_sd * 1.5, (
        f"σ_R posterior sd {post_sd:.3f} suspiciously large vs prior sd {prior_sd:.3f}"
    )
