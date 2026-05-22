"""The cuthbert-backed EKF and our hand-rolled UKF should give close marginal log-likelihoods.

For our model (mild nonlinearity — one ``exp(gamma * feedback_term)`` per step)
the EKF and UKF should agree to within a percent or so on the marginal log-lik.
This test asserts that, which establishes the cuthbert EKF as a valid drop-in
``marginal_loglik_fn`` for ``smc2.run_smc2``.
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr

from smc_renewal.config import default_config
from smc_renewal.smc2.ekf_cuthbert import marginal_log_likelihood as ekf_ll
from smc_renewal.smc2.ukf import marginal_log_likelihood as ukf_ll
from smc_renewal.state import ParticleParams
from smc_renewal.synthetic import simulate


def test_ekf_agrees_with_ukf_on_synthetic():
    cfg = default_config()
    truth = ParticleParams(
        log_tau_R=jnp.asarray(-4.0),
        log_tau_F=jnp.asarray(-12.0),
        log_phi=jnp.asarray(2.5),
    )
    ds = simulate(jr.key(2), cfg, 80, params=truth)
    assert bool(jnp.isfinite(ds.mu_y).all())

    # At several thetas in a reasonable neighbourhood of truth.
    for log_phi in [2.0, 2.5, 3.0]:
        th = truth._replace(log_phi=jnp.asarray(log_phi))
        u = float(ukf_ll(cfg, ds.y, th))
        e = float(ekf_ll(cfg, ds.y, th))
        rel = abs(u - e) / max(abs(u), 1.0)
        assert rel < 0.02, f"EKF/UKF disagree at log_phi={log_phi}: ukf={u:.3f}, ekf={e:.3f}"
