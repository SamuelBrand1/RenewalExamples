"""Verify our NegBin loglik matches NumPyro's NegativeBinomial2 to machine precision."""

from __future__ import annotations

import jax.numpy as jnp
import numpyro.distributions as dist

from smc_renewal.observation import gaussian_obs_moments, negbin_loglik


def test_negbin_loglik_matches_numpyro_nb2():
    mu = jnp.array([1.0, 5.0, 25.0, 100.0, 1000.0], dtype=jnp.float64)
    phi = jnp.array([0.5, 1.0, 5.0, 25.0, 200.0], dtype=jnp.float64)
    y = jnp.array([0, 3, 10, 90, 1100], dtype=jnp.float64)

    ours = negbin_loglik(y, mu, phi)
    theirs = dist.NegativeBinomial2(mean=mu, concentration=phi).log_prob(y)

    assert jnp.allclose(ours, theirs, atol=1e-5, rtol=1e-5), (
        f"max abs diff: {jnp.max(jnp.abs(ours - theirs))}"
    )


def test_gaussian_obs_moments_match_negbin_moments():
    mu = jnp.array([5.0, 50.0])
    phi = jnp.array([2.0, 10.0])
    m, v = gaussian_obs_moments(mu, phi)
    assert jnp.allclose(m, mu)
    assert jnp.allclose(v, mu + mu * mu / phi)
