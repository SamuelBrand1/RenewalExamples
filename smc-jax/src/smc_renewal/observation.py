"""Observation model: NegBin likelihood and moment-matched Gaussian for the UKF.

NegBin2 parametrization: ``y ~ NB(mean=mu, concentration=phi)`` with
variance ``mu + mu^2 / phi``.  This matches ``numpyro.distributions.NegativeBinomial2``
and ``pyrenew.observation.noise.NegativeBinomialNoise`` (which wraps it).
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from jax.scipy.special import gammaln


def negbin_loglik(y: Array, mu: Array, phi: Array) -> Array:
    """Log-pmf of NegativeBinomial2(mu, concentration=phi) at integer ``y``.

    Uses the NB2 form:

        log p(y | mu, phi)
          = gammaln(y + phi) - gammaln(phi) - gammaln(y + 1)
            + phi * (log phi - log(phi + mu))
            + y   * (log mu  - log(phi + mu))

    Equivalent to ``numpyro.distributions.NegativeBinomial2(mean=mu, concentration=phi).log_prob(y)``.
    """
    mu = jnp.maximum(mu, 1e-12)
    phi = jnp.maximum(phi, 1e-12)
    y_f = y.astype(mu.dtype)
    log_phi = jnp.log(phi)
    log_phi_plus_mu = jnp.log(phi + mu)
    return (
        gammaln(y_f + phi)
        - gammaln(phi)
        - gammaln(y_f + 1.0)
        + phi * (log_phi - log_phi_plus_mu)
        + y_f * (jnp.log(mu) - log_phi_plus_mu)
    )


def gaussian_obs_moments(mu: Array, phi: Array) -> tuple[Array, Array]:
    """Moment-matched Gaussian approximation to NB2(mu, phi).

    Mean = mu, variance = mu + mu^2 / phi.  Used as the observation model
    inside the UKF.
    """
    mu = jnp.maximum(mu, 1e-12)
    phi = jnp.maximum(phi, 1e-12)
    return mu, mu + mu * mu / phi
