"""Observation model: NegBin likelihood and moment-matched Gaussian for the UKF.

NegBin2 parametrization: ``y ~ NB(mean=mu, concentration=phi)`` with
variance ``mu + mu^2 / phi``.  This matches ``numpyro.distributions.NegativeBinomial2``
and ``pyrenew.observation.noise.NegativeBinomialNoise`` (which wraps it).
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from numpyro.distributions import NegativeBinomial2


def negbin_loglik(y: Array, mu: Array, phi: Array) -> Array:
    """Log-pmf of NegativeBinomial2(mu, concentration=phi) at integer ``y``.

    Thin wrapper over ``numpyro.distributions.NegativeBinomial2``.  ``mu`` and
    ``phi`` are clamped away from zero so particles with a near-zero mean don't
    blow up the likelihood.
    """
    mu = jnp.maximum(mu, 1e-12)
    phi = jnp.maximum(phi, 1e-12)
    return NegativeBinomial2(mean=mu, concentration=phi).log_prob(y)


def gaussian_obs_moments(mu: Array, phi: Array) -> tuple[Array, Array]:
    """Moment-matched Gaussian approximation to NB2(mu, phi).

    Mean = mu, variance = mu + mu^2 / phi.  Used as the observation model
    inside the UKF.
    """
    mu = jnp.maximum(mu, 1e-12)
    phi = jnp.maximum(phi, 1e-12)
    return mu, mu + mu * mu / phi
