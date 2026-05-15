"""Fixed/known PMFs: generation interval, infection-feedback, reporting delay.

Defaults are covid/flu-like: short-tailed discretized Gamma on integer day support
[1..L] (or [0..L-1] for the delay).  Pure-JAX so they can be jit'd.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from jax.scipy.special import gammaln


def _discretized_gamma(shape: float, scale: float, support: Array) -> Array:
    """Probability mass on integer points `support` from a Gamma(shape, scale), normalized.

    Uses the unnormalized Gamma density at the integer points and renormalizes.
    Sufficient for the short PMFs we use here.
    """
    x = support.astype(jnp.float32)
    log_dens = (shape - 1.0) * jnp.log(jnp.maximum(x, 1e-12)) - x / scale - shape * jnp.log(scale) - gammaln(shape)
    dens = jnp.exp(log_dens)
    dens = jnp.where(x > 0, dens, 0.0)
    return dens / jnp.sum(dens)


def default_generation_interval(max_lag: int = 14) -> Array:
    """Discretized Gamma(shape=3.0, scale=1.5) on [1..max_lag]; mean ~ 4.5 days."""
    return _discretized_gamma(3.0, 1.5, jnp.arange(1, max_lag + 1))


def default_feedback_pmf(max_lag: int = 14) -> Array:
    """Feedback weights track recent incidence — narrower than the generation interval.

    Discretized Gamma(shape=2.0, scale=2.0) on [1..max_lag]; mean ~ 4.0 days.
    """
    return _discretized_gamma(2.0, 2.0, jnp.arange(1, max_lag + 1))


def default_delay_pmf(max_lag: int = 14) -> Array:
    """Reporting delay (infection → observation): on [0..max_lag-1].

    Discretized Gamma(shape=2.5, scale=2.5) on [1..max_lag] shifted to [0..max_lag-1];
    mean ~ 6.25 days, with mass at zero day to allow same-day reports.
    """
    raw = _discretized_gamma(2.5, 2.5, jnp.arange(1, max_lag + 1))
    return raw
