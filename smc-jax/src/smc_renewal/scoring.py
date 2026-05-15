"""Forecast skill scoring: weighted CRPS and interval coverage.

CRPS is the standard "continuous" version (proper score on the full predictive
distribution).  We use the ensemble form

    CRPS(F, y) ≈ E_F|X - y| - 0.5 * E_F|X - X'|,

which is exact for the empirical distribution defined by the (weighted)
particle sample.  This matches ``properscoring.crps_ensemble`` for equal weights.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from jax import Array


def _normalize_weights(log_weights: Array) -> Array:
    lw = log_weights - jnp.max(log_weights)
    w = jnp.exp(lw)
    return w / w.sum()


def weighted_quantile(values: Array, log_weights: Array, q: float | Array) -> Array:
    """Quantile ``q`` of a 1-D weighted empirical distribution."""
    w = _normalize_weights(log_weights)
    order = jnp.argsort(values)
    v_sorted = values[order]
    w_sorted = w[order]
    cumw = jnp.cumsum(w_sorted)
    idx = jnp.searchsorted(cumw, jnp.asarray(q)).clip(0, values.shape[0] - 1)
    return v_sorted[idx]


def crps_weighted(y_true: Array, y_samples: Array, log_weights: Array) -> Array:
    """Weighted ensemble CRPS for a single observation.

    ``y_true`` scalar; ``y_samples`` shape ``(N,)``; ``log_weights`` shape ``(N,)``.
    """
    w = _normalize_weights(log_weights)
    term1 = jnp.sum(w * jnp.abs(y_samples - y_true))
    diff = jnp.abs(y_samples[:, None] - y_samples[None, :])
    term2 = 0.5 * jnp.sum(w[:, None] * w[None, :] * diff)
    return term1 - term2


def crps_weighted_batch(y_true: Array, y_samples: Array, log_weights: Array) -> Array:
    """Batched CRPS.

    ``y_true`` shape ``(H,)``; ``y_samples`` shape ``(H, N)``; ``log_weights`` shape ``(N,)``.
    Returns shape ``(H,)``.
    """
    return jax.vmap(lambda yt, ys: crps_weighted(yt, ys, log_weights))(y_true, y_samples)


def interval_coverage(
    y_true: Array, y_samples: Array, log_weights: Array, level: float = 0.5
) -> Array:
    """Indicator: is ``y_true`` inside the central ``level``-interval?

    Vectorized over leading dims of ``y_true`` (paired with ``y_samples`` first axis).
    """
    alpha = (1.0 - level) / 2.0

    def _one(yt, ys):
        lo = weighted_quantile(ys, log_weights, alpha)
        hi = weighted_quantile(ys, log_weights, 1.0 - alpha)
        return ((yt >= lo) & (yt <= hi)).astype(jnp.float64)

    return jax.vmap(_one)(y_true, y_samples)
