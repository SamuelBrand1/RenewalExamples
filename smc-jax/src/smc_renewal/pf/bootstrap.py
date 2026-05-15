"""Fixed-theta bootstrap particle filter — thin wrapper over ``pfjax``.

For the Liu-West variant (per-particle theta with shrink-jitter), see
``pf/runner.py``.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr
import pfjax
from jax import Array
from pfjax.particle_resamplers import resample_multinomial

from smc_renewal.config import ModelConfig
from smc_renewal.pf.model import RenewalModel
from smc_renewal.state import ParticleParams, ParticleState


class PFResult(NamedTuple):
    """Filtering output from a fixed-theta pfjax run.

    ``x_particles`` and ``ancestors`` come straight from pfjax; pass them to
    ``pfjax.particle_smooth`` to draw smoothed-trajectory samples.

    Shapes (T = len(y_meas)):
        x_particles: ParticleState batched (T, N, ...)
        logw:        (T, N)  -- post-resample log-weights at each time step
        ancestors:   (T-1, N) -- ancestor index from t-1 to t
        log_lik:     scalar marginal log-likelihood
    """

    x_particles: ParticleState
    logw: Array
    ancestors: Array
    log_lik: Array
    resample_out: dict


def weighted_quantile(values: Array, log_weights: Array, q: float | Array) -> Array:
    """Weighted quantile of a 1-D ``values`` array with given log-weights."""
    lZ = jax.scipy.special.logsumexp(log_weights)
    w = jnp.exp(log_weights - lZ)
    order = jnp.argsort(values)
    v_sorted = values[order]
    w_sorted = w[order]
    cumw = jnp.cumsum(w_sorted)
    idx = jnp.searchsorted(cumw, jnp.asarray(q)).clip(0, values.shape[0] - 1)
    return v_sorted[idx]


def run_pf(
    key: Array,
    cfg: ModelConfig,
    y: Array,
    params: ParticleParams,
    n_particles: int = 1000,
) -> PFResult:
    """Bootstrap PF over ``y`` with fixed ``params``, via ``pfjax.particle_filter``.

    pfjax expects ``y_meas`` with leading dim ``T`` (the initial measurement is
    treated specially), so we reshape and route through its standard interface.
    Uses multinomial resampling — required for compatibility with
    ``pfjax.particle_smooth``.
    """
    resampler = resample_multinomial
    model = RenewalModel(cfg)
    # pfjax's particle_filter expects y_meas as a leading-time array.  Its
    # interface treats y_meas[0] as the "initial" measurement passed to pf_init,
    # and y_meas[1:] as the subsequent measurements driving pf_step.
    out = pfjax.particle_filter(
        model, key, y_meas=jnp.asarray(y), theta=params,
        n_particles=n_particles, resampler=resampler, history=True,
    )
    # pfjax returns a dict; canonical keys are x_particles, logw, ancestors,
    # loglik (sometimes "loglik" key, depending on version).
    return PFResult(
        x_particles=out["x_particles"],
        logw=out["logw"],
        ancestors=out["resample_out"]["ancestors"],
        log_lik=jnp.asarray(out["loglik"]),
        resample_out=out["resample_out"],
    )
