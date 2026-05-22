"""SMC^2 top-level entry — convenience wrappers around ``tempered`` and ``update``."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array

from smc_renewal.config import ModelConfig
from smc_renewal.smc2.tempered import (
    SMC2Result,
    _theta_to_params,
    run_smc2,
)
from smc_renewal.smc2.ukf import marginal_log_likelihood as _default_marginal_loglik
from smc_renewal.smc2.update import extend_smc2


class SMC2Cache(NamedTuple):
    """Posterior + cached per-particle log-lik over the data fitted so far."""

    result: SMC2Result
    cached_log_lik: Array  # (N,) — log p(y_{1..T_fitted} | theta_i)
    T_fitted: int


def fit_smc2(
    key: Array,
    cfg: ModelConfig,
    y: Array,
    n_particles: int = 128,
    marginal_loglik_fn=_default_marginal_loglik,
    **kwargs,
) -> SMC2Cache:
    """Fit SMC^2 from scratch on ``y`` and cache per-particle log-lik.

    ``marginal_loglik_fn`` defaults to the hand-rolled UKF.  Pass
    ``smc_renewal.smc2.ekf_cuthbert.marginal_log_likelihood`` to use the
    library-backed EKF instead.
    """
    result = run_smc2(
        key, cfg, y, n_particles=n_particles,
        marginal_loglik_fn=marginal_loglik_fn, **kwargs,
    )
    cached = jax.vmap(
        lambda th: marginal_loglik_fn(cfg, y, _theta_to_params(th))
    )(result.particles)
    return SMC2Cache(result=result, cached_log_lik=cached, T_fitted=int(y.shape[0]))


def extend_fit(
    key: Array,
    cfg: ModelConfig,
    y_full: Array,
    cache: SMC2Cache,
    marginal_loglik_fn=_default_marginal_loglik,
    **kwargs,
) -> SMC2Cache:
    """Update an existing SMC^2 cache with the longer series ``y_full``."""
    new_result, new_cached = extend_smc2(
        key, cfg, y_full, cache.cached_log_lik, cache.result,
        marginal_loglik_fn=marginal_loglik_fn, **kwargs,
    )
    return SMC2Cache(
        result=new_result, cached_log_lik=new_cached, T_fitted=int(y_full.shape[0])
    )
