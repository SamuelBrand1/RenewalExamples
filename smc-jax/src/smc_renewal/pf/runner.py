"""Liu-West bootstrap particle filter, Model A (nested-RW volatility).

The Liu-West parameter cloud carries ``(log_tau_R, log_tau_F, log_phi)`` per
particle; the volatilities ``log_sigma_R`` and ``log_sigma_F`` are part of the
latent state and evolve as nested random walks with rates ``tau_R``, ``tau_F``.

This module is a thin per-variant wrapper over the shared loop in
``pf/_runner_core.py``.  Sibling modules ``runner_sigma``, ``runner_trend``,
``runner_discrete`` follow the same pattern for Models B / C / D.

The core uses ``pfjax``-shaped ``RenewalModel`` primitives and tracks ancestor
indices throughout so ``pfjax.particle_smooth`` can be applied to the result
for trajectory smoothing.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.random as jr
from jax import Array

from smc_renewal.config import ModelConfig
from smc_renewal.pf._runner_core import _ess, _multinomial_indices, run_liu_west_core
from smc_renewal.pf.model import RenewalModel
from smc_renewal.state import ParticleParams, ParticleState

# Re-export the legacy private names so ``pf/update.py`` keeps importing them
# from here without churn.
__all__ = [
    "LiuWestResult",
    "init_params_from_prior",
    "run_liu_west",
    "_ess",
    "_multinomial_indices",
]


class LiuWestResult(NamedTuple):
    """History of particles + parameter cloud across time, pfjax-compatible.

    Shapes (T = len(y)):
        particles_history:    ParticleState batched (T, N, ...)
        params_history:       ParticleParams batched (T, N)
        log_weights_history:  (T, N) post-resample log-weights
        ancestors:            (T-1, N) ancestor index from t-1 to t (for particle_smooth)
        ess_history:          (T,)
        log_lik:              (T,) cumulative marginal log-likelihood
    """

    particles_history: ParticleState
    params_history: ParticleParams
    log_weights_history: Array
    ancestors: Array
    ess_history: Array
    log_lik: Array


def _sample_initial_thetas(key: Array, cfg: ModelConfig, N: int) -> ParticleParams:
    keys = jr.split(key, 3)
    return ParticleParams(
        log_tau_R=cfg.prior_log_tau_R_mean + cfg.prior_log_tau_R_sd * jr.normal(keys[0], (N,)),
        log_tau_F=cfg.prior_log_tau_F_mean + cfg.prior_log_tau_F_sd * jr.normal(keys[1], (N,)),
        log_phi=cfg.prior_log_phi_mean + cfg.prior_log_phi_sd * jr.normal(keys[2], (N,)),
    )


def run_liu_west(
    key: Array,
    cfg: ModelConfig,
    y: Array,
    n_particles: int = 2000,
    h: float | None = None,
    ess_threshold: float | None = None,
    fixed_lag_L: int = 1,
) -> LiuWestResult:
    """Liu-West-augmented bootstrap PF using ``RenewalModel`` primitives.

    Multinomial resampling is used throughout so the result is directly
    consumable by ``pfjax.particle_smooth``.

    Parameters
    ----------
    fixed_lag_L
        Steyn-style fixed-lag resampling window.  When resampling at time t,
        the *last* ``fixed_lag_L`` time-steps of state+param history are
        permuted by the ancestor indices (rather than just the current cloud).
        Default ``1`` is the standard bootstrap-PF behaviour.  Steyn-style
        values are around the mean observation delay (~14–30 for our model).

        Rationale: observations at time t are informative about the latent
        state from a few days ago (due to reporting delay + serial interval),
        not from much further back.  Restricting the resample window to
        ``L`` past steps avoids inducing path degeneracy by re-permuting
        history that the data can't sensibly update.
    """
    if h is None:
        h = cfg.liu_west_h
    if ess_threshold is None:
        ess_threshold = n_particles / 2.0

    out = run_liu_west_core(
        key,
        cfg,
        y,
        model=RenewalModel(cfg),
        sample_initial_params=_sample_initial_thetas,
        n_particles=n_particles,
        h=h,
        ess_threshold=ess_threshold,
        fixed_lag_L=fixed_lag_L,
    )
    return LiuWestResult(*out)


# Back-compat alias for the old name used elsewhere.
def init_params_from_prior(key: Array, cfg: ModelConfig, N: int) -> ParticleParams:
    return _sample_initial_thetas(key, cfg, N)
