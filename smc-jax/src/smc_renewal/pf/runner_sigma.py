"""Liu-West bootstrap PF for Model B (Liu-West on log_σ directly).

Same structure as ``pf/runner.py::run_liu_west``, but:
  - The Liu-West parameter cloud is ``(log_σ_R, log_σ_F, log_φ)`` per particle.
  - The latent state has no log_σ fields (those are now Liu-West params).
  - Transition is via ``model_sigma.step_sigma``.

Thin wrapper over ``pf/_runner_core.py``.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.random as jr
from jax import Array

from smc_renewal.config import ModelConfig
from smc_renewal.pf._runner_core import run_liu_west_core
from smc_renewal.pf.model_sigma import (
    ParticleParamsSigma,
    ParticleStateSigma,
    RenewalModelSigma,
)


class LiuWestSigmaResult(NamedTuple):
    """History from Model B fit.

    Shapes (T = len(y)):
        particles_history:    ParticleStateSigma batched (T, N, ...)
        params_history:       ParticleParamsSigma batched (T, N)  -- log_σ_R, log_σ_F, log_φ
        log_weights_history:  (T, N)
        ancestors:            (T-1, N) for particle_smooth
        ess_history:          (T,)
        log_lik:              (T,) cumulative marginal log-likelihood
    """

    particles_history: ParticleStateSigma
    params_history: ParticleParamsSigma
    log_weights_history: Array
    ancestors: Array
    ess_history: Array
    log_lik: Array


def _sample_initial_sigmas(
    key: Array, cfg: ModelConfig, N: int
) -> ParticleParamsSigma:
    """Draw initial (log_σ_R, log_σ_F, log_φ) cloud from cfg priors."""
    keys = jr.split(key, 3)
    return ParticleParamsSigma(
        log_sigma_R=cfg.init_log_sigma_R_mean
        + cfg.init_log_sigma_R_sd * jr.normal(keys[0], (N,)),
        log_sigma_F=cfg.init_log_sigma_F_mean
        + cfg.init_log_sigma_F_sd * jr.normal(keys[1], (N,)),
        log_phi=cfg.prior_log_phi_mean + cfg.prior_log_phi_sd * jr.normal(keys[2], (N,)),
    )


def run_liu_west_sigma(
    key: Array,
    cfg: ModelConfig,
    y: Array,
    n_particles: int = 2000,
    h: float | None = None,
    ess_threshold: float | None = None,
    fixed_lag_L: int = 1,
) -> LiuWestSigmaResult:
    """Bootstrap PF with Liu-West on (log_σ_R, log_σ_F, log_φ) — Model B."""
    if h is None:
        h = cfg.liu_west_h
    if ess_threshold is None:
        ess_threshold = n_particles / 2.0

    out = run_liu_west_core(
        key,
        cfg,
        y,
        model=RenewalModelSigma(cfg),
        sample_initial_params=_sample_initial_sigmas,
        n_particles=n_particles,
        h=h,
        ess_threshold=ess_threshold,
        fixed_lag_L=fixed_lag_L,
    )
    return LiuWestSigmaResult(*out)
