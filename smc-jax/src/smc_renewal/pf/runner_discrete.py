"""Liu-West bootstrap PF for Model D (discrete Poisson renewal + immigration).

Same overall shape as ``run_liu_west_trend`` — integrated-Brownian-motion
on log_Rt with velocity state — but with discrete (Poisson) latent
infections and a Liu-West cloud over ``(log_σ_vR, log_σ_vF, log_μ, log_φ)``.

Thin wrapper over ``pf/_runner_core.py``.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.random as jr
from jax import Array

from smc_renewal.config import ModelConfig
from smc_renewal.pf._runner_core import run_liu_west_core
from smc_renewal.pf.model_discrete import (
    ParticleParamsDiscrete,
    ParticleStateDiscrete,
    RenewalModelDiscrete,
)


class LiuWestDiscreteResult(NamedTuple):
    particles_history: ParticleStateDiscrete
    params_history: ParticleParamsDiscrete
    log_weights_history: Array
    ancestors: Array
    ess_history: Array
    log_lik: Array


def _sample_initial_discrete_params(
    key: Array, cfg: ModelConfig, N: int
) -> ParticleParamsDiscrete:
    keys = jr.split(key, 4)
    return ParticleParamsDiscrete(
        log_sigma_vR=cfg.init_log_sigma_vR_mean
        + cfg.init_log_sigma_vR_sd * jr.normal(keys[0], (N,)),
        log_sigma_vF=cfg.init_log_sigma_vF_mean
        + cfg.init_log_sigma_vF_sd * jr.normal(keys[1], (N,)),
        log_mu=cfg.init_log_mu_mean + cfg.init_log_mu_sd * jr.normal(keys[2], (N,)),
        log_phi=cfg.prior_log_phi_mean + cfg.prior_log_phi_sd * jr.normal(keys[3], (N,)),
    )


def run_liu_west_discrete(
    key: Array,
    cfg: ModelConfig,
    y: Array,
    n_particles: int = 2000,
    h: float | None = None,
    ess_threshold: float | None = None,
    fixed_lag_L: int = 1,
) -> LiuWestDiscreteResult:
    """Bootstrap PF with Liu-West on (log σ_vR, log σ_vF, log μ, log φ) — Model D."""
    if h is None:
        h = cfg.liu_west_h
    if ess_threshold is None:
        ess_threshold = n_particles / 2.0

    out = run_liu_west_core(
        key,
        cfg,
        y,
        model=RenewalModelDiscrete(cfg),
        sample_initial_params=_sample_initial_discrete_params,
        n_particles=n_particles,
        h=h,
        ess_threshold=ess_threshold,
        fixed_lag_L=fixed_lag_L,
    )
    return LiuWestDiscreteResult(*out)
