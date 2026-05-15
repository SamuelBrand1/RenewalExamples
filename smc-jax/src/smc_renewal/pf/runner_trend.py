"""Liu-West bootstrap PF for Model C (integrated Brownian motion on log Rt / log F).

Same structure as ``run_liu_west`` (Model A) and ``run_liu_west_sigma``
(Model B), but the latent state includes per-coordinate velocities (v_R, v_F)
and the Liu-West parameter cloud is ``(log σ_vR, log σ_vF, log φ)`` — the
volatilities of the velocity walks.

Thin wrapper over ``pf/_runner_core.py``.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.random as jr
from jax import Array

from smc_renewal.config import ModelConfig
from smc_renewal.pf._runner_core import run_liu_west_core
from smc_renewal.pf.model_trend import (
    ParticleParamsTrend,
    ParticleStateTrend,
    RenewalModelTrend,
)


class LiuWestTrendResult(NamedTuple):
    particles_history: ParticleStateTrend
    params_history: ParticleParamsTrend
    log_weights_history: Array
    ancestors: Array
    ess_history: Array
    log_lik: Array


def _sample_initial_trend_params(
    key: Array, cfg: ModelConfig, N: int
) -> ParticleParamsTrend:
    keys = jr.split(key, 3)
    return ParticleParamsTrend(
        log_sigma_vR=cfg.init_log_sigma_vR_mean
        + cfg.init_log_sigma_vR_sd * jr.normal(keys[0], (N,)),
        log_sigma_vF=cfg.init_log_sigma_vF_mean
        + cfg.init_log_sigma_vF_sd * jr.normal(keys[1], (N,)),
        log_phi=cfg.prior_log_phi_mean + cfg.prior_log_phi_sd * jr.normal(keys[2], (N,)),
    )


def run_liu_west_trend(
    key: Array,
    cfg: ModelConfig,
    y: Array,
    n_particles: int = 2000,
    h: float | None = None,
    ess_threshold: float | None = None,
    fixed_lag_L: int = 1,
) -> LiuWestTrendResult:
    """Bootstrap PF with Liu-West on (log σ_vR, log σ_vF, log φ) — Model C."""
    if h is None:
        h = cfg.liu_west_h
    if ess_threshold is None:
        ess_threshold = n_particles / 2.0

    out = run_liu_west_core(
        key,
        cfg,
        y,
        model=RenewalModelTrend(cfg),
        sample_initial_params=_sample_initial_trend_params,
        n_particles=n_particles,
        h=h,
        ess_threshold=ess_threshold,
        fixed_lag_L=fixed_lag_L,
    )
    return LiuWestTrendResult(*out)
