"""Sequential PF update: extend an existing Liu-West cloud with new observations.

This module is the Model-A face of ``extend_liu_west_core`` (in
``pf/_runner_core.py``).  Variant-specific sequential updates for Models
B/C/D can be added by wrapping the core in the same shape as ``extend_liu_west``
below.
"""

from __future__ import annotations

from typing import TypeVar

import jax
from jax import Array

from smc_renewal.config import ModelConfig
from smc_renewal.pf._runner_core import extend_liu_west_core
from smc_renewal.pf.model import RenewalModel
from smc_renewal.pf.runner import LiuWestResult
from smc_renewal.state import ParticleParams, ParticleState

StateT = TypeVar("StateT")
ParamsT = TypeVar("ParamsT")


def extend_liu_west(
    key: Array,
    cfg: ModelConfig,
    y_new: Array,
    particles: ParticleState,
    params: ParticleParams,
    log_weights: Array,
    h: float | None = None,
    ess_threshold: float | None = None,
) -> LiuWestResult:
    """Continue a Liu-West bootstrap PF on new observations ``y_new``."""
    if h is None:
        h = cfg.liu_west_h
    if ess_threshold is None:
        ess_threshold = log_weights.shape[0] / 2.0

    out = extend_liu_west_core(
        key,
        cfg,
        y_new,
        model=RenewalModel(cfg),
        particles=particles,
        params=params,
        log_weights=log_weights,
        h=h,
        ess_threshold=ess_threshold,
    )
    return LiuWestResult(*out)


def _slice_final(result) -> tuple[StateT, ParamsT, Array]:
    """Return ``(final_particles, final_params, final_log_weights)``.

    Generic over the variant Result NamedTuple — any
    ``LiuWest{,Sigma,Trend,Discrete}Result`` works because the slicing only
    touches ``particles_history``, ``params_history`` and ``log_weights_history``.
    """
    final_particles = jax.tree.map(lambda x: x[-1], result.particles_history)
    final_params = jax.tree.map(lambda x: x[-1], result.params_history)
    final_lw = result.log_weights_history[-1]
    return final_particles, final_params, final_lw
