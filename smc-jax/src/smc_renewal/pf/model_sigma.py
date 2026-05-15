"""Model B: Liu-West applied to (log_σ_R, log_σ_F, log_φ) directly.

Compared with the original model (``pf/model.py``), the doubly-stochastic
nested-RW structure is collapsed by one level:

  - **Model A** (``pf/model.py``): ``log Rt`` walks with ``σ_R`` (state),
    ``log σ_R`` walks with ``τ_R`` (static), Liu-West on ``log τ_R``.
  - **Model B** (this file): ``log Rt`` walks with ``σ_R`` (Liu-West
    parameter, slowly evolving via shrink-jitter dynamics).  No ``τ``,
    no nested RW.

Methodologically the second design is cleaner when the data signal on
``τ`` is weak (which it usually is): Liu-West gets to work on the strongly-
identified ``log σ`` rather than the weakly-identified ``log τ``.  The
shrinkage parameter ``h`` plays the role ``τ`` played in Model A.

Same observation model and renewal+feedback dynamics as Model A; the only
change is in the latent volatility hierarchy.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import jax.random as jr
from jax import Array
from pfjax import BaseModel

from smc_renewal.config import ModelConfig
from smc_renewal.observation import negbin_loglik
from smc_renewal.transition import _floor, _pad_pmf, expected_observation


class ParticleStateSigma(NamedTuple):
    """Latent state for Model B (no log_σ_R / log_σ_F — those are in params)."""

    log_Rt: Array
    log_F: Array
    log_I0: Array
    I_buf: Array


class ParticleParamsSigma(NamedTuple):
    """Liu-West parameters for Model B: directly the latent volatilities + φ."""

    log_sigma_R: Array
    log_sigma_F: Array
    log_phi: Array


class TransitionNoiseSigma(NamedTuple):
    """Two standard-normal innovations per step (no inner η-noises)."""

    eps_R: Array  # drives log Rt
    eps_F: Array  # drives log F


def step_sigma(
    state: ParticleStateSigma,
    params: ParticleParamsSigma,
    noise: TransitionNoiseSigma,
    cfg: ModelConfig,
) -> ParticleStateSigma:
    """One step of Model B's generative dynamics.

    ``σ_R`` and ``σ_F`` come from ``params`` (per-particle, slowly drifting
    via Liu-West outside this step).  Renewal+feedback identical to Model A.
    """
    L = cfg.buffer_len
    g_pad = _pad_pmf(cfg.generation_interval, L)

    sigma_R = _floor(params.log_sigma_R, cfg.sigma_floor)
    sigma_F = _floor(params.log_sigma_F, cfg.sigma_floor)

    log_Rt_new = state.log_Rt + sigma_R * noise.eps_R
    log_F_new = state.log_F + sigma_F * noise.eps_F

    g_conv_I = jnp.dot(g_pad, state.I_buf)
    F_new = jnp.exp(log_F_new)

    log_Rt_clipped = jnp.clip(log_Rt_new, -20.0, 20.0)
    exponent = jnp.clip(log_Rt_clipped - F_new * g_conv_I, -20.0, 20.0)
    Rt_eff = jnp.exp(exponent)
    I_new = jnp.minimum(Rt_eff * g_conv_I, 1e15)
    I_buf_new = jnp.concatenate([I_new[jnp.newaxis], state.I_buf[:-1]])

    return ParticleStateSigma(
        log_Rt=log_Rt_new,
        log_F=log_F_new,
        log_I0=state.log_I0,
        I_buf=I_buf_new,
    )


def expected_observation_sigma(state: ParticleStateSigma, cfg: ModelConfig) -> Array:
    """Expected observed cases at time t given the post-step I_buf."""
    d_pad = _pad_pmf(cfg.delay_pmf, cfg.buffer_len)
    return jnp.dot(d_pad, state.I_buf)


class RenewalModelSigma(BaseModel):
    """``pfjax.BaseModel`` wrapper for Model B."""

    def __init__(self, cfg: ModelConfig):
        super().__init__(bootstrap=True)
        self.cfg = cfg

    def prior_sample(self, key: Array, theta: ParticleParamsSigma) -> ParticleStateSigma:
        cfg = self.cfg
        keys = jr.split(key, 3)
        log_Rt = cfg.init_log_Rt_mean + cfg.init_log_Rt_sd * jr.normal(keys[0])
        log_F = cfg.init_log_F_mean + cfg.init_log_F_sd * jr.normal(keys[1])
        log_I0 = cfg.init_log_I0_mean + cfg.init_log_I0_sd * jr.normal(keys[2])
        I_buf = jnp.full((cfg.buffer_len,), jnp.exp(log_I0))
        return ParticleStateSigma(
            log_Rt=log_Rt, log_F=log_F, log_I0=log_I0, I_buf=I_buf
        )

    def prior_lpdf(self, x_curr, theta) -> Array:
        cfg = self.cfg
        return (
            -0.5 * ((x_curr.log_Rt - cfg.init_log_Rt_mean) / cfg.init_log_Rt_sd) ** 2
            - 0.5 * ((x_curr.log_F - cfg.init_log_F_mean) / cfg.init_log_F_sd) ** 2
            - 0.5 * ((x_curr.log_I0 - cfg.init_log_I0_mean) / cfg.init_log_I0_sd) ** 2
        )

    def state_sample(
        self, key: Array, x_prev: ParticleStateSigma, theta: ParticleParamsSigma
    ) -> ParticleStateSigma:
        eps = jr.normal(key, (2,))
        noise = TransitionNoiseSigma(eps_R=eps[0], eps_F=eps[1])
        return step_sigma(x_prev, theta, noise, self.cfg)

    def state_lpdf(self, x_curr, x_prev, theta) -> Array:
        cfg = self.cfg
        sigma_R = jnp.maximum(jnp.exp(theta.log_sigma_R), cfg.sigma_floor)
        sigma_F = jnp.maximum(jnp.exp(theta.log_sigma_F), cfg.sigma_floor)

        def _logN(x, mu, sd):
            return -0.5 * jnp.log(2.0 * jnp.pi * sd * sd) - 0.5 * ((x - mu) / sd) ** 2

        return (
            _logN(x_curr.log_Rt, x_prev.log_Rt, sigma_R)
            + _logN(x_curr.log_F, x_prev.log_F, sigma_F)
        )

    def meas_lpdf(
        self, y_curr: Array, x_curr: ParticleStateSigma, theta: ParticleParamsSigma
    ) -> Array:
        mu_y = expected_observation_sigma(x_curr, self.cfg)
        phi = jnp.exp(theta.log_phi)
        return negbin_loglik(jnp.asarray(y_curr), mu_y, phi).reshape(())
