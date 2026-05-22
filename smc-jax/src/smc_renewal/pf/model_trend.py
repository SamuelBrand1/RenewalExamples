"""Model C: integrated-Brownian-motion dynamics on log Rt and log F.

Compared with Models A and B, the latent log-Rt and log-F processes are
**once-integrated**: their first derivative (velocity) walks, and the level
itself moves by the current velocity.  In discrete time:

  v_R[t]     = v_R[t-1]    + σ_vR · η_R[t],   η_R[t] ~ N(0,1) iid
  log Rt[t]  = log Rt[t-1] + v_R[t-1]

Analogously for log F (with its own velocity ``v_F`` and innovation std
``σ_vF``).  The renewal+feedback math is unchanged:

  I[t] = exp(log Rt[t] - exp(log F[t]) · conv(I, g)) · conv(I, g)

The forecast story differs sharply from Models A/B:
  - **Models A/B**: median log Rt forecast stays at current level (RW on
    log Rt).  Cases extrapolate at the *current growth rate*.
  - **Model C**: median log Rt forecast continues whatever velocity it
    currently has.  Cases extrapolate with potential *acceleration*.

Liu-West is applied to ``(log σ_vR, log σ_vF, log φ)`` — the per-step
volatility of the velocity walks.  Mirroring Model B's choice to put
Liu-West on the σ that the data is most informative about.
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


class ParticleStateTrend(NamedTuple):
    """Model C latent state: level + velocity for each of log Rt and log F."""

    log_Rt: Array
    v_R: Array        # d/dt log Rt — velocity / trend
    log_F: Array
    v_F: Array        # d/dt log F  — velocity / trend
    log_I0: Array
    I_buf: Array


class ParticleParamsTrend(NamedTuple):
    """Liu-West parameters for Model C: per-step volatilities of the velocities."""

    log_sigma_vR: Array   # std of η_R in the v_R walk
    log_sigma_vF: Array   # std of η_F in the v_F walk
    log_phi: Array        # NegBin overdispersion


class TransitionNoiseTrend(NamedTuple):
    """Standard-normal innovations driving the two velocity walks."""

    eta_R: Array   # drives v_R
    eta_F: Array   # drives v_F


def step_trend(
    state: ParticleStateTrend,
    params: ParticleParamsTrend,
    noise: TransitionNoiseTrend,
    cfg: ModelConfig,
) -> ParticleStateTrend:
    """One step of Model C.

    Order matches the standard local-linear-trend form:
      1. Step the level by the OLD velocity:    log Rt[t] = log Rt[t-1] + v_R[t-1]
      2. Then update the velocity:             v_R[t]    = v_R[t-1]    + σ_vR · η[t]
    The renewal+feedback math is identical to Models A/B.
    """
    L = cfg.buffer_len
    g_pad = _pad_pmf(cfg.generation_interval, L)

    sigma_vR = _floor(params.log_sigma_vR, cfg.sigma_floor)
    sigma_vF = _floor(params.log_sigma_vF, cfg.sigma_floor)

    # Step 1: level moves by the OLD velocity (deterministic, no level noise).
    log_Rt_new = state.log_Rt + state.v_R
    log_F_new = state.log_F + state.v_F

    # Step 2: velocity does a RW.
    v_R_new = state.v_R + sigma_vR * noise.eta_R
    v_F_new = state.v_F + sigma_vF * noise.eta_F

    # Renewal + feedback (same as Models A/B post-step values).
    g_conv_I = jnp.dot(g_pad, state.I_buf)
    F_new = jnp.exp(log_F_new)

    log_Rt_clipped = jnp.clip(log_Rt_new, -20.0, 20.0)
    exponent = jnp.clip(log_Rt_clipped - F_new * g_conv_I, -20.0, 20.0)
    Rt_eff = jnp.exp(exponent)
    I_new = jnp.minimum(Rt_eff * g_conv_I, 1e15)
    I_buf_new = jnp.concatenate([I_new[jnp.newaxis], state.I_buf[:-1]])

    return ParticleStateTrend(
        log_Rt=log_Rt_new,
        v_R=v_R_new,
        log_F=log_F_new,
        v_F=v_F_new,
        log_I0=state.log_I0,
        I_buf=I_buf_new,
    )


def expected_observation_trend(state: ParticleStateTrend, cfg: ModelConfig) -> Array:
    """Expected observed cases at time t given the post-step I_buf."""
    d_pad = _pad_pmf(cfg.delay_pmf, cfg.buffer_len)
    return jnp.dot(d_pad, state.I_buf)


class RenewalModelTrend(BaseModel):
    """pfjax.BaseModel wrapper for Model C."""

    def __init__(self, cfg: ModelConfig):
        super().__init__(bootstrap=True)
        self.cfg = cfg

    def prior_sample(
        self, key: Array, theta: ParticleParamsTrend
    ) -> ParticleStateTrend:
        cfg = self.cfg
        keys = jr.split(key, 5)
        log_Rt = cfg.init_log_Rt_mean + cfg.init_log_Rt_sd * jr.normal(keys[0])
        # Initial velocity: small (we don't know which direction Rt is trending).
        v_R = cfg.init_v_R_mean + cfg.init_v_R_sd * jr.normal(keys[1])
        log_F = cfg.init_log_F_mean + cfg.init_log_F_sd * jr.normal(keys[2])
        v_F = cfg.init_v_F_mean + cfg.init_v_F_sd * jr.normal(keys[3])
        log_I0 = cfg.init_log_I0_mean + cfg.init_log_I0_sd * jr.normal(keys[4])
        I_buf = jnp.full((cfg.buffer_len,), jnp.exp(log_I0))
        return ParticleStateTrend(
            log_Rt=log_Rt, v_R=v_R, log_F=log_F, v_F=v_F,
            log_I0=log_I0, I_buf=I_buf,
        )

    def prior_lpdf(self, x_curr, theta) -> Array:
        cfg = self.cfg
        return (
            -0.5 * ((x_curr.log_Rt - cfg.init_log_Rt_mean) / cfg.init_log_Rt_sd) ** 2
            - 0.5 * ((x_curr.v_R - cfg.init_v_R_mean) / cfg.init_v_R_sd) ** 2
            - 0.5 * ((x_curr.log_F - cfg.init_log_F_mean) / cfg.init_log_F_sd) ** 2
            - 0.5 * ((x_curr.v_F - cfg.init_v_F_mean) / cfg.init_v_F_sd) ** 2
            - 0.5 * ((x_curr.log_I0 - cfg.init_log_I0_mean) / cfg.init_log_I0_sd) ** 2
        )

    def state_sample(
        self, key: Array, x_prev: ParticleStateTrend, theta: ParticleParamsTrend
    ) -> ParticleStateTrend:
        eta = jr.normal(key, (2,))
        noise = TransitionNoiseTrend(eta_R=eta[0], eta_F=eta[1])
        return step_trend(x_prev, theta, noise, self.cfg)

    def state_lpdf(self, x_curr, x_prev, theta) -> Array:
        cfg = self.cfg
        sigma_vR = jnp.maximum(jnp.exp(theta.log_sigma_vR), cfg.sigma_floor)
        sigma_vF = jnp.maximum(jnp.exp(theta.log_sigma_vF), cfg.sigma_floor)

        def _logN(x, mu, sd):
            return -0.5 * jnp.log(2.0 * jnp.pi * sd * sd) - 0.5 * ((x - mu) / sd) ** 2

        # Only v_R and v_F are stochastic; level updates are deterministic
        # given the previous state.  The level coordinates contribute a Dirac
        # (= 0 here) — non-strict density on the manifold of (level, velocity).
        return (
            _logN(x_curr.v_R, x_prev.v_R, sigma_vR)
            + _logN(x_curr.v_F, x_prev.v_F, sigma_vF)
        )

    def meas_lpdf(
        self, y_curr: Array, x_curr: ParticleStateTrend, theta: ParticleParamsTrend
    ) -> Array:
        mu_y = expected_observation_trend(x_curr, self.cfg)
        phi = jnp.exp(theta.log_phi)
        return negbin_loglik(jnp.asarray(y_curr), mu_y, phi).reshape(())
