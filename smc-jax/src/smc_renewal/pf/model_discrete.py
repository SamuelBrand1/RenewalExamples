"""Model D: discrete (Poisson) renewal with F-feedback and immigration.

Compared with Model C, the latent infections are **stochastic per step** —
``I[t] ~ Poisson(λ[t])`` rather than the deterministic renewal-equation
output — and a small immigration rate ``μ`` is added to prevent extinction:

    v_R[t]      = v_R[t-1]     + σ_vR · η_R[t]
    log Rt[t]   = log Rt[t-1]  + v_R[t-1]
    v_F[t]      = v_F[t-1]     + σ_vF · η_F[t]
    log F[t]    = log F[t-1]   + v_F[t-1]
    λ[t]        = μ + exp(log Rt[t]) · exp(-exp(log F[t]) · conv(I, g)) · conv(I, g)
    I[t]        ~ Poisson(λ[t])
    μ_y[t]      = Σ_s d_s · I[t-s]
    y[t]        ~ NegBin(μ_y[t], φ)

So Model D = Model C + Poisson I + immigration μ.  F-feedback (positive F,
``exp(-F · conv)``) preserves the susceptibility-depletion / strain-evolution
interpretation and keeps the discrete-renewal dynamics from running away
exponentially.

Liu-West parameters: **4-dimensional** ``(log σ_vR, log σ_vF, log μ, log φ)``.

The methods point of this variant is to demonstrate that the bootstrap PF
handles the discrete-count latent process just as comfortably as the
deterministic-renewal variants — the integer-valued ``I_buf`` and the
Poisson per-step draw don't change anything fundamental about the SMC
mechanics.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import jax.random as jr
from jax import Array
from jax.scipy.special import gammaln
from pfjax import BaseModel

from smc_renewal.config import ModelConfig
from smc_renewal.observation import negbin_loglik
from smc_renewal.transition import _floor, _pad_pmf


class ParticleStateDiscrete(NamedTuple):
    """Model D latent state: level + velocity for log Rt AND log F."""

    log_Rt: Array
    v_R: Array
    log_F: Array
    v_F: Array
    log_I0: Array
    I_buf: Array   # integer-valued (Poisson draws); float dtype for JAX-friendliness


class ParticleParamsDiscrete(NamedTuple):
    """Liu-West parameters for Model D — 4-D cloud."""

    log_sigma_vR: Array
    log_sigma_vF: Array
    log_mu: Array     # immigration rate (positive via exp)
    log_phi: Array    # NegBin overdispersion


def step_discrete(
    state: ParticleStateDiscrete,
    params: ParticleParamsDiscrete,
    eta_R: Array,
    eta_F: Array,
    k_poisson: Array,
    cfg: ModelConfig,
) -> ParticleStateDiscrete:
    """One step of Model D.

    Order matches the local-linear-trend form (level moves by OLD velocity,
    then velocity does a RW).  Renewal+feedback math identical to Model C
    except infections are then Poisson-sampled around the deterministic rate.
    """
    L = cfg.buffer_len
    g_pad = _pad_pmf(cfg.generation_interval, L)

    sigma_vR = _floor(params.log_sigma_vR, cfg.sigma_floor)
    sigma_vF = _floor(params.log_sigma_vF, cfg.sigma_floor)
    mu = jnp.exp(params.log_mu)

    log_Rt_new = state.log_Rt + state.v_R
    log_F_new = state.log_F + state.v_F
    v_R_new = state.v_R + sigma_vR * eta_R
    v_F_new = state.v_F + sigma_vF * eta_F

    # Renewal with F-feedback (matching Model C).
    g_conv_I = jnp.dot(g_pad, state.I_buf)
    F_new = jnp.exp(log_F_new)
    log_Rt_clipped = jnp.clip(log_Rt_new, -20.0, 20.0)
    exponent = jnp.clip(log_Rt_clipped - F_new * g_conv_I, -20.0, 20.0)
    Rt_eff = jnp.exp(exponent)

    lambda_t = jnp.clip(mu + Rt_eff * g_conv_I, 1e-12, 1e12)

    # Poisson sample.  Cast to float for buffer consistency with the other models.
    I_new = jr.poisson(k_poisson, lambda_t).astype(jnp.float64)

    I_buf_new = jnp.concatenate([I_new[jnp.newaxis], state.I_buf[:-1]])

    return ParticleStateDiscrete(
        log_Rt=log_Rt_new,
        v_R=v_R_new,
        log_F=log_F_new,
        v_F=v_F_new,
        log_I0=state.log_I0,
        I_buf=I_buf_new,
    )


def expected_observation_discrete(
    state: ParticleStateDiscrete, cfg: ModelConfig
) -> Array:
    """μ_y(t) = Σ d_s · I[t-s] — same as in Models A/B/C."""
    d_pad = _pad_pmf(cfg.delay_pmf, cfg.buffer_len)
    return jnp.dot(d_pad, state.I_buf)


class RenewalModelDiscrete(BaseModel):
    """pfjax.BaseModel wrapper for Model D."""

    def __init__(self, cfg: ModelConfig):
        super().__init__(bootstrap=True)
        self.cfg = cfg

    def prior_sample(
        self, key: Array, theta: ParticleParamsDiscrete
    ) -> ParticleStateDiscrete:
        cfg = self.cfg
        keys = jr.split(key, 5)
        log_Rt = cfg.init_log_Rt_mean + cfg.init_log_Rt_sd * jr.normal(keys[0])
        v_R = cfg.init_v_R_mean + cfg.init_v_R_sd * jr.normal(keys[1])
        log_F = cfg.init_log_F_mean + cfg.init_log_F_sd * jr.normal(keys[2])
        v_F = cfg.init_v_F_mean + cfg.init_v_F_sd * jr.normal(keys[3])
        log_I0 = cfg.init_log_I0_mean + cfg.init_log_I0_sd * jr.normal(keys[4])
        I_buf = jnp.full((cfg.buffer_len,), jnp.exp(log_I0))
        return ParticleStateDiscrete(
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
        self, key: Array, x_prev: ParticleStateDiscrete, theta: ParticleParamsDiscrete
    ) -> ParticleStateDiscrete:
        k_eta, k_pois = jr.split(key, 2)
        eta = jr.normal(k_eta, (2,))
        return step_discrete(x_prev, theta, eta[0], eta[1], k_pois, self.cfg)

    def state_lpdf(self, x_curr, x_prev, theta) -> Array:
        """log p(x_curr | x_prev, theta): Gaussian on the two velocity
        increments + Poisson PMF on the new infection count.  Level updates
        are deterministic and contribute Dirac (= 0).
        """
        cfg = self.cfg
        sigma_vR = jnp.maximum(jnp.exp(theta.log_sigma_vR), cfg.sigma_floor)
        sigma_vF = jnp.maximum(jnp.exp(theta.log_sigma_vF), cfg.sigma_floor)
        mu = jnp.exp(theta.log_mu)
        L = cfg.buffer_len
        g_pad = _pad_pmf(cfg.generation_interval, L)

        def _logN(x, m, sd):
            return -0.5 * jnp.log(2.0 * jnp.pi * sd * sd) - 0.5 * ((x - m) / sd) ** 2

        log_p_v = (
            _logN(x_curr.v_R, x_prev.v_R, sigma_vR)
            + _logN(x_curr.v_F, x_prev.v_F, sigma_vF)
        )

        # Poisson PMF on I_new = x_curr.I_buf[0], given λ_t from x_prev.
        log_Rt_curr = x_prev.log_Rt + x_prev.v_R
        log_F_curr = x_prev.log_F + x_prev.v_F
        R_t = jnp.exp(jnp.clip(log_Rt_curr, -20.0, 20.0))
        F_t = jnp.exp(log_F_curr)
        g_conv = jnp.dot(g_pad, x_prev.I_buf)
        Rt_eff = jnp.exp(jnp.clip(jnp.clip(log_Rt_curr, -20.0, 20.0) - F_t * g_conv, -20.0, 20.0))
        lambda_t = jnp.clip(mu + Rt_eff * g_conv, 1e-12, 1e12)
        I_new = x_curr.I_buf[0]
        log_p_I = I_new * jnp.log(lambda_t) - lambda_t - gammaln(I_new + 1.0)

        return log_p_v + log_p_I

    def meas_lpdf(
        self, y_curr: Array, x_curr: ParticleStateDiscrete, theta: ParticleParamsDiscrete
    ) -> Array:
        mu_y = expected_observation_discrete(x_curr, self.cfg)
        phi = jnp.exp(theta.log_phi)
        return negbin_loglik(jnp.asarray(y_curr), mu_y, phi).reshape(())
