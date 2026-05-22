"""A ``pfjax.BaseModel`` for the nested-RW renewal model.

We expose the three primitives a bootstrap PF needs (``prior_sample``,
``state_sample``, ``meas_lpdf``) plus the optional ``state_lpdf`` /
``prior_lpdf`` so smoothers and Rao-Blackwellised filters can also use this
model.  Everything routes through the existing ``transition.step`` and
``observation.negbin_loglik`` so this is a thin pfjax adapter, not a second
implementation.

The pfjax convention: ``theta`` is a single object shared across particles.
For the Liu-West variant we run a custom scan that pre-jitters a per-particle
theta and calls into these model primitives directly (see ``pf/runner.py``).
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr
from jax import Array
from pfjax import BaseModel

from smc_renewal.config import ModelConfig
from smc_renewal.observation import negbin_loglik
from smc_renewal.state import ParticleParams, ParticleState
from smc_renewal.transition import (
    TransitionNoise,
    expected_observation,
    step,
)


class RenewalModel(BaseModel):
    """``pfjax.BaseModel`` wrapping our pure-JAX renewal transition.

    Parameters
    ----------
    cfg
        ``ModelConfig`` carrying the discretized PMFs, sigma floor, priors.

    Notes
    -----
    The state ``x`` is a ``ParticleState`` (``NamedTuple`` — JAX tree-friendly).
    The parameter ``theta`` is a ``ParticleParams``.  Both shapes are scalar
    *within* a particle; pfjax will vmap the calls across particles.
    """

    def __init__(self, cfg: ModelConfig):
        super().__init__(bootstrap=True)
        self.cfg = cfg

    # ---- prior on the initial latent state ----

    def prior_sample(self, key: Array, theta: ParticleParams) -> ParticleState:
        cfg = self.cfg
        keys = jr.split(key, 5)
        log_Rt = cfg.init_log_Rt_mean + cfg.init_log_Rt_sd * jr.normal(keys[0])
        log_sigma_R = cfg.init_log_sigma_R_mean + cfg.init_log_sigma_R_sd * jr.normal(keys[1])
        log_F = cfg.init_log_F_mean + cfg.init_log_F_sd * jr.normal(keys[2])
        log_sigma_F = cfg.init_log_sigma_F_mean + cfg.init_log_sigma_F_sd * jr.normal(keys[3])
        log_I0 = cfg.init_log_I0_mean + cfg.init_log_I0_sd * jr.normal(keys[4])
        I_buf = jnp.full((cfg.buffer_len,), jnp.exp(log_I0))
        return ParticleState(
            log_Rt=log_Rt,
            log_sigma_R=log_sigma_R,
            log_F=log_F,
            log_sigma_F=log_sigma_F,
            log_I0=log_I0,
            I_buf=I_buf,
        )

    def prior_lpdf(self, x_curr: ParticleState, theta: ParticleParams) -> Array:
        cfg = self.cfg
        return (
            -0.5 * ((x_curr.log_Rt - cfg.init_log_Rt_mean) / cfg.init_log_Rt_sd) ** 2
            - 0.5 * ((x_curr.log_sigma_R - cfg.init_log_sigma_R_mean) / cfg.init_log_sigma_R_sd) ** 2
            - 0.5 * ((x_curr.log_F - cfg.init_log_F_mean) / cfg.init_log_F_sd) ** 2
            - 0.5 * ((x_curr.log_sigma_F - cfg.init_log_sigma_F_mean) / cfg.init_log_sigma_F_sd) ** 2
            - 0.5 * ((x_curr.log_I0 - cfg.init_log_I0_mean) / cfg.init_log_I0_sd) ** 2
        )

    # ---- transition ----

    def state_sample(
        self, key: Array, x_prev: ParticleState, theta: ParticleParams
    ) -> ParticleState:
        eps = jr.normal(key, (4,))
        noise = TransitionNoise(
            eps_R=eps[0], eta_R=eps[1], eps_F=eps[2], eta_F=eps[3]
        )
        return step(x_prev, theta, noise, self.cfg)

    def state_lpdf(
        self, x_curr: ParticleState, x_prev: ParticleState, theta: ParticleParams
    ) -> Array:
        """Log p(x_curr | x_prev, theta).

        The transition is degenerate (the buffer update is deterministic given
        the four noise dimensions).  Conditional on x_prev and theta, the joint
        density on the 4 noisy state coordinates is a product of Gaussians:

            log_Rt[t]      ~ N(log_Rt[t-1],     sigma_R[t-1])
            log_sigma_R[t] ~ N(log_sigma_R[t-1], tau_R)
            log_F[t]       ~ N(log_F[t-1],      sigma_F[t-1])
            log_sigma_F[t] ~ N(log_sigma_F[t-1], tau_F)

        The other state entries (log_I0, I_buf) are deterministic given these.
        We return the log-density on the 4 free coordinates; the deterministic
        coordinates contribute 0 (they are dirac).  This is the form needed by
        FFBS-style smoothers; the trajectory smoother in pfjax doesn't use it.
        """
        cfg = self.cfg
        sigma_R = jnp.maximum(jnp.exp(x_prev.log_sigma_R), cfg.sigma_floor)
        sigma_F = jnp.maximum(jnp.exp(x_prev.log_sigma_F), cfg.sigma_floor)
        tau_R = jnp.maximum(jnp.exp(theta.log_tau_R), cfg.sigma_floor)
        tau_F = jnp.maximum(jnp.exp(theta.log_tau_F), cfg.sigma_floor)

        def _logN(x, mu, sd):
            return -0.5 * jnp.log(2.0 * jnp.pi * sd * sd) - 0.5 * ((x - mu) / sd) ** 2

        return (
            _logN(x_curr.log_Rt, x_prev.log_Rt, sigma_R)
            + _logN(x_curr.log_sigma_R, x_prev.log_sigma_R, tau_R)
            + _logN(x_curr.log_F, x_prev.log_F, sigma_F)
            + _logN(x_curr.log_sigma_F, x_prev.log_sigma_F, tau_F)
        )

    # ---- observation likelihood ----

    def meas_lpdf(
        self, y_curr: Array, x_curr: ParticleState, theta: ParticleParams
    ) -> Array:
        mu_y = expected_observation(x_curr, self.cfg)
        phi = jnp.exp(theta.log_phi)
        return negbin_loglik(jnp.asarray(y_curr), mu_y, phi).reshape(())
