"""Seedable synthetic-data generator.

Drives ``transition.step`` forward from a sampled initial state under user-chosen
parameters (or sampled from the prior).  Emits a NamedTuple with the full latent
trajectory and the observed case time series, so tests/diagnostics can compare
PF/SMC^2 posteriors against ground truth.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro.distributions as dist
from jax import Array

from smc_renewal.config import ModelConfig
from smc_renewal.observation import gaussian_obs_moments  # noqa: F401  (used by callers)
from smc_renewal.state import ParticleParams, ParticleState
from smc_renewal.transition import (
    TransitionNoise,
    expected_observation,
    step,
)


class SyntheticDataset(NamedTuple):
    """Ground truth + observations from one synthetic run."""

    params: ParticleParams
    log_Rt: Array          # shape (T,)
    log_sigma_R: Array     # shape (T,)
    log_F: Array           # shape (T,)  -- log feedback strength (F = exp(log_F) > 0)
    log_sigma_F: Array     # shape (T,)
    infections: Array      # shape (T,)
    mu_y: Array            # shape (T,)
    y: Array               # shape (T,)  -- observed cases (integers cast to float)
    initial_state: ParticleState


def sample_initial_state(key: Array, cfg: ModelConfig) -> ParticleState:
    """Draw an initial particle state from the prior in ``cfg``.

    The infection buffer is initialised to a constant ``exp(log_I0)``, which is
    the natural quasi-stationary state under ``Rt = 1`` and weak feedback.
    """
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


def sample_params_from_prior(key: Array, cfg: ModelConfig) -> ParticleParams:
    """Draw ``(log_tau_R, log_tau_F, log_phi)`` from the priors in ``cfg``."""
    keys = jr.split(key, 3)
    return ParticleParams(
        log_tau_R=cfg.prior_log_tau_R_mean + cfg.prior_log_tau_R_sd * jr.normal(keys[0]),
        log_tau_F=cfg.prior_log_tau_F_mean + cfg.prior_log_tau_F_sd * jr.normal(keys[1]),
        log_phi=cfg.prior_log_phi_mean + cfg.prior_log_phi_sd * jr.normal(keys[2]),
    )


def simulate(
    key: Array,
    cfg: ModelConfig,
    T: int,
    params: ParticleParams | None = None,
    initial_state: ParticleState | None = None,
) -> SyntheticDataset:
    """Forward-simulate ``T`` days of the model.

    Returns the full latent trajectory, expected observation ``mu_y``, and a
    NegBin sample ``y`` per day.
    """
    k_params, k_init, k_noise, k_obs = jr.split(key, 4)
    if params is None:
        params = sample_params_from_prior(k_params, cfg)
    if initial_state is None:
        initial_state = sample_initial_state(k_init, cfg)

    all_noise = jr.normal(k_noise, (T, 4))
    noises = TransitionNoise(
        eps_R=all_noise[:, 0],
        eta_R=all_noise[:, 1],
        eps_F=all_noise[:, 2],
        eta_F=all_noise[:, 3],
    )

    def body(state, n):
        new_state = step(
            state,
            params,
            TransitionNoise(
                eps_R=n.eps_R, eta_R=n.eta_R, eps_F=n.eps_F, eta_F=n.eta_F
            ),
            cfg,
        )
        mu_y = expected_observation(new_state, cfg)
        return new_state, (new_state, mu_y)

    _, (traj, mu_y) = jax.lax.scan(body, initial_state, noises)

    phi = jnp.exp(params.log_phi)
    y = dist.NegativeBinomial2(mean=mu_y, concentration=phi).sample(k_obs)

    return SyntheticDataset(
        params=params,
        log_Rt=traj.log_Rt,
        log_sigma_R=traj.log_sigma_R,
        log_F=traj.log_F,
        log_sigma_F=traj.log_sigma_F,
        infections=traj.I_buf[:, 0],
        mu_y=mu_y,
        y=y.astype(jnp.float32),
        initial_state=initial_state,
    )
