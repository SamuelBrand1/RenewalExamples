"""Forward-simulate forecasts from a posterior particle cloud.

Each particle is propagated ``h_max`` days into the future with its own fresh
noise.  The output retains particle weights so quantiles/CRPS can use them.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr
import numpyro.distributions as dist
from jax import Array

from smc_renewal.config import ModelConfig
from smc_renewal.state import ParticleParams, ParticleState
from smc_renewal.transition import (
    TransitionNoise,
    expected_observation,
    step,
)


class ForecastSamples(NamedTuple):
    """Posterior forecast samples from a particle cloud.

    Shapes:
      mu_y:        (h_max, N)  expected observations
      y:           (h_max, N)  NegBin samples
      log_weights: (N,)        copy of cloud weights (same for every horizon)
    """

    mu_y: Array
    y: Array
    log_weights: Array


def forecast_from_cloud(
    key: Array,
    cfg: ModelConfig,
    particles: ParticleState,
    params: ParticleParams,
    log_weights: Array,
    h_max: int,
) -> ForecastSamples:
    """Forward-simulate ``h_max`` days from each particle.

    Particle parameters are held fixed within a particle (no further Liu-West
    jitter inside the forecast — that's the right thing for predictive sampling).
    """
    N = log_weights.shape[0]
    k_noise, k_obs = jr.split(key, 2)

    all_noise = jr.normal(k_noise, (h_max, N, 4))

    def body(state, n_t):
        # n_t shape (N, 4)
        noises = TransitionNoise(
            eps_R=n_t[:, 0],
            eta_R=n_t[:, 1],
            eps_F=n_t[:, 2],
            eta_F=n_t[:, 3],
        )
        new_state = jax.vmap(lambda s, p, n: step(s, p, n, cfg))(state, params, noises)
        mu_y = jax.vmap(lambda s: expected_observation(s, cfg))(new_state)
        return new_state, mu_y

    _, mu_y_traj = jax.lax.scan(body, particles, all_noise)
    phi = jnp.exp(params.log_phi)  # (N,)
    y_traj = dist.NegativeBinomial2(
        mean=mu_y_traj, concentration=phi[None, :]
    ).sample(k_obs)
    return ForecastSamples(
        mu_y=mu_y_traj, y=y_traj.astype(jnp.float64), log_weights=log_weights
    )
