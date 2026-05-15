"""Library-backed Extended Kalman Filter via ``cuthbert.gaussian.moments``.

Drop-in alternative to ``smc2.ukf.marginal_log_likelihood``.  Same signature,
different implementation:

- The hand-rolled UKF in ``smc2.ukf`` uses sigma-point linearization.
- This one uses Taylor / Jacobian linearization via ``cuthbert``'s ``moments``
  filter (auto-diff under the hood, so still all-JAX, still jit/vmap-friendly).

For our model the nonlinearity is mild (one ``exp(gamma * feedback)`` per
step) so the two should agree closely on the marginal log-likelihood; an EKF
also unlocks ``cuthbert.gaussian.moments.build_smoother`` for free Kalman
smoothing on the same model, but that's not used here.
"""

from __future__ import annotations

import jax.numpy as jnp
from cuthbert.filtering import filter as cuthbert_filter
from cuthbert.gaussian.moments import build_filter
from jax import Array

from smc_renewal.config import ModelConfig
from smc_renewal.observation import gaussian_obs_moments
from smc_renewal.smc2.ukf import (
    _f_det,
    _process_noise_cov,
    initial_moments,
)
from smc_renewal.state import ParticleParams, state_dim, unpack


def _build_filter(cfg: ModelConfig, theta: ParticleParams):
    m0, P0 = initial_moments(cfg)
    P0 = P0 + 1e-9 * jnp.eye(P0.shape[0])
    chol_P0 = jnp.linalg.cholesky(P0)

    def get_init_params(model_inputs):
        return m0, chol_P0

    def get_dynamics_params(state, model_inputs):
        # Linearize around the previous filtered mean.
        m_prev = state.mean

        def mean_and_chol_cov(x):
            mean = _f_det(x, theta, cfg)
            Q = _process_noise_cov(x, theta, cfg) + 1e-9 * jnp.eye(state_dim(cfg.buffer_len))
            chol = jnp.linalg.cholesky(Q)
            return mean, chol

        return mean_and_chol_cov, m_prev

    def get_observation_params(state, model_inputs):
        # state.mean here is the predicted state mean.
        m_pred = state.mean
        y_t = jnp.atleast_1d(model_inputs)  # (1,)

        def mean_and_chol_cov(x):
            # Observation mean = mu_y(x); variance = NegBin moment-matched.
            from smc_renewal.transition import expected_observation

            packed = unpack(x, cfg.buffer_len)
            mu_y = expected_observation(packed, cfg).reshape(())
            phi = jnp.exp(theta.log_phi)
            _m, var = gaussian_obs_moments(jnp.maximum(mu_y, 1e-12), phi)
            chol = jnp.sqrt(var).reshape(1, 1)
            return mu_y.reshape(1), chol

        return mean_and_chol_cov, m_pred, y_t

    return build_filter(
        get_init_params=get_init_params,
        get_dynamics_params=get_dynamics_params,
        get_observation_params=get_observation_params,
        associative=False,
    )


def _run_filter_states(cfg: ModelConfig, y: Array, params: ParticleParams):
    """Run the cuthbert EKF and return the full filter trajectory."""
    filt = _build_filter(cfg, params)
    model_inputs = jnp.concatenate([jnp.zeros(1, dtype=y.dtype), jnp.asarray(y)])
    return cuthbert_filter(filt, model_inputs)


def marginal_log_likelihood(
    cfg: ModelConfig, y: Array, params: ParticleParams
) -> Array:
    """EKF marginal log-likelihood ``log p(y_{1:T} | params)`` via cuthbert.

    Drop-in replacement for ``smc2.ukf.marginal_log_likelihood``.
    """
    return _run_filter_states(cfg, y, params).log_normalizing_constant[-1]


def filter_trajectory(
    cfg: ModelConfig, y: Array, params: ParticleParams
) -> tuple[Array, Array]:
    """Return (means, chol_covs) for the EKF over ``y``.

    Shapes (T+1 = len(y) + 1 because cuthbert prepends the initial state):
        means:     (T+1, D)
        chol_covs: (T+1, D, D)

    Caller typically discards index 0 (the prior) to align with ``y[0]`` at index 1.
    """
    states = _run_filter_states(cfg, y, params)
    return states.mean, states.chol_cov
