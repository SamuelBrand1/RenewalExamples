"""Unscented Kalman filter for marginal-likelihood evaluation of the renewal SSM.

The filter operates on the flattened state vector ``x = pack(ParticleState)`` of
dimension ``D = 5 + L``.  The deterministic transition is provided as
``f: (x, theta) -> x_new``; process noise is approximated as additive with a
state-dependent diagonal covariance ``Q(x, theta) = diag(sigma_R^2, tau_R^2,
sigma_F^2, tau_F^2, 0, ..., 0)`` (the I_buf entries are deterministic given
the noise-injected upstream entries).

The observation model is the moment-matched Gaussian
``y_t ~ N(mu_y, mu_y + mu_y^2 / phi)``.

Returns the marginal log-likelihood ``log p(y_{1:T} | theta)``.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import Array

from smc_renewal.config import ModelConfig
from smc_renewal.observation import gaussian_obs_moments
from smc_renewal.state import (
    ParticleParams,
    ParticleState,
    pack,
    state_dim,
    unpack,
)
from smc_renewal.transition import (
    TransitionNoise,
    expected_observation,
    step,
)


class UKFParams(NamedTuple):
    """Sigma-point weights and scaling for the UKF."""

    alpha: float = 1e-3
    beta: float = 2.0
    kappa: float = 0.0

    def lambda_(self, n: int) -> float:
        return self.alpha**2 * (n + self.kappa) - n


def _sigma_points(m: Array, P: Array, ukf: UKFParams) -> tuple[Array, Array, Array]:
    """Generate ``2n+1`` sigma points and their mean/cov weights.

    Returns ``(chi, Wm, Wc)`` where ``chi`` has shape ``(2n+1, n)``.
    """
    n = m.shape[0]
    lam = ukf.lambda_(n)
    # P may need a tiny ridge for cholesky stability.
    P_reg = P + 1e-9 * jnp.eye(n)
    L = jnp.linalg.cholesky((n + lam) * P_reg)

    chi0 = m
    chi_plus = m[None, :] + L.T  # rows are sigma offsets
    chi_minus = m[None, :] - L.T
    chi = jnp.concatenate([chi0[None, :], chi_plus, chi_minus], axis=0)

    Wm0 = lam / (n + lam)
    Wc0 = Wm0 + (1.0 - ukf.alpha**2 + ukf.beta)
    Wi = 1.0 / (2.0 * (n + lam))
    Wm = jnp.concatenate([jnp.array([Wm0]), jnp.full((2 * n,), Wi)])
    Wc = jnp.concatenate([jnp.array([Wc0]), jnp.full((2 * n,), Wi)])
    return chi, Wm, Wc


def _process_noise_cov(m: Array, params: ParticleParams, cfg: ModelConfig) -> Array:
    """State-dependent process-noise covariance at linearization point ``m``.

    Only four entries are noisy: log_Rt (driven by σ_R), log_σ_R (by τ_R),
    log_F (by σ_F), log_σ_F (by τ_F).  log_I0 and the buffer entries have
    no exogenous noise (the buffer is deterministic given the noisy upstreams).
    """
    state = unpack(m, cfg.buffer_len)
    sigma_R = jnp.maximum(jnp.exp(state.log_sigma_R), cfg.sigma_floor)
    sigma_F = jnp.maximum(jnp.exp(state.log_sigma_F), cfg.sigma_floor)
    tau_R = jnp.maximum(jnp.exp(params.log_tau_R), cfg.sigma_floor)
    tau_F = jnp.maximum(jnp.exp(params.log_tau_F), cfg.sigma_floor)
    D = state_dim(cfg.buffer_len)
    Q = jnp.zeros((D, D))
    Q = Q.at[0, 0].set(sigma_R**2)
    Q = Q.at[1, 1].set(tau_R**2)
    Q = Q.at[2, 2].set(sigma_F**2)
    Q = Q.at[3, 3].set(tau_F**2)
    return Q


def _f_det(m: Array, params: ParticleParams, cfg: ModelConfig) -> Array:
    """Deterministic part of the transition (zero noise)."""
    state = unpack(m, cfg.buffer_len)
    noise = TransitionNoise(
        eps_R=jnp.asarray(0.0),
        eta_R=jnp.asarray(0.0),
        eps_F=jnp.asarray(0.0),
        eta_F=jnp.asarray(0.0),
    )
    new_state = step(state, params, noise, cfg)
    return pack(new_state)


def _h_obs(m: Array, cfg: ModelConfig) -> Array:
    """Observation mean as a scalar function of the flattened state."""
    state = unpack(m, cfg.buffer_len)
    return expected_observation(state, cfg).reshape(())


def _ukf_step(
    m: Array,
    P: Array,
    y_t: Array,
    params: ParticleParams,
    cfg: ModelConfig,
    ukf: UKFParams,
) -> tuple[Array, Array, Array]:
    """One UKF update; returns ``(m_new, P_new, log_lik_increment)``."""
    n = m.shape[0]

    # Predict.
    chi, Wm, Wc = _sigma_points(m, P, ukf)
    chi_pred = jax.vmap(lambda x: _f_det(x, params, cfg))(chi)
    m_pred = jnp.sum(Wm[:, None] * chi_pred, axis=0)
    diff = chi_pred - m_pred[None, :]
    P_pred = (Wc[:, None, None] * diff[:, :, None] * diff[:, None, :]).sum(axis=0)
    Q = _process_noise_cov(m_pred, params, cfg)
    P_pred = P_pred + Q

    # Observation update.
    chi2, Wm2, Wc2 = _sigma_points(m_pred, P_pred, ukf)
    mu_y_pts = jax.vmap(lambda x: _h_obs(x, cfg))(chi2)  # (2n+1,)
    y_hat = jnp.sum(Wm2 * mu_y_pts)
    diff_y = mu_y_pts - y_hat
    # NB-moment-matched observation noise at the predicted mean.
    phi = jnp.exp(params.log_phi)
    _mu, R_var = gaussian_obs_moments(jnp.maximum(y_hat, 1e-12), phi)
    S = jnp.sum(Wc2 * diff_y * diff_y) + R_var
    diff_x = chi2 - m_pred[None, :]
    C = jnp.sum(Wc2[:, None] * diff_x * diff_y[:, None], axis=0)  # (n,)
    K = C / S
    innov = y_t - y_hat
    m_new = m_pred + K * innov
    P_new = P_pred - jnp.outer(K, K) * S
    # log p(y_t) = log N(y_t | y_hat, S)
    log_lik = -0.5 * (jnp.log(2.0 * jnp.pi * S) + innov * innov / S)
    return m_new, P_new, log_lik


def initial_moments(cfg: ModelConfig) -> tuple[Array, Array]:
    """Initial mean and covariance of the latent state, from priors in ``cfg``."""
    D = state_dim(cfg.buffer_len)
    m = jnp.zeros(D)
    m = m.at[0].set(cfg.init_log_Rt_mean)
    m = m.at[1].set(cfg.init_log_sigma_R_mean)
    m = m.at[2].set(cfg.init_log_F_mean)
    m = m.at[3].set(cfg.init_log_sigma_F_mean)
    m = m.at[4].set(cfg.init_log_I0_mean)
    # Buffer entries set to exp(log_I0_mean) (quasi-stationary).
    m = m.at[5:].set(jnp.exp(cfg.init_log_I0_mean))

    diag = jnp.zeros(D)
    diag = diag.at[0].set(cfg.init_log_Rt_sd**2)
    diag = diag.at[1].set(cfg.init_log_sigma_R_sd**2)
    diag = diag.at[2].set(cfg.init_log_F_sd**2)
    diag = diag.at[3].set(cfg.init_log_sigma_F_sd**2)
    diag = diag.at[4].set(cfg.init_log_I0_sd**2)
    # Small variance on the buffer entries.
    diag = diag.at[5:].set(jnp.exp(cfg.init_log_I0_mean) ** 2 * 0.01)
    P = jnp.diag(diag)
    return m, P


def marginal_log_likelihood(
    cfg: ModelConfig,
    y: Array,
    params: ParticleParams,
    ukf: UKFParams | None = None,
) -> Array:
    """Run UKF over ``y`` and return ``log p(y_{1:T} | params)``."""
    if ukf is None:
        ukf = UKFParams()
    m0, P0 = initial_moments(cfg)

    def body(carry, y_t):
        m, P = carry
        m_new, P_new, ll = _ukf_step(m, P, y_t, params, cfg, ukf)
        return (m_new, P_new), ll

    _, lls = jax.lax.scan(body, (m0, P0), y)
    return jnp.sum(lls)
