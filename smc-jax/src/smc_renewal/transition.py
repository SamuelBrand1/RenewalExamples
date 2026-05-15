"""Pure-JAX one-step transition for the nested-RW renewal model.

The state at time ``t`` carries four random-walk variables driven by two
nested random walks:

    log Rt[t]       = log Rt[t-1]      + σ_R[t-1] · eps_R[t]
    log σ_R[t]      = log σ_R[t-1]     + τ_R     · η_R[t]
    log F[t]        = log F[t-1]       + σ_F[t-1] · eps_F[t]
    log σ_F[t]      = log σ_F[t-1]     + τ_F     · η_F[t]

Renewal + feedback at time ``t``:

    I(t) = Rt(t) · exp(-F(t) · sum_τ g(τ) I(t-τ)) · sum_τ g(τ) I(t-τ)

where ``F = exp(log_F) > 0`` is strictly positive (so the feedback always
damps), and the feedback convolution uses the generation interval ``g``
(common modelling choice — no separate feedback PMF).

Layout: ``I_buf[0]`` is the most-recent infection (= I(t-1)); ``I_buf[k]`` is
``I(t-1-k)``.  ``g`` is a forward-time PMF over lags 1..L, zero-padded to
``buffer_len`` so a single dot product covers the buffer.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array

from smc_renewal.config import ModelConfig
from smc_renewal.state import ParticleParams, ParticleState


class TransitionNoise(NamedTuple):
    """Standard-normal innovations for one step."""

    eps_R: Array  # drives log Rt
    eta_R: Array  # drives log sigma_R
    eps_F: Array  # drives log_F
    eta_F: Array  # drives log sigma_F


def _pad_pmf(pmf: Array, length: int) -> Array:
    """Zero-pad a forward-time PMF to ``length`` (lags 1..length)."""
    return jnp.concatenate([pmf, jnp.zeros((length - pmf.shape[0],), dtype=pmf.dtype)])


def _floor(log_sigma: Array, floor: float) -> Array:
    """Clamp ``exp(log_sigma)`` to ``floor`` from below for numerical conditioning."""
    return jnp.maximum(jnp.exp(log_sigma), floor)


def step(
    state: ParticleState,
    params: ParticleParams,
    noise: TransitionNoise,
    cfg: ModelConfig,
) -> ParticleState:
    """One step of the generative model.

    Order of operations:
        1. Use the OLD σ_R / σ_F to drive log_Rt and log_F.
        2. Update log σ_R / log σ_F via the outer random walks driven by τ.
        3. Compute feedback term ``= sum(g · I_buf)`` (same as renewal term).
        4. Compute ``I(t) = exp(log_Rt_new - exp(log_F_new) · feedback) · feedback``.
        5. Shift ``I_buf`` to put ``I(t)`` at index 0.
    """
    L = cfg.buffer_len
    g_pad = _pad_pmf(cfg.generation_interval, L)

    sigma_R_old = _floor(state.log_sigma_R, cfg.sigma_floor)
    sigma_F_old = _floor(state.log_sigma_F, cfg.sigma_floor)
    tau_R = _floor(params.log_tau_R, cfg.sigma_floor)
    tau_F = _floor(params.log_tau_F, cfg.sigma_floor)

    log_Rt_new = state.log_Rt + sigma_R_old * noise.eps_R
    log_F_new = state.log_F + sigma_F_old * noise.eps_F

    log_sigma_R_new = state.log_sigma_R + tau_R * noise.eta_R
    log_sigma_F_new = state.log_sigma_F + tau_F * noise.eta_F

    # Feedback / renewal share the same convolution under the simplification
    # "use the generation interval as the feedback kernel".
    g_conv_I = jnp.dot(g_pad, state.I_buf)
    F_new = jnp.exp(log_F_new)

    # Soft clamps prevent float overflow on pathological particles; invisible
    # in normal operation (Rt typically in [0.5, 2], F·conv usually in [0, 1]).
    log_Rt_clipped = jnp.clip(log_Rt_new, -20.0, 20.0)
    exponent = jnp.clip(log_Rt_clipped - F_new * g_conv_I, -20.0, 20.0)
    Rt_eff = jnp.exp(exponent)
    I_new = jnp.minimum(Rt_eff * g_conv_I, 1e15)

    I_buf_new = jnp.concatenate([I_new[jnp.newaxis], state.I_buf[:-1]])

    return ParticleState(
        log_Rt=log_Rt_new,
        log_sigma_R=log_sigma_R_new,
        log_F=log_F_new,
        log_sigma_F=log_sigma_F_new,
        log_I0=state.log_I0,
        I_buf=I_buf_new,
    )


def expected_observation(state: ParticleState, cfg: ModelConfig) -> Array:
    """Expected observed cases at time ``t`` given the post-step ``I_buf``.

    The delay pmf ``d`` has support 0..Td-1; same-day reports come from
    ``I_buf[0]`` (= ``I(t)`` post-step).
    """
    d_pad = _pad_pmf(cfg.delay_pmf, cfg.buffer_len)
    return jnp.dot(d_pad, state.I_buf)
