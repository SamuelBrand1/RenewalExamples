"""Particle state and flat-vector pack/unpack helpers.

The state at time t per particle:
  - log_Rt:        scalar
  - log_sigma_R:   scalar
  - log_F:         scalar  (log of the strictly-positive feedback strength F)
  - log_sigma_F:   scalar
  - log_I0:        scalar  (kept around as a fixed per-particle nuisance)
  - I_buf:         shape (L,)  most-recent infections, newest at index 0

The top-level parameters carried by each particle (Liu-West / SMC^2):
  - log_tau_R:   scalar
  - log_tau_F:   scalar
  - log_phi:     scalar

The renewal+feedback dynamics use F = exp(log_F) > 0 as the feedback magnitude
and apply ``Rt_eff = Rt_raw * exp(-F * conv(I, g))`` — strictly damping.  The
feedback convolution kernel is the generation interval g (no separate
feedback PMF).

The UKF needs a flat real vector. `pack` / `unpack` build that flat layout.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
from jax import Array


class ParticleState(NamedTuple):
    """Per-particle latent state (does NOT include theta = (tau_R, tau_F, log_phi))."""

    log_Rt: Array
    log_sigma_R: Array
    log_F: Array
    log_sigma_F: Array
    log_I0: Array
    I_buf: Array  # shape (L,), newest at index 0


class ParticleParams(NamedTuple):
    """Per-particle top-level parameters (Liu-West / SMC^2 targets)."""

    log_tau_R: Array
    log_tau_F: Array
    log_phi: Array


# -- flat pack/unpack for the UKF -----------------------------------------------------
# Layout (size = 5 + L):
#   [0] log_Rt
#   [1] log_sigma_R
#   [2] log_F
#   [3] log_sigma_F
#   [4] log_I0
#   [5:5+L] I_buf


def state_dim(buffer_len: int) -> int:
    return 5 + buffer_len


def pack(state: ParticleState) -> Array:
    return jnp.concatenate(
        [
            jnp.atleast_1d(state.log_Rt),
            jnp.atleast_1d(state.log_sigma_R),
            jnp.atleast_1d(state.log_F),
            jnp.atleast_1d(state.log_sigma_F),
            jnp.atleast_1d(state.log_I0),
            jnp.asarray(state.I_buf).reshape(-1),
        ],
        axis=0,
    )


def unpack(flat: Array, buffer_len: int) -> ParticleState:
    return ParticleState(
        log_Rt=flat[0],
        log_sigma_R=flat[1],
        log_F=flat[2],
        log_sigma_F=flat[3],
        log_I0=flat[4],
        I_buf=flat[5 : 5 + buffer_len],
    )
