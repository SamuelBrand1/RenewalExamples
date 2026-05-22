"""Verify the per-step JAX transition reproduces pyrenew's renewal-with-feedback.

Our parameterization (``Rt_eff = Rt_raw · exp(-F · conv(I, g))`` with
``F = exp(log_F) > 0``, feedback PMF = generation interval) maps to pyrenew's
(``Rt_eff = Rt_raw · exp(γ · conv(I, f))``) by setting ``γ_pyrenew = -F`` and
``f = g``.  Feeding both formulations a matching deterministic trajectory
should produce identical infections.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr

from pyrenew.latent.infection_functions import (
    compute_infections_from_rt_with_feedback,
)

from smc_renewal.config import default_config
from smc_renewal.state import ParticleParams, ParticleState
from smc_renewal.transition import TransitionNoise, step


def _initial_state(cfg, log_Rt0, log_F0, I0_vec):
    """Build an initial state whose I_buf matches a chronological I0 history."""
    L = cfg.buffer_len
    Tg = cfg.Tg
    newest_first = I0_vec[::-1]
    if L > Tg:
        I_buf = jnp.concatenate([newest_first, jnp.zeros((L - Tg,))])
    else:
        I_buf = newest_first[:L]
    return ParticleState(
        log_Rt=jnp.asarray(log_Rt0),
        log_sigma_R=jnp.asarray(0.0),
        log_F=jnp.asarray(log_F0),
        log_sigma_F=jnp.asarray(0.0),
        log_I0=jnp.asarray(0.0),
        I_buf=I_buf,
    )


def test_step_matches_pyrenew_renewal_with_feedback():
    cfg = default_config()
    Tg = cfg.Tg

    T = 30
    key = jr.key(7)
    k1, k2, k3 = jr.split(key, 3)
    log_Rt_seq = 0.3 * jnp.sin(0.1 * jnp.arange(T)) + 0.05 * jr.normal(k1, (T,))
    # Positive F values so the feedback damps; vary mildly across time for richness.
    log_F_seq = -4.0 + 0.3 * jnp.sin(0.05 * jnp.arange(T)) + 0.05 * jr.normal(k2, (T,))
    F_seq = jnp.exp(log_F_seq)

    I0_vec = 50.0 * jnp.exp(0.05 * jnp.arange(Tg)) + 0.5 * jr.normal(k3, (Tg,))

    # --- pyrenew reference: γ_pyrenew = -F, feedback PMF = generation interval ---
    rev_gi = cfg.generation_interval[::-1]
    pyrenew_I, _Rt_adj = compute_infections_from_rt_with_feedback(
        I0=I0_vec,
        Rt_raw=jnp.exp(log_Rt_seq),
        infection_feedback_strength=-F_seq,
        reversed_generation_interval_pmf=rev_gi,
        reversed_infection_feedback_pmf=rev_gi,  # same PMF
    )

    # --- our per-step ---
    state0 = _initial_state(cfg, log_Rt_seq[0], log_F_seq[0], I0_vec)
    params = ParticleParams(
        log_tau_R=jnp.asarray(0.0), log_tau_F=jnp.asarray(0.0), log_phi=jnp.asarray(0.0)
    )

    def body(state, t):
        # Force log_Rt and log_F to the reference values at each step.
        forced = state._replace(log_Rt=log_Rt_seq[t], log_F=log_F_seq[t])
        noise = TransitionNoise(
            eps_R=jnp.asarray(0.0),
            eta_R=jnp.asarray(0.0),
            eps_F=jnp.asarray(0.0),
            eta_F=jnp.asarray(0.0),
        )
        new_state = step(forced, params, noise, cfg)
        return new_state, new_state.I_buf[0]

    _, my_I = jax.lax.scan(body, state0, jnp.arange(T))

    assert pyrenew_I.shape == my_I.shape
    assert jnp.allclose(pyrenew_I, my_I, atol=1e-5, rtol=1e-5), (
        f"max abs diff: {jnp.max(jnp.abs(pyrenew_I - my_I))}"
    )
