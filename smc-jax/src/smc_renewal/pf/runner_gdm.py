"""Liu-West PF for Model E (discrete renewal + GDM observation delay).

Structurally mirrors ``runner_discrete.py`` but with a custom scan loop —
the guided proposal returns ``(new_state, log_w_inc)`` directly, replacing
the bootstrap ``state_sample`` + ``meas_lpdf`` pair that
``_runner_core.run_liu_west_core`` assumes.  Liu-West shrink-jitter,
ESS-triggered multinomial resampling, and Steyn-style fixed-lag
re-permutation are reused unchanged via ``_runner_core``'s helpers.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array
from jax.scipy.special import logsumexp

from smc_renewal.config import ModelConfig
from smc_renewal.pf._runner_core import _ess, _multinomial_indices
from smc_renewal.pf.liu_west import shrink_jitter
from smc_renewal.pf.model_gdm import (
    ParticleParamsGDM,
    ParticleStateGDM,
    prior_sample_gdm,
    propose_step,
    sample_initial_gdm_params,
)


class LiuWestGDMResult(NamedTuple):
    particles_history: ParticleStateGDM
    params_history: ParticleParamsGDM
    log_weights_history: Array
    ancestors: Array
    ess_history: Array
    log_lik: Array


def run_liu_west_gdm(
    key: Array,
    cfg: ModelConfig,
    y: Array,
    n_particles: int = 4000,
    h: float | None = None,
    ess_threshold: float | None = None,
    fixed_lag_L: int = 1,
) -> LiuWestGDMResult:
    """Liu-West PF for Model E.

    ``y`` is the observed daily case series (shape (T,) int/float).  Returns
    a ``LiuWestGDMResult`` with the per-step particle history (filter cloud),
    log-weights, ancestor indices, ESS, and cumulative marginal log-lik.
    """
    if h is None:
        h = cfg.liu_west_h
    if ess_threshold is None:
        ess_threshold = n_particles / 2.0

    T = y.shape[0]
    L_buf = max(int(fixed_lag_L), 1)
    N = int(n_particles)

    k_init_state, k_init_params, k_run = jr.split(key, 3)

    params0 = sample_initial_gdm_params(k_init_params, cfg, N)
    state_keys = jr.split(k_init_state, N)
    states0 = jax.vmap(lambda kk: prior_sample_gdm(kk, cfg))(state_keys)

    # --- Initial step: weight by proposal log_w_inc against y[0] ---
    propose_keys0 = jr.split(jr.fold_in(k_run, 0), N)
    new_states0, init_log_w = jax.vmap(
        lambda kk, s, p: propose_step(kk, s, p, jnp.asarray(y[0]), cfg)
    )(propose_keys0, states0, params0)
    lw0 = init_log_w - logsumexp(init_log_w)

    step_keys = jr.split(k_run, T - 1)

    state_buf0 = jax.tree.map(
        lambda x: jnp.broadcast_to(x[None], (L_buf,) + x.shape), new_states0
    )
    params_buf0 = jax.tree.map(
        lambda x: jnp.broadcast_to(x[None], (L_buf,) + x.shape), params0
    )

    def _roll(buf, new_entry):
        return jax.tree.map(
            lambda b, n: jnp.concatenate([b[1:], n[None]]), buf, new_entry
        )

    def _permute_buf(buf, idx):
        return jax.tree.map(lambda b: b[:, idx], buf)

    def body(carry, inputs):
        state_buf, params_buf, logw = carry
        y_t, k = inputs
        k_lw, k_step, k_res = jr.split(k, 3)

        states = jax.tree.map(lambda x: x[-1], state_buf)
        params = jax.tree.map(lambda x: x[-1], params_buf)

        params_j = shrink_jitter(params, logw, h, k_lw)

        propose_keys = jr.split(k_step, N)
        new_states, log_w_inc = jax.vmap(
            lambda kk, s, p: propose_step(kk, s, p, y_t, cfg)
        )(propose_keys, states, params_j)

        # log_w_inc may contain -inf (zero-weight escapes); logsumexp is
        # robust to -inf entries.
        unnorm = logw + log_w_inc
        # Guard against all-(-inf) collapse: if logsumexp(unnorm) = -inf,
        # we replace it with a uniform distribution so the scan can continue.
        log_norm = logsumexp(unnorm)
        all_dead = ~jnp.isfinite(log_norm)
        normed = jnp.where(
            all_dead,
            jnp.full_like(unnorm, -jnp.log(N)),
            unnorm - log_norm,
        )
        log_marg_inc = jnp.where(all_dead, -jnp.inf, log_norm - logsumexp(logw))
        ess = _ess(normed)

        rolled_state_buf = _roll(state_buf, new_states)
        rolled_params_buf = _roll(params_buf, params_j)

        def do_resample(args):
            sb, pb, lw, k = args
            idx = _multinomial_indices(k, lw)
            return (
                _permute_buf(sb, idx),
                _permute_buf(pb, idx),
                jnp.full((N,), -jnp.log(N)),
                idx,
            )

        def no_resample(args):
            sb, pb, lw, _k = args
            return sb, pb, lw, jnp.arange(N)

        sb_out, pb_out, lw_out, anc = jax.lax.cond(
            ess < ess_threshold, do_resample, no_resample,
            (rolled_state_buf, rolled_params_buf, normed, k_res),
        )

        out_state = jax.tree.map(lambda x: x[-1], sb_out)
        out_params = jax.tree.map(lambda x: x[-1], pb_out)
        return (sb_out, pb_out, lw_out), (
            out_state, out_params, lw_out, anc, ess, log_marg_inc,
        )

    _carry, (s_hist, th_hist, lw_hist, anc_hist, ess_hist, log_incs) = jax.lax.scan(
        body, (state_buf0, params_buf0, lw0), (y[1:], step_keys)
    )

    x_history = jax.tree.map(
        lambda a, b: jnp.concatenate([a[None], b]), new_states0, s_hist
    )
    th_history = jax.tree.map(
        lambda a, b: jnp.concatenate([a[None], b]), params0, th_hist
    )
    logw_history = jnp.concatenate([lw0[None], lw_hist])
    ess_history = jnp.concatenate([jnp.asarray([_ess(lw0)]), ess_hist])
    log_lik = jnp.cumsum(
        jnp.concatenate(
            [jnp.asarray([logsumexp(init_log_w) - jnp.log(N)]), log_incs]
        )
    )

    return LiuWestGDMResult(
        particles_history=x_history,
        params_history=th_history,
        log_weights_history=logw_history,
        ancestors=anc_hist,
        ess_history=ess_history,
        log_lik=log_lik,
    )
