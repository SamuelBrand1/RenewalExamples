"""Shared scan loop for the four Liu-West bootstrap PF runners.

The four runners (`run_liu_west`, `run_liu_west_sigma`, `run_liu_west_trend`,
`run_liu_west_discrete`) differ in exactly four things:

  1. The pfjax ``Model`` instance (encapsulates ``prior_sample``, ``state_sample``,
     ``meas_lpdf``).
  2. The initial parameter-cloud sampler.
  3. The parameter pytree itself (NamedTuple of (N,) leaves) and therefore its
     dimensionality.
  4. The variant-specific Result NamedTuple.

Items 1–3 are passed in to ``run_liu_west_core`` below.  Item 4 is the wrapping
each per-variant runner does around the returned 6-tuple.  Everything else —
the Liu-West shrink-jitter, ESS-triggered multinomial resampling, Steyn-style
fixed-lag re-permutation of the recent state/param history, and the
log-marginal accumulation — is identical across variants and lives here.
"""

from __future__ import annotations

from typing import Callable, TypeVar

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array
from jax.scipy.special import logsumexp

from smc_renewal.config import ModelConfig
from smc_renewal.pf.liu_west import shrink_jitter

StateT = TypeVar("StateT")
ParamsT = TypeVar("ParamsT")


def _stratified_resample_indices(key: Array, log_weights: Array) -> Array:
    N = log_weights.shape[0]
    lZ = logsumexp(log_weights)
    w = jnp.exp(log_weights - lZ)
    cumw = jnp.cumsum(w)
    u = jr.uniform(key, (N,))
    positions = (jnp.arange(N) + u) / N
    return jnp.searchsorted(cumw, positions).clip(0, N - 1)


def _multinomial_indices(key: Array, log_weights: Array) -> Array:
    """Multinomial resampling — required by ``pfjax.particle_smooth``."""
    N = log_weights.shape[0]
    lZ = logsumexp(log_weights)
    w = jnp.exp(log_weights - lZ)
    return jr.choice(key, N, shape=(N,), p=w)


def _ess(log_weights: Array) -> Array:
    lZ = logsumexp(log_weights)
    w = jnp.exp(log_weights - lZ)
    return 1.0 / jnp.sum(w * w)


def run_liu_west_core(
    key: Array,
    cfg: ModelConfig,
    y: Array,
    *,
    model,
    sample_initial_params: Callable[[Array, ModelConfig, int], ParamsT],
    n_particles: int,
    h: float,
    ess_threshold: float,
    fixed_lag_L: int,
) -> tuple[StateT, ParamsT, Array, Array, Array, Array]:
    """Generic Liu-West bootstrap PF loop.

    Returns ``(particles_history, params_history, log_weights_history,
    ancestors, ess_history, log_lik)`` — shapes match the per-variant
    ``LiuWest*Result`` NamedTuples.  Per-variant runners wrap this 6-tuple
    in their own Result type so call sites stay unchanged.

    Parameters
    ----------
    model
        Any object exposing ``prior_sample(key, theta) -> state``,
        ``state_sample(key, state, theta) -> state`` and
        ``meas_lpdf(y, state, theta) -> scalar``.  In practice this is one of
        ``RenewalModel{,Sigma,Trend,Discrete}``.
    sample_initial_params
        Variant-specific draw from the parameter prior.  Signature
        ``(key, cfg, N) -> ParamsT`` where ParamsT is the variant's
        ``ParticleParams*`` NamedTuple.
    fixed_lag_L
        Steyn-style fixed-lag resampling window.  When resampling at time t,
        the *last* ``fixed_lag_L`` time-steps of state+param history are
        permuted by the ancestor indices (rather than just the current cloud).
        Default ``1`` is the standard bootstrap-PF behaviour.
    """
    T = y.shape[0]
    L_buf = max(int(fixed_lag_L), 1)

    k_init_state, k_init_params, k_run = jr.split(key, 3)

    params0 = sample_initial_params(k_init_params, cfg, n_particles)
    state_keys = jr.split(k_init_state, n_particles)
    states0 = jax.vmap(lambda kk, th: model.prior_sample(kk, th))(state_keys, params0)

    init_loglik = jax.vmap(lambda x, th: model.meas_lpdf(y[0], x, th))(states0, params0)
    lw0 = init_loglik - logsumexp(init_loglik)

    step_keys = jr.split(k_run, T - 1)

    state_buf0 = jax.tree.map(
        lambda x: jnp.broadcast_to(x[None], (L_buf,) + x.shape), states0
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

        propose_keys = jr.split(k_step, n_particles)
        new_states = jax.vmap(model.state_sample)(propose_keys, states, params_j)

        log_lik = jax.vmap(lambda x, th: model.meas_lpdf(y_t, x, th))(new_states, params_j)
        unnorm = logw + log_lik
        log_marg_inc = logsumexp(unnorm) - logsumexp(logw)
        normed = unnorm - logsumexp(unnorm)
        ess = _ess(normed)

        rolled_state_buf = _roll(state_buf, new_states)
        rolled_params_buf = _roll(params_buf, params_j)

        def do_resample(args):
            sb, pb, lw, k = args
            idx = _multinomial_indices(k, lw)
            return (
                _permute_buf(sb, idx),
                _permute_buf(pb, idx),
                jnp.full((n_particles,), -jnp.log(n_particles)),
                idx,
            )

        def no_resample(args):
            sb, pb, lw, _k = args
            return sb, pb, lw, jnp.arange(n_particles)

        sb_out, pb_out, lw_out, anc = jax.lax.cond(
            ess < ess_threshold, do_resample, no_resample,
            (rolled_state_buf, rolled_params_buf, normed, k_res),
        )

        out_state = jax.tree.map(lambda x: x[-1], sb_out)
        out_params = jax.tree.map(lambda x: x[-1], pb_out)
        return (sb_out, pb_out, lw_out), (out_state, out_params, lw_out, anc, ess, log_marg_inc)

    _carry, (s_hist, th_hist, lw_hist, anc_hist, ess_hist, log_incs) = jax.lax.scan(
        body, (state_buf0, params_buf0, lw0), (y[1:], step_keys)
    )

    x_history = jax.tree.map(lambda a, b: jnp.concatenate([a[None], b]), states0, s_hist)
    th_history = jax.tree.map(lambda a, b: jnp.concatenate([a[None], b]), params0, th_hist)
    logw_history = jnp.concatenate([lw0[None], lw_hist])
    ess_history = jnp.concatenate([jnp.asarray([_ess(lw0)]), ess_hist])
    log_lik = jnp.cumsum(
        jnp.concatenate([jnp.asarray([logsumexp(init_loglik) - jnp.log(n_particles)]), log_incs])
    )

    return x_history, th_history, logw_history, anc_hist, ess_history, log_lik


def extend_liu_west_core(
    key: Array,
    cfg: ModelConfig,
    y_new: Array,
    *,
    model,
    particles: StateT,
    params: ParamsT,
    log_weights: Array,
    h: float,
    ess_threshold: float,
) -> tuple[StateT, ParamsT, Array, Array, Array, Array]:
    """Generic sequential-update loop for a Liu-West bootstrap PF.

    Returns the same 6-tuple shape as ``run_liu_west_core``: per-step history
    of ``(particles, params, log_weights, ancestors, ess, log_lik)``.

    No fixed-lag buffer: sequential updates always re-permute only the current
    cloud on resample.  The reasoning is that the recent past was already
    constrained by the previous fit, so re-permuting deep history adds nothing.

    ``model`` is any object with ``state_sample(key, state, theta) -> state``
    and ``meas_lpdf(y, state, theta) -> scalar``.  ``particles`` and ``params``
    are the per-particle (state, theta) clouds at the end of the previous fit;
    ``log_weights`` are the corresponding (normalized) log-weights.
    """
    del cfg  # unused at this level; retained for parity with run_liu_west_core
    N = log_weights.shape[0]
    T_new = y_new.shape[0]
    step_keys = jr.split(key, T_new)

    def body(carry, inputs):
        states, thetas, logw = carry
        y_t, k = inputs
        k_lw, k_step, k_res = jr.split(k, 3)
        thetas_j = shrink_jitter(thetas, logw, h, k_lw)
        propose_keys = jr.split(k_step, N)
        new_states = jax.vmap(model.state_sample)(propose_keys, states, thetas_j)
        log_lik = jax.vmap(lambda x, th: model.meas_lpdf(y_t, x, th))(new_states, thetas_j)
        unnorm = logw + log_lik
        log_marg_inc = logsumexp(unnorm) - logsumexp(logw)
        normed = unnorm - logsumexp(unnorm)
        ess = _ess(normed)

        def do_resample(args):
            s, th, lw, k = args
            idx = _multinomial_indices(k, lw)
            return (
                jax.tree.map(lambda a: a[idx], s),
                jax.tree.map(lambda a: a[idx], th),
                jnp.full((N,), -jnp.log(N)),
                idx,
            )

        def no_resample(args):
            s, th, lw, _k = args
            return s, th, lw, jnp.arange(N)

        s_out, th_out, lw_out, anc = jax.lax.cond(
            ess < ess_threshold, do_resample, no_resample,
            (new_states, thetas_j, normed, k_res),
        )
        return (s_out, th_out, lw_out), (s_out, th_out, lw_out, anc, ess, log_marg_inc)

    _carry, (s_hist, th_hist, lw_hist, anc_hist, ess_hist, log_incs) = jax.lax.scan(
        body, (particles, params, log_weights), (y_new, step_keys)
    )
    return s_hist, th_hist, lw_hist, anc_hist, ess_hist, jnp.cumsum(log_incs)
