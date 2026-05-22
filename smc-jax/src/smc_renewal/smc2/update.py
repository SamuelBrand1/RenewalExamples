"""Sequential update of an SMC^2 parameter posterior with new observations.

Given a posterior particle cloud over ``theta`` fitted on data through ``T``,
and new data ``y_{T+1..T+k}``, we re-target

    p(theta | y_{1..T+k}) ∝ p(y_{1..T+k} | theta) p(theta)
                        = p(y_{1..T} | theta) p(theta) * p(y_{T+1..T+k} | y_{1..T}, theta).

The standard "IBIS"-style trick: importance-reweight by the *conditional*
likelihood ratio ``Δ = log p(y_{T+1..T+k} | theta) - log p(y_{T+1..T+k} | y_{1..T}, theta_marginal)``,
which for our state-space model simplifies to recomputing the marginal log-lik
over the longer series and taking the difference from the cached value.

If ESS degrades below threshold we run a short tempering sub-step + RWMH.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array
from jax.scipy.special import logsumexp

from smc_renewal.config import ModelConfig
from smc_renewal.smc2.tempered import (
    SMC2Result,
    _ess_from_log_weights,
    _logprior_fn,
    _rwmh_step,
    _stratified_resample,
    _theta_to_params,
)
from smc_renewal.smc2.ukf import marginal_log_likelihood as _default_marginal_loglik


def extend_smc2(
    key: Array,
    cfg: ModelConfig,
    y_full: Array,
    cached_log_lik_prefix: Array,
    prev_result: SMC2Result,
    num_mcmc_steps: int = 5,
    rwmh_scale: float = 0.3,
    ess_threshold_frac: float = 0.5,
    marginal_loglik_fn=_default_marginal_loglik,
) -> tuple[SMC2Result, Array]:
    """Update the parameter posterior with the longer series ``y_full``.

    ``cached_log_lik_prefix`` has shape ``(N,)`` and is the marginal log-lik of
    the prefix evaluated at the previous particle locations (using the same
    ``marginal_loglik_fn``).

    Returns ``(updated_result, new_cached_log_lik)``.
    """
    theta = prev_result.particles
    N = theta.shape[0]

    # Compute log-lik over the full series at each particle.
    new_loglik = jax.vmap(
        lambda th: marginal_loglik_fn(cfg, y_full, _theta_to_params(th))
    )(theta)
    dlw = new_loglik - cached_log_lik_prefix
    lw = prev_result.log_weights + dlw
    log_marginal_update = logsumexp(lw) - logsumexp(prev_result.log_weights)
    lw = lw - logsumexp(lw)
    ess = _ess_from_log_weights(lw)

    # Resample + rejuvenate if ESS dipped.
    key, k_res, k_mcmc = jr.split(key, 3)

    def do_rejuvenate():
        idx = _stratified_resample(k_res, lw)
        th = theta[idx]
        ll = new_loglik[idx]
        lp = jax.vmap(lambda x: _logprior_fn(x, cfg))(th)
        cov = jnp.cov(th.T) + 1e-6 * jnp.eye(3)
        cov_chol = jnp.linalg.cholesky(cov)

        def per_particle_chain(args, k_inner):
            t, l, p = args
            keys_inner = jr.split(k_inner, num_mcmc_steps)

            def one_step(state, k):
                tt, ll_, pp = state
                return _rwmh_step(
                    k, tt, ll_, pp, 1.0, cov_chol, cfg, y_full, rwmh_scale,
                    marginal_loglik_fn=marginal_loglik_fn,
                ), None

            (nt, nl, np_), _ = jax.lax.scan(one_step, (t, l, p), keys_inner)
            return nt, nl, np_

        particle_keys = jr.split(k_mcmc, N)
        nt, nl, _ = jax.vmap(per_particle_chain)((th, ll, lp), particle_keys)
        return nt, nl, jnp.full((N,), -jnp.log(N))

    def no_rejuvenate():
        return theta, new_loglik, lw

    rejuvenate = ess < ess_threshold_frac * N
    new_theta, new_ll, new_lw = jax.lax.cond(
        rejuvenate, do_rejuvenate, no_rejuvenate
    )

    return (
        SMC2Result(
            particles=new_theta,
            log_weights=new_lw,
            log_marginal=prev_result.log_marginal + log_marginal_update,
            n_steps=prev_result.n_steps + 1,
        ),
        new_ll,
    )
