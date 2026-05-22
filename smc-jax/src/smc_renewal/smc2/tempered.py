"""Adaptive-tempered SMC over ``theta = (log_tau_R, log_tau_F, log_phi)``.

Outer loop:  particles in the 3-D parameter space, with weights determined by
``beta * loglik(theta) + logprior(theta)`` where ``beta`` is the tempering
parameter.  ``beta`` is increased adaptively to keep ESS at a target level.

Inner MCMC: Random-walk Metropolis (RWMH) on theta with a covariance proposal
derived from the current particle cloud — applied ``num_mcmc_steps`` times per
tempering step to rejuvenate.

``loglik(theta)`` is the UKF marginal log-likelihood from ``smc2.ukf``.

We hand-roll the adaptive-tempered loop (≈80 lines) rather than reach into the
fairly intricate ``blackjax.smc.adaptive_tempered`` API.  The hand-rolled loop
is fully jit/scan compatible.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array
from jax.scipy.special import logsumexp

from smc_renewal.config import ModelConfig
from smc_renewal.smc2.ukf import marginal_log_likelihood as _default_marginal_loglik
from smc_renewal.state import ParticleParams

# Type alias for the marginal-loglik plug-in.
MarginalLoglikFn = "Callable[[ModelConfig, Array, ParticleParams], Array]"


class SMC2State(NamedTuple):
    """Tempered SMC state at the end of one outer step."""

    particles: Array       # (N, 3)
    log_weights: Array     # (N,)
    log_lik: Array         # (N,)   cached loglik per particle (since recomputing is expensive)
    beta: Array            # scalar tempering parameter, in [0, 1]


class SMC2Result(NamedTuple):
    """Final tempered SMC output."""

    particles: Array        # (N, 3)
    log_weights: Array      # (N,)
    log_marginal: Array     # log Z estimate accumulated across tempering steps
    n_steps: int


def _logprior_fn(theta: Array, cfg: ModelConfig) -> Array:
    """Gaussian log-prior on (log_tau_R, log_tau_F, log_phi)."""
    return (
        -0.5 * ((theta[0] - cfg.prior_log_tau_R_mean) / cfg.prior_log_tau_R_sd) ** 2
        - 0.5 * ((theta[1] - cfg.prior_log_tau_F_mean) / cfg.prior_log_tau_F_sd) ** 2
        - 0.5 * ((theta[2] - cfg.prior_log_phi_mean) / cfg.prior_log_phi_sd) ** 2
    )


def _theta_to_params(theta: Array) -> ParticleParams:
    return ParticleParams(log_tau_R=theta[0], log_tau_F=theta[1], log_phi=theta[2])


def _sample_prior(key: Array, cfg: ModelConfig, N: int) -> Array:
    """Sample ``N`` initial parameter particles."""
    keys = jr.split(key, 3)
    means = jnp.array(
        [cfg.prior_log_tau_R_mean, cfg.prior_log_tau_F_mean, cfg.prior_log_phi_mean]
    )
    sds = jnp.array(
        [cfg.prior_log_tau_R_sd, cfg.prior_log_tau_F_sd, cfg.prior_log_phi_sd]
    )
    eps = jnp.stack([jr.normal(keys[i], (N,)) for i in range(3)], axis=-1)
    return means[None, :] + sds[None, :] * eps


def _ess_from_log_weights(lw: Array) -> Array:
    lZ = logsumexp(lw)
    w = jnp.exp(lw - lZ)
    return 1.0 / jnp.sum(w * w)


def _find_next_beta(
    log_lik: Array, current_beta: float, target_ess: float
) -> float:
    """Bisection for ``beta_new ∈ (current_beta, 1]`` such that ESS(beta_new) ≈ target.

    Done in Python (not jit) because it's a small while loop and we run it once
    per outer step.
    """
    def ess_at(b):
        return float(_ess_from_log_weights(b * log_lik))

    if ess_at(1.0 - current_beta) >= target_ess:  # incremental ESS at beta=1
        return 1.0
    lo, hi = 0.0, 1.0 - current_beta
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if ess_at(mid) >= target_ess:
            lo = mid
        else:
            hi = mid
    return current_beta + lo


def _stratified_resample(key: Array, log_weights: Array) -> Array:
    N = log_weights.shape[0]
    lZ = logsumexp(log_weights)
    w = jnp.exp(log_weights - lZ)
    cumw = jnp.cumsum(w)
    u = jr.uniform(key, (N,))
    positions = (jnp.arange(N) + u) / N
    return jnp.searchsorted(cumw, positions).clip(0, N - 1)


def _rwmh_step(
    key: Array,
    theta: Array,
    log_lik_theta: Array,
    log_prior_theta: Array,
    beta: float,
    cov_chol: Array,
    cfg: ModelConfig,
    y: Array,
    scale: float,
    marginal_loglik_fn=_default_marginal_loglik,
) -> tuple[Array, Array, Array]:
    """One RWMH proposal in parameter space; returns updated (theta, ll, lp)."""
    k_prop, k_acc = jr.split(key, 2)
    z = jr.normal(k_prop, (3,))
    prop = theta + scale * (cov_chol @ z)
    ll_prop = marginal_loglik_fn(cfg, y, _theta_to_params(prop))
    lp_prop = _logprior_fn(prop, cfg)
    log_alpha = (beta * ll_prop + lp_prop) - (beta * log_lik_theta + log_prior_theta)
    log_u = jnp.log(jr.uniform(k_acc, ()))
    accept = log_u < log_alpha
    return (
        jnp.where(accept, prop, theta),
        jnp.where(accept, ll_prop, log_lik_theta),
        jnp.where(accept, lp_prop, log_prior_theta),
    )


def run_smc2(
    key: Array,
    cfg: ModelConfig,
    y: Array,
    n_particles: int = 128,
    target_ess_frac: float = 0.5,
    num_mcmc_steps: int = 5,
    rwmh_scale: float = 0.5,
    max_outer_steps: int = 40,
    marginal_loglik_fn=_default_marginal_loglik,
) -> SMC2Result:
    """Run adaptive-tempered SMC over ``theta`` with a Gaussian marginal log-lik.

    ``marginal_loglik_fn(cfg, y, params) -> scalar`` defaults to the hand-rolled
    UKF in ``smc2.ukf``.  Pass ``smc2.ekf_cuthbert.marginal_log_likelihood`` to
    use the library-backed EKF instead.

    Returns the final particle cloud (with equal weights after final resample)
    and a log-Z estimate accumulated over tempering steps.
    """
    k_init, key = jr.split(key, 2)
    theta = _sample_prior(k_init, cfg, n_particles)  # (N, 3)

    # Precompute initial loglik & logprior.
    ll = jax.vmap(lambda th: marginal_loglik_fn(cfg, y, _theta_to_params(th)))(theta)
    lp = jax.vmap(lambda th: _logprior_fn(th, cfg))(theta)
    lw = jnp.full((n_particles,), -jnp.log(n_particles))
    beta = 0.0
    target_ess = target_ess_frac * n_particles
    log_marginal = 0.0

    step = 0
    while beta < 1.0 and step < max_outer_steps:
        step += 1
        # Adaptive next temperature.
        beta_new = _find_next_beta(jnp.asarray(ll), beta, target_ess)
        dbeta = beta_new - beta
        incremental_lw = lw + dbeta * ll
        # Accumulate marginal contribution.
        log_marginal = log_marginal + (logsumexp(incremental_lw) - logsumexp(lw))
        # Resample.
        key, k_res = jr.split(key, 2)
        idx = _stratified_resample(k_res, incremental_lw)
        theta = theta[idx]
        ll = ll[idx]
        lp = lp[idx]
        lw = jnp.full((n_particles,), -jnp.log(n_particles))
        beta = beta_new

        # Build proposal cov from resampled cloud.
        cov = jnp.cov(theta.T) + 1e-6 * jnp.eye(3)
        cov_chol = jnp.linalg.cholesky(cov)

        # MCMC rejuvenation: num_mcmc_steps RWMH iterations per particle.
        key, k_mcmc = jr.split(key, 2)
        mcmc_keys = jr.split(k_mcmc, num_mcmc_steps)

        def per_particle_chain(args, k_inner):
            th, lli, lpi = args

            def one_step(state, k):
                t, l, p = state
                return _rwmh_step(
                    k, t, l, p, beta, cov_chol, cfg, y, rwmh_scale,
                    marginal_loglik_fn=marginal_loglik_fn,
                ), None

            keys_inner = jr.split(k_inner, num_mcmc_steps)
            (new_th, new_ll, new_lp), _ = jax.lax.scan(
                one_step, (th, lli, lpi), keys_inner
            )
            return new_th, new_ll, new_lp

        particle_keys = jr.split(k_mcmc, n_particles)
        theta, ll, lp = jax.vmap(per_particle_chain)((theta, ll, lp), particle_keys)

    return SMC2Result(
        particles=theta,
        log_weights=lw,
        log_marginal=jnp.asarray(log_marginal),
        n_steps=step,
    )
