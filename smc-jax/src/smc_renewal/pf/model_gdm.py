"""Model E: discrete renewal + GDM observation delay with a guided proposal.

Dynamics match Model D exactly (velocity-driven log Rt and log F, Poisson
infections, immigration μ).  Two new things:

  1. **GDM observation delay** (Stoner et al).  Each cohort's eventually-
     observable cases are stick-broken across reporting stages via
     independent Beta-Binomials, with per-stage Beta means controlled by a
     probit-linear-in-stage law.  Replaces the simple delay-conv + NegBin
     observation of Model D.

  2. **Guided proposal** for the cohort partition.  At each step we sample
     ``(O_0, …, O_{L−1}) ~ MultivariateHypergeometric(U, y_t)`` — which
     automatically (a) hits ``Σ O_s = y_t`` and (b) respects per-cohort
     budget ``O_s ≤ U[s]``.  The IS weight against the target BetaBin
     product has a clean closed form (binomial coefficients cancel).

The Liu-West cloud is 6-D:
``(log σ_vR, log σ_vF, log μ, b_0, b_1, log_M)``.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
import jax.random as jr
from jax import Array
from jax.scipy.special import gammaln, logsumexp

from smc_renewal.config import ModelConfig
from smc_renewal.observation_gdm import (
    betabinom_loglik,
    gdm_beta_params,
    gdm_hypergeom_logweight,
)
from smc_renewal.transition import _pad_pmf


# Static upper bound on the support of any single univariate-Binomial draw
# (used for Poisson-thinned ascertainment E_t).  Must be ≥ N_t across all
# particles, stages, timesteps.
GDM_MAX_K = 800

# Static upper bound on y_t — used by the Wallenius sequential sampler to
# size its fixed-length inner scan.  Iterations beyond y_t are masked out.
WALLENIUS_MAX_Y = 250


class ParticleStateGDM(NamedTuple):
    """Model E state: same dynamics state as Model D, plus a U-buffer that
    tracks per-cohort remaining unreported counts."""

    log_Rt: Array
    v_R: Array
    log_F: Array
    v_F: Array
    log_I0: Array
    I_buf: Array   # shape (L,), Poisson draws — used for renewal
    U_buf: Array   # shape (L,), remaining unreported per cohort age


class ParticleParamsGDM(NamedTuple):
    """6-D Liu-West cloud for Model E."""

    log_sigma_vR: Array
    log_sigma_vF: Array
    log_mu: Array
    b_0: Array
    b_1: Array
    log_M: Array


# ---------------------------------------------------------------------------
# Discrete-distribution samplers (JAX has no native Binomial / Hypergeometric)
# ---------------------------------------------------------------------------

def _sample_binomial(key: Array, n: Array, p: Array) -> Array:
    """``X ~ Binomial(n, p)`` via inverse-CDF over a static support.

    ``n`` and ``p`` may be traced scalars.  Support truncated at
    ``GDM_MAX_K``; invalid values masked with ``−∞``.  Returns int64.
    """
    k_range = jnp.arange(GDM_MAX_K + 1, dtype=jnp.float64)
    n_f = jnp.asarray(n, dtype=jnp.float64)
    log_p = jnp.log(jnp.maximum(p, 1e-30))
    log_1mp = jnp.log(jnp.maximum(1.0 - p, 1e-30))
    log_pmf = (
        gammaln(n_f + 1.0)
        - gammaln(k_range + 1.0)
        - gammaln(n_f - k_range + 1.0)
        + k_range * log_p
        + (n_f - k_range) * log_1mp
    )
    valid = (k_range <= n_f)
    log_pmf = jnp.where(valid, log_pmf, -jnp.inf)
    log_pmf = log_pmf - logsumexp(log_pmf)
    return jr.categorical(key, log_pmf).astype(jnp.int64)


def sample_univar_hypergeom(
    key: Array, N: Array, K: Array, n: Array
) -> Array:
    """``X ~ Hypergeometric(N, K, n)``: draw ``n`` without replacement from a
    pool of ``N`` with ``K`` marked, count marked.

    Pmf: P(X=k) ∝ C(K, k) · C(N−K, n−k).  Support:
    ``max(0, n−(N−K)) ≤ k ≤ min(K, n)``.  Implemented via inverse-CDF over
    a static range ``[0, GDM_MAX_K]`` with mask outside the support.
    Returns int64.
    """
    k_range = jnp.arange(GDM_MAX_K + 1, dtype=jnp.float64)
    N_f = jnp.asarray(N, dtype=jnp.float64)
    K_f = jnp.asarray(K, dtype=jnp.float64)
    n_f = jnp.asarray(n, dtype=jnp.float64)

    log_C_K_k = gammaln(K_f + 1.0) - gammaln(k_range + 1.0) - gammaln(K_f - k_range + 1.0)
    log_C_NmK_nmk = (
        gammaln(N_f - K_f + 1.0)
        - gammaln(n_f - k_range + 1.0)
        - gammaln(N_f - K_f - (n_f - k_range) + 1.0)
    )
    log_pmf = log_C_K_k + log_C_NmK_nmk

    valid = (
        (k_range <= K_f)
        & (k_range <= n_f)
        & (k_range >= n_f - (N_f - K_f))
    )
    log_pmf = jnp.where(valid, log_pmf, -jnp.inf)
    log_pmf = log_pmf - logsumexp(log_pmf)
    return jr.categorical(key, log_pmf).astype(jnp.int64)


def sample_mv_hypergeom(key: Array, U: Array, y_t: Array, L: int) -> Array:
    """Sequential multivariate-hypergeometric sample.

    Draws ``y_t`` without replacement from a pool of ``Σ U`` balls split
    into ``L`` cohorts of size ``U[s]``.  Returns shape-(L,) counts ``O``
    with ``Σ O = y_t`` and ``O_s ≤ U[s]`` by construction.

    Implementation: L−1 sequential univariate-Hypergeometric draws; last
    stage is the residual.
    """
    keys = jr.split(key, L - 1)

    def body(carry, inputs):
        N_remain, y_remain = carry
        K, key_s = inputs
        O_s = sample_univar_hypergeom(key_s, N_remain, K, y_remain)
        return (N_remain - K, y_remain - O_s), O_s

    init = (jnp.sum(U).astype(jnp.int64), y_t.astype(jnp.int64))
    _, O_head = jax.lax.scan(body, init, (U[: L - 1].astype(jnp.int64), keys))
    O_last = y_t.astype(jnp.int64) - jnp.sum(O_head)
    return jnp.concatenate([O_head, O_last[None]])


def sample_wallenius(
    key: Array, U: Array, weights: Array, y_t: Array, L: int
) -> tuple[Array, Array]:
    """Multivariate **Wallenius** noncentral hypergeometric — sequential
    weighted-without-replacement.

    For each of ``y_t`` draws, pick cohort ``s`` with probability proportional
    to ``U_curr[s] · weights[s]``, then decrement ``U_curr[s]``.  Tracks
    ``log_q_ordered = Σ_i log π^(i)_{s_i}`` (the proposal density on
    orderings) for the importance-weight bookkeeping.

    Inner scan is fixed-length ``WALLENIUS_MAX_Y``; iterations beyond
    ``y_t`` are masked out so the sampler is JIT-compatible with varying
    ``y_t``.

    Returns ``(O, log_q_ordered)``.  ``O`` shape (L,); ``log_q_ordered``
    scalar.
    """
    keys = jr.split(key, WALLENIUS_MAX_Y)

    def body(carry, inp):
        U_curr, log_q_acc = carry
        i, k_step = inp
        active = i < y_t
        w = U_curr.astype(jnp.float64) * weights
        # log_w = -inf for empty cohorts (zero proposal mass).
        log_w = jnp.where(U_curr > 0, jnp.log(jnp.maximum(w, 1e-30)), -jnp.inf)
        log_norm = logsumexp(log_w)
        s_drawn = jr.categorical(k_step, log_w)
        log_pi = log_w[s_drawn] - log_norm
        U_new = jnp.where(active, U_curr.at[s_drawn].add(-1), U_curr)
        log_q_new = jnp.where(active, log_q_acc + log_pi, log_q_acc)
        return (U_new, log_q_new), s_drawn

    init = (U.astype(jnp.int64), jnp.asarray(0.0))
    indices = jnp.arange(WALLENIUS_MAX_Y)
    (_, log_q_total), s_seq = jax.lax.scan(body, init, (indices, keys))

    # Count which stage was drawn at each ACTIVE iteration.
    valid = indices < y_t
    one_hot = jax.nn.one_hot(s_seq, L, dtype=jnp.int64)
    O = jnp.sum(one_hot * valid[:, None].astype(jnp.int64), axis=0)
    return O, log_q_total


def wallenius_logweight(
    O: Array, U: Array, log_q_ordered: Array, alpha_s: Array, beta_s: Array
) -> Array:
    """IS log-weight for the Wallenius proposal against the BetaBin product.

    log w = Σ_s log BetaBin(O_s; U_s, α_s, β_s)
          − [gammaln(y_t+1) − Σ_s gammaln(O_s+1)]        ← multinomial coef
          − log q_ordered                                   ← proposal on ordering
    """
    O_f = O.astype(jnp.float64)
    U_f = U.astype(jnp.float64)
    y_t = jnp.sum(O_f)
    log_target = jnp.sum(betabinom_loglik(O_f, U_f, alpha_s, beta_s))
    log_mc = gammaln(y_t + 1.0) - jnp.sum(gammaln(O_f + 1.0))
    return log_target - log_mc - log_q_ordered


# ---------------------------------------------------------------------------
# Per-particle dynamics + proposal
# ---------------------------------------------------------------------------

def step_gdm_dynamics(
    state: ParticleStateGDM,
    params: ParticleParamsGDM,
    eta_R: Array,
    eta_F: Array,
    k_pois: Array,
    k_asc: Array,
    cfg: ModelConfig,
) -> tuple[ParticleStateGDM, Array, Array, Array]:
    """Advance levels/velocities, draw ``N_t`` and ``E_t``.  Returns
    ``(state_partial, N_t, E_t, λ_t)``; buffers not yet shifted."""
    L = cfg.buffer_len
    g_pad = _pad_pmf(cfg.generation_interval, L)

    sigma_vR = jnp.maximum(jnp.exp(params.log_sigma_vR), cfg.sigma_floor)
    sigma_vF = jnp.maximum(jnp.exp(params.log_sigma_vF), cfg.sigma_floor)
    mu = jnp.exp(params.log_mu)
    alpha_asc = cfg.ascertainment_alpha

    log_Rt_new = state.log_Rt + state.v_R
    log_F_new = state.log_F + state.v_F
    v_R_new = state.v_R + sigma_vR * eta_R
    v_F_new = state.v_F + sigma_vF * eta_F

    g_conv_I = jnp.dot(g_pad, state.I_buf.astype(jnp.float64))
    F_new = jnp.exp(log_F_new)
    log_Rt_clipped = jnp.clip(log_Rt_new, -20.0, 20.0)
    exponent = jnp.clip(log_Rt_clipped - F_new * g_conv_I, -20.0, 20.0)
    Rt_eff = jnp.exp(exponent)
    lambda_t = jnp.clip(mu + Rt_eff * g_conv_I, 1e-12, 1e12)

    N_t = jr.poisson(k_pois, lambda_t).astype(jnp.int64)
    E_t = _sample_binomial(k_asc, N_t, jnp.asarray(alpha_asc, dtype=jnp.float64))

    state_partial = ParticleStateGDM(
        log_Rt=log_Rt_new,
        v_R=v_R_new,
        log_F=log_F_new,
        v_F=v_F_new,
        log_I0=state.log_I0,
        I_buf=state.I_buf,
        U_buf=state.U_buf,
    )
    return state_partial, N_t, E_t, lambda_t


def propose_gdm_partition(
    key: Array,
    U_buf_prev: Array,
    E_t: Array,
    y_t: Array,
    b_0: Array,
    b_1: Array,
    log_M: Array,
    cfg: ModelConfig,
) -> tuple[Array, Array]:
    """Guided proposal: sample the cohort partition via **Wallenius**
    noncentral hypergeometric (weighted sequential without-replacement) and
    return ``(O, Δlog w)``.

    Weights are the per-stage Beta means ``p_s = α_s / (α_s + β_s)``, which
    matches the target's marginal mean structure — substantially lower IS
    weight variance than the vanilla multivariate-hypergeometric proposal.
    Cost: ``O(y_t · L)`` per particle per step.

    ``Δlog w = −∞`` when ``y_t > Σ U`` (incompatible history).
    """
    L = cfg.buffer_len
    U_prop = jnp.concatenate(
        [E_t[None], U_buf_prev[:-1]]
    ).astype(jnp.int64)
    sum_U = jnp.sum(U_prop)

    alpha_s, beta_s = gdm_beta_params(b_0, b_1, log_M, L)
    p_s = alpha_s / (alpha_s + beta_s)

    feasible = y_t <= sum_U
    safe_y = jnp.where(feasible, y_t, jnp.zeros_like(y_t))

    O, log_q_ord = sample_wallenius(key, U_prop, p_s, safe_y, L)
    log_w = wallenius_logweight(O, U_prop, log_q_ord, alpha_s, beta_s)
    log_w = jnp.where(feasible, log_w, -jnp.inf)
    O = jnp.where(feasible, O, jnp.zeros_like(O))
    return O, log_w


def apply_partition_and_shift(
    state_partial: ParticleStateGDM,
    N_t: Array,
    E_t: Array,
    O: Array,
) -> ParticleStateGDM:
    """Finalise the step: shift ``I_buf`` (prepend ``N_t``); set
    ``U_buf_new = U_prop − O`` where ``U_prop = [E_t, U_buf_prev[:-1]]``.

    No additional shift on ``U_buf_new`` because the indexing convention is
    ``U_buf[k] = remaining for cohort (t)−k`` at the *end* of step t (which
    becomes ``cohort (t+1)−1−k`` at the start of step t+1 — same value)."""
    I_buf_new = jnp.concatenate(
        [N_t.astype(state_partial.I_buf.dtype)[None], state_partial.I_buf[:-1]]
    )
    U_prop = jnp.concatenate(
        [E_t.astype(state_partial.U_buf.dtype)[None], state_partial.U_buf[:-1]]
    )
    U_buf_new = U_prop - O.astype(U_prop.dtype)
    return state_partial._replace(I_buf=I_buf_new, U_buf=U_buf_new)


def propose_step(
    key: Array,
    state_prev: ParticleStateGDM,
    theta: ParticleParamsGDM,
    y_t: Array,
    cfg: ModelConfig,
) -> tuple[ParticleStateGDM, Array]:
    """One full guided-proposal step: dynamics → ascertainment → partition.

    Returns ``(new_state, log_w_inc)``.  Replaces the bootstrap
    ``state_sample`` + ``meas_lpdf`` pair for Model E.
    """
    k_eta, k_pois, k_asc, k_part = jr.split(key, 4)
    eta = jr.normal(k_eta, (2,))
    state_partial, N_t, E_t, _ = step_gdm_dynamics(
        state_prev, theta, eta[0], eta[1], k_pois, k_asc, cfg
    )
    y_t_int = jnp.asarray(y_t, dtype=jnp.int64)
    O, log_w_inc = propose_gdm_partition(
        k_part, state_prev.U_buf, E_t, y_t_int,
        theta.b_0, theta.b_1, theta.log_M, cfg,
    )
    new_state = apply_partition_and_shift(state_partial, N_t, E_t, O)
    return new_state, log_w_inc


# ---------------------------------------------------------------------------
# Unguided forward sampler — used by truth simulation and by the forecaster
# ---------------------------------------------------------------------------

def forward_step(
    key: Array,
    state_prev: ParticleStateGDM,
    theta: ParticleParamsGDM,
    cfg: ModelConfig,
) -> tuple[ParticleStateGDM, Array, Array]:
    """Forward-simulate one step (no conditioning on a y_t).

    Returns ``(new_state, y_t_simulated, O)``: the partition is drawn fresh
    from independent BetaBin per (cohort, age) — the GDM data-generating
    mechanism.
    """
    k_eta, k_pois, k_asc, k_part = jr.split(key, 4)
    eta = jr.normal(k_eta, (2,))
    state_partial, N_t, E_t, _ = step_gdm_dynamics(
        state_prev, theta, eta[0], eta[1], k_pois, k_asc, cfg
    )
    L = cfg.buffer_len
    alpha_s, beta_s = gdm_beta_params(theta.b_0, theta.b_1, theta.log_M, L)
    U_prop = jnp.concatenate(
        [E_t.astype(state_prev.U_buf.dtype)[None], state_prev.U_buf[:-1]]
    ).astype(jnp.int64)
    keys_stage = jr.split(k_part, L)

    def per_stage(U_s, a_s, b_s, k_s):
        k_q, k_b = jr.split(k_s, 2)
        q = jr.beta(k_q, a_s, b_s)
        return _sample_binomial(k_b, U_s, q)

    O = jax.vmap(per_stage)(U_prop, alpha_s, beta_s, keys_stage)
    y_t = jnp.sum(O).astype(jnp.float64)
    new_state = apply_partition_and_shift(state_partial, N_t, E_t, O)
    return new_state, y_t, O


# ---------------------------------------------------------------------------
# Priors
# ---------------------------------------------------------------------------

def prior_sample_gdm(
    key: Array, cfg: ModelConfig
) -> ParticleStateGDM:
    """Initial state draw for the PF.  Matches ``RenewalModelDiscrete.prior_sample``
    on the dynamics; ``U_buf`` initialised to zeros — for a fresh outbreak
    there are no "phantom" pre-history cohorts with unreported cases."""
    keys = jr.split(key, 5)
    log_Rt = cfg.init_log_Rt_mean + cfg.init_log_Rt_sd * jr.normal(keys[0])
    v_R = cfg.init_v_R_mean + cfg.init_v_R_sd * jr.normal(keys[1])
    log_F = cfg.init_log_F_mean + cfg.init_log_F_sd * jr.normal(keys[2])
    v_F = cfg.init_v_F_mean + cfg.init_v_F_sd * jr.normal(keys[3])
    log_I0 = cfg.init_log_I0_mean + cfg.init_log_I0_sd * jr.normal(keys[4])
    I0_int = jnp.maximum(jnp.round(jnp.exp(log_I0)), 1.0).astype(jnp.int64)
    I_buf = jnp.full((cfg.buffer_len,), I0_int)
    U_buf = jnp.zeros((cfg.buffer_len,), dtype=jnp.int64)
    return ParticleStateGDM(
        log_Rt=log_Rt, v_R=v_R, log_F=log_F, v_F=v_F,
        log_I0=log_I0, I_buf=I_buf, U_buf=U_buf,
    )


def sample_initial_gdm_params(
    key: Array, cfg: ModelConfig, N: int
) -> ParticleParamsGDM:
    keys = jr.split(key, 6)
    return ParticleParamsGDM(
        log_sigma_vR=cfg.init_log_sigma_vR_mean
        + cfg.init_log_sigma_vR_sd * jr.normal(keys[0], (N,)),
        log_sigma_vF=cfg.init_log_sigma_vF_mean
        + cfg.init_log_sigma_vF_sd * jr.normal(keys[1], (N,)),
        log_mu=cfg.init_log_mu_mean + cfg.init_log_mu_sd * jr.normal(keys[2], (N,)),
        b_0=cfg.init_b0_mean + cfg.init_b0_sd * jr.normal(keys[3], (N,)),
        b_1=cfg.init_b1_mean + cfg.init_b1_sd * jr.normal(keys[4], (N,)),
        log_M=cfg.init_log_M_mean + cfg.init_log_M_sd * jr.normal(keys[5], (N,)),
    )
