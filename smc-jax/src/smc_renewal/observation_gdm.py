"""Generalised Dirichlet Multinomial observation model (Stoner et al).

Each cohort's eventually-observable cases are stick-broken across reporting
stages via independent Beta-Binomials.  Per-stage Beta hyperparameters are
parameterised on the probit scale:

    Φ⁻¹(p_s)  =  b_0 + b_1 · s             s = 0, 1, …, L−1
    α_s       =  p_s · M
    β_s       =  (1 − p_s) · M             M = exp(log_M)

For the guided PF in ``pf/model_gdm.py``, the cohort partition is proposed
via a multivariate hypergeometric draw, and the IS log-weight contribution
has a clean closed form where the ``C(U[s], O_s)`` terms cancel between
target (BetaBin product) and proposal:

    Δlog w  =  Σ_s [log B(O_s + α_s, U[s] − O_s + β_s) − log B(α_s, β_s)]
             +  log C(Σ_s U[s], y_t)

with `y_t = Σ_s O_s`.
"""

from __future__ import annotations

import jax.numpy as jnp
from jax import Array
from jax.scipy.special import gammaln
from jax.scipy.stats import norm


def gdm_beta_params(
    b_0: Array, b_1: Array, log_M: Array, L: int
) -> tuple[Array, Array]:
    """Per-stage (α_s, β_s) for the GDM Beta marginals.

    Minimum reporting delay is **1 day**: stage 0 corresponds to the same-day
    cohort and is forced to ``p_0 ≈ 0`` (essentially never reported same-day).
    The probit-linear law then applies from stage 1 onwards:

        Φ⁻¹(p_s)  =  b_0  +  b_1 · (s − 1)        for s = 1, 2, …, L−1

    So ``b_0`` is the probit value at the *first reportable lag* (lag 1).

    ``b_0``, ``b_1``, ``log_M`` are scalars (or shape-() arrays); ``L`` is a
    static int.  Returns two shape-(L,) arrays.
    """
    stages = jnp.arange(L, dtype=jnp.float64)
    p_s_raw = norm.cdf(b_0 + b_1 * (stages - 1.0))
    # Force p_0 ≈ 0 — today's cohort can't be reported today.
    p_s = jnp.where(stages < 0.5, 1e-6, p_s_raw)
    # Pin away from 0/1 to keep α_s, β_s strictly positive and finite.
    p_s = jnp.clip(p_s, 1e-6, 1.0 - 1e-6)
    M = jnp.exp(log_M)
    alpha_s = p_s * M
    beta_s = (1.0 - p_s) * M
    return alpha_s, beta_s


def log_beta_fn(a: Array, b: Array) -> Array:
    """``log B(a, b) = gammaln(a) + gammaln(b) − gammaln(a + b)``."""
    return gammaln(a) + gammaln(b) - gammaln(a + b)


def log_binom_coef(n: Array, k: Array) -> Array:
    """``log C(n, k) = gammaln(n+1) − gammaln(k+1) − gammaln(n−k+1)``."""
    return gammaln(n + 1.0) - gammaln(k + 1.0) - gammaln(n - k + 1.0)


def gdm_hypergeom_logweight(
    O: Array, U: Array, alpha_s: Array, beta_s: Array
) -> Array:
    """Importance-weight contribution from the multivariate-hypergeometric
    proposal against the target BetaBin product.

    All inputs shape (L,) (per-stage).  Returns a scalar (the IS log-weight
    contribution; up to a particle-shared normalisation constant).

    Derivation: target = ∏_s BetaBin(O_s; U_s, α_s, β_s),
    proposal = MVHypergeom(U, y_t).  Binomial coefficients cancel, leaving
    only Beta-function ratios plus a single ``log C(ΣU, y_t)`` correction.
    """
    O_f = O.astype(jnp.float64)
    U_f = U.astype(jnp.float64)
    y_t = jnp.sum(O_f)
    sum_U = jnp.sum(U_f)
    log_target = jnp.sum(
        log_beta_fn(O_f + alpha_s, U_f - O_f + beta_s) - log_beta_fn(alpha_s, beta_s)
    )
    log_corr = log_binom_coef(sum_U, y_t)
    return log_target + log_corr


def betabinom_loglik(x: Array, n: Array, alpha: Array, beta: Array) -> Array:
    """Beta-Binomial log-pmf.  Used in tests; not on the PF hot path.

    log P(x | n, α, β) = log C(n, x) + log B(x+α, n−x+β) − log B(α, β).
    """
    return (
        log_binom_coef(n, x)
        + log_beta_fn(x + alpha, n - x + beta)
        - log_beta_fn(alpha, beta)
    )


def binomial_loglik(x: Array, n: Array, p: Array) -> Array:
    """Binomial log-pmf — used by the Rao-Blackwellised auxiliary-q proposal.

    log P(x | n, p) = log C(n, x) + x · log p + (n − x) · log(1 − p).
    """
    log_p = jnp.log(jnp.maximum(p, 1e-30))
    log_1mp = jnp.log(jnp.maximum(1.0 - p, 1e-30))
    return log_binom_coef(n, x) + x * log_p + (n - x) * log_1mp


def wallenius_aux_logweight(
    O: Array,
    U: Array,
    q_s: Array,
    log_q_ordered: Array,
) -> Array:
    """IS log-weight for the **Rao-Blackwellised auxiliary-q** proposal.

    Proposal:  q_s ~ Beta(α_s, β_s);  O ~ Wallenius(U, weights = q_s, y_t).
    Augmented target:
        p_aug(O, q | y_t) ∝ ∏_s Bin(O_s; U_s, q_s) · Beta(q_s; α_s, β_s) · 𝟙[ΣO=y_t]

    The Beta factors cancel between target and proposal, leaving

        Δlog w  =  Σ_s log Bin(O_s; U_s, q_s)
                 − [gammaln(y_t+1) − Σ_s gammaln(O_s+1)]    ← multinomial coef
                 − log q_ordered                             ← Wallenius on ordering

    The over-dispersion that previously made `BetaBin / Wallenius` have
    heavy IS tails is absorbed by `q_s` itself being drawn from its prior
    (rather than fixed at the mean p_s).
    """
    O_f = O.astype(jnp.float64)
    U_f = U.astype(jnp.float64)
    y_t = jnp.sum(O_f)
    log_target = jnp.sum(binomial_loglik(O_f, U_f, q_s))
    log_mc = gammaln(y_t + 1.0) - jnp.sum(gammaln(O_f + 1.0))
    return log_target - log_mc - log_q_ordered
