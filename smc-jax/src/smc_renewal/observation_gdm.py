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

    ``b_0``, ``b_1``, ``log_M`` are scalars (or shape-() arrays); ``L`` is a
    static int.  Returns two shape-(L,) arrays.
    """
    stages = jnp.arange(L, dtype=jnp.float64)
    p_s = norm.cdf(b_0 + b_1 * stages)
    # Pin away from 0/1 to keep α_s, β_s strictly positive.
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
