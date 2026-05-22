"""Tests for the GDM observation utilities."""

from __future__ import annotations

import jax
jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import scipy.stats
from numpyro.distributions import BetaBinomial

from smc_renewal.observation_gdm import (
    betabinom_loglik,
    binomial_loglik,
    gdm_beta_params,
    gdm_hypergeom_logweight,
    log_beta_fn,
    log_binom_coef,
    wallenius_aux_logweight,
)


def test_log_beta_fn_sanity():
    assert float(log_beta_fn(jnp.asarray(1.0), jnp.asarray(1.0))) == 0.0


def test_log_binom_coef_matches_scipy():
    for n, k in [(10, 3), (20, 7), (100, 50)]:
        got = float(log_binom_coef(jnp.asarray(float(n)), jnp.asarray(float(k))))
        want = float(np.log(scipy.special.comb(n, k)))
        assert abs(got - want) < 1e-6, f"{n}C{k}: got {got}, want {want}"


def test_gdm_beta_params_uniform_case():
    """``b_0=0, b_1=0, log_M=log(20)`` → α_s=β_s=10 for stages 1..L−1; stage 0
    is forced to p_0 ≈ 0 (minimum reporting delay = 1 day)."""
    alpha_s, beta_s = gdm_beta_params(
        jnp.asarray(0.0), jnp.asarray(0.0), jnp.log(jnp.asarray(20.0)), 14
    )
    # Stage 0: p_0 ≈ 0 → α_0 ≈ 0, β_0 ≈ 20
    assert float(alpha_s[0]) < 1e-3, f"stage-0 α should be near zero, got {float(alpha_s[0])}"
    assert abs(float(beta_s[0]) - 20.0) < 1e-3, f"stage-0 β should be ≈ 20, got {float(beta_s[0])}"
    # Stages 1..L-1: α = β = 10
    assert jnp.allclose(alpha_s[1:], 10.0, atol=1e-6)
    assert jnp.allclose(beta_s[1:], 10.0, atol=1e-6)


def test_gdm_beta_params_min_delay_one():
    """For any (b_0, b_1, log_M), stage 0 always has p ≈ 0."""
    alpha_s, beta_s = gdm_beta_params(
        jnp.asarray(2.0), jnp.asarray(0.5), jnp.log(jnp.asarray(50.0)), 14
    )
    p_0 = float(alpha_s[0] / (alpha_s[0] + beta_s[0]))
    assert p_0 < 1e-4, f"min delay = 1 day requires p_0 ≈ 0, got {p_0}"


def test_gdm_beta_params_monotone_in_b1():
    """Positive b_1 ⇒ p_s monotonically non-decreasing in s (saturates near 1
    at the high end due to the 1−1e-6 clip)."""
    alpha_s, beta_s = gdm_beta_params(
        jnp.asarray(0.0), jnp.asarray(0.4), jnp.log(jnp.asarray(30.0)), 14
    )
    p_s = alpha_s / (alpha_s + beta_s)
    assert jnp.all(jnp.diff(p_s) >= -1e-9), "p_s should be non-decreasing with b_1 > 0"


def test_betabinom_loglik_matches_numpyro():
    """``betabinom_loglik`` agrees with numpyro's reference implementation."""
    n = 12
    alpha = 2.5
    beta = 3.7
    nd = BetaBinomial(jnp.asarray(alpha), jnp.asarray(beta), jnp.asarray(n))
    for x in (0, 1, 4, 7, 12):
        got = float(betabinom_loglik(
            jnp.asarray(float(x)), jnp.asarray(float(n)),
            jnp.asarray(alpha), jnp.asarray(beta),
        ))
        want = float(nd.log_prob(jnp.asarray(x)))
        assert abs(got - want) < 1e-6, f"x={x}: got {got}, want {want}"


def test_gdm_hypergeom_logweight_is_finite():
    """For a feasible partition, the IS log-weight is finite."""
    O = jnp.asarray([3, 2, 1, 0, 0], dtype=jnp.int64)
    U = jnp.asarray([5, 4, 3, 2, 1], dtype=jnp.int64)
    alpha_s = jnp.asarray([2.0, 3.0, 4.0, 5.0, 6.0])
    beta_s = jnp.asarray([3.0, 3.0, 3.0, 3.0, 3.0])
    val = float(gdm_hypergeom_logweight(O, U, alpha_s, beta_s))
    assert np.isfinite(val), f"Expected finite log-weight, got {val}"


def test_binomial_loglik_matches_scipy():
    """``binomial_loglik`` matches scipy's Binomial pmf."""
    n = 12
    p = 0.4
    for x in (0, 1, 4, 7, 12):
        got = float(binomial_loglik(
            jnp.asarray(float(x)), jnp.asarray(float(n)), jnp.asarray(p)
        ))
        want = float(scipy.stats.binom.logpmf(x, n, p))
        assert abs(got - want) < 1e-6, f"x={x}: got {got}, want {want}"


def test_wallenius_aux_logweight_is_finite():
    """Auxiliary-q IS log-weight is finite for feasible inputs."""
    O = jnp.asarray([3, 2, 1, 0, 0], dtype=jnp.int64)
    U = jnp.asarray([5, 4, 3, 2, 1], dtype=jnp.int64)
    q_s = jnp.asarray([0.3, 0.5, 0.4, 0.6, 0.2])
    log_q_ord = jnp.asarray(-5.0)
    val = float(wallenius_aux_logweight(O, U, q_s, log_q_ord))
    assert np.isfinite(val), f"Expected finite log-weight, got {val}"
