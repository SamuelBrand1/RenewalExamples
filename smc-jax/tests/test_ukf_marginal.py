"""Sanity-check the UKF marginal log-lik on a 1D linear-Gaussian sub-case.

For a strictly linear-Gaussian SSM the UKF should agree with the exact Kalman
recursion.  We test by running a custom UKF over

    x_{t+1} = a * x_t + w_t,    w_t ~ N(0, q^2)
    y_t    = x_t + v_t,         v_t ~ N(0, r^2)

with known a, q, r and a series of observations; we compare its log-lik against
the standard Kalman recursion implemented inline.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import jax.random as jr

from smc_renewal.smc2.ukf import UKFParams, _sigma_points


def _ukf_logZ_1d_linear(a, q, r, y, m0, P0, ukf):
    """Run a UKF on the 1-D linear model and return cumulative log-lik."""
    def f(x):
        return a * x

    def h(x):
        return x

    def body(carry, y_t):
        m, P = carry
        chi, Wm, Wc = _sigma_points(jnp.asarray([m]), jnp.asarray([[P]]), ukf)
        chi_pred = jax.vmap(lambda x: f(x[0]))(chi)
        m_pred = jnp.sum(Wm * chi_pred)
        diff = chi_pred - m_pred
        P_pred = jnp.sum(Wc * diff * diff) + q * q

        chi2, Wm2, Wc2 = _sigma_points(jnp.asarray([m_pred]), jnp.asarray([[P_pred]]), ukf)
        y_pts = jax.vmap(lambda x: h(x[0]))(chi2)
        y_hat = jnp.sum(Wm2 * y_pts)
        diff_y = y_pts - y_hat
        S = jnp.sum(Wc2 * diff_y * diff_y) + r * r
        diff_x = chi2[:, 0] - m_pred
        C = jnp.sum(Wc2 * diff_x * diff_y)
        K = C / S
        innov = y_t - y_hat
        m_new = m_pred + K * innov
        P_new = P_pred - K * K * S
        ll = -0.5 * (jnp.log(2 * jnp.pi * S) + innov * innov / S)
        return (m_new, P_new), ll

    _, lls = jax.lax.scan(body, (m0, P0), y)
    return jnp.sum(lls)


def _exact_kalman_logZ_1d_linear(a, q, r, y, m0, P0):
    """Exact 1-D Kalman cumulative log-lik."""
    def body(carry, y_t):
        m, P = carry
        m_pred = a * m
        P_pred = a * a * P + q * q
        S = P_pred + r * r
        K = P_pred / S
        innov = y_t - m_pred
        m_new = m_pred + K * innov
        P_new = (1 - K) * P_pred
        ll = -0.5 * (jnp.log(2 * jnp.pi * S) + innov * innov / S)
        return (m_new, P_new), ll

    _, lls = jax.lax.scan(body, (m0, P0), y)
    return jnp.sum(lls)


def test_ukf_marginal_matches_kalman_1d():
    a, q, r = 0.9, 0.3, 0.5
    T = 50
    key = jr.key(0)
    k1, k2 = jr.split(key, 2)
    # Generate a 1-D AR(1)+noise series.
    x = [0.0]
    for _ in range(T):
        x.append(a * x[-1] + q * float(jr.normal(jr.fold_in(k1, _))))
    x_arr = jnp.array(x[1:])
    y = x_arr + r * jr.normal(k2, (T,))

    m0, P0 = 0.0, 1.0
    ukf = UKFParams()
    lkf_ll = float(_exact_kalman_logZ_1d_linear(a, q, r, y, m0, P0))
    ukf_ll = float(_ukf_logZ_1d_linear(a, q, r, y, m0, P0, ukf))
    rel = abs(lkf_ll - ukf_ll) / max(abs(lkf_ll), 1.0)
    assert rel < 1e-6, f"UKF logZ = {ukf_ll:.6f} vs Kalman = {lkf_ll:.6f} (rel = {rel:.2e})"
