"""Tests for the Model E sampling primitives and proposal."""

from __future__ import annotations

import jax
jax.config.update("jax_enable_x64", True)

import dataclasses

import jax.numpy as jnp
import jax.random as jr
import numpy as np
import scipy.stats

from smc_renewal.config import default_config
from smc_renewal.pf.model_gdm import (
    ParticleParamsGDM,
    ParticleStateGDM,
    forward_step,
    propose_gdm_partition,
    propose_step,
    sample_mv_hypergeom,
    sample_univar_hypergeom,
    sample_wallenius,
)


def test_univar_hypergeom_mean():
    """Empirical mean ≈ theoretical n·K/N."""
    key = jr.key(0)
    N, K, n = 50, 20, 15
    keys = jr.split(key, 5000)
    samples = jax.vmap(
        lambda k: sample_univar_hypergeom(
            k, jnp.asarray(N), jnp.asarray(K), jnp.asarray(n)
        )
    )(keys)
    theoretical = n * K / N
    empirical = float(samples.mean())
    assert abs(empirical - theoretical) < 0.1, (
        f"Hypergeom mean: empirical {empirical}, theoretical {theoretical}"
    )


def test_univar_hypergeom_distribution():
    """Sample histogram matches scipy's hypergeometric pmf."""
    key = jr.key(1)
    N, K, n = 30, 12, 10
    keys = jr.split(key, 20000)
    samples = jax.vmap(
        lambda k: sample_univar_hypergeom(
            k, jnp.asarray(N), jnp.asarray(K), jnp.asarray(n)
        )
    )(keys)
    samples_np = np.asarray(samples)
    support = np.arange(0, min(K, n) + 1)
    empirical_pmf = np.array([(samples_np == k).mean() for k in support])
    theoretical_pmf = scipy.stats.hypergeom.pmf(support, N, K, n)
    assert np.allclose(empirical_pmf, theoretical_pmf, atol=0.01), (
        f"empirical: {empirical_pmf}\ntheoretical: {theoretical_pmf}"
    )


def test_mv_hypergeom_sum_and_caps():
    """``Σ O = y_t`` and ``O_s ≤ U[s]`` for every draw."""
    key = jr.key(2)
    U = jnp.asarray([10, 15, 8, 12, 5], dtype=jnp.int64)
    y_t = jnp.asarray(20, dtype=jnp.int64)
    keys = jr.split(key, 1000)
    samples = jax.vmap(
        lambda k: sample_mv_hypergeom(k, U, y_t, 5)
    )(keys)
    sums = samples.sum(axis=1)
    assert jnp.all(sums == y_t), "MV-Hypergeom samples must sum to y_t"
    assert jnp.all(samples <= U[None, :]), "MV-Hypergeom samples must respect caps"


def test_wallenius_sum_and_caps():
    """Wallenius sampler also respects sum and caps."""
    key = jr.key(3)
    U = jnp.asarray([10, 15, 8, 12, 5], dtype=jnp.int64)
    weights = jnp.asarray([0.3, 0.5, 0.7, 0.4, 0.6])
    y_t = jnp.asarray(20, dtype=jnp.int64)
    keys = jr.split(key, 200)
    out = jax.vmap(
        lambda k: sample_wallenius(k, U, weights, y_t, 5)
    )(keys)
    samples = out[0]
    sums = samples.sum(axis=1)
    assert jnp.all(sums == y_t), "Wallenius samples must sum to y_t"
    assert jnp.all(samples <= U[None, :]), "Wallenius samples must respect caps"


def test_propose_gdm_partition_infeasible_returns_neginf():
    """When y_t > Σ U, the log-weight is −∞."""
    cfg = dataclasses.replace(default_config(), ascertainment_alpha=0.9)
    L = cfg.buffer_len
    U_buf_prev = jnp.zeros(L, dtype=jnp.int64)
    E_t = jnp.asarray(2, dtype=jnp.int64)
    y_t = jnp.asarray(50, dtype=jnp.int64)  # way too big given Σ U = 2
    O, log_w = propose_gdm_partition(
        jr.key(0), U_buf_prev, E_t, y_t,
        jnp.asarray(0.0), jnp.asarray(0.4), jnp.asarray(3.0), cfg,
    )
    assert jnp.all(O == 0), "Infeasible: O should be zeros"
    assert not jnp.isfinite(log_w), f"Infeasible: expected -inf, got {log_w}"


def test_propose_step_runs_end_to_end():
    """Smoke test: ``propose_step`` returns a valid state + finite log-weight
    on a typical setup."""
    cfg = dataclasses.replace(default_config(), ascertainment_alpha=0.9)
    L = cfg.buffer_len
    state = ParticleStateGDM(
        log_Rt=jnp.asarray(0.5),
        v_R=jnp.asarray(0.0),
        log_I0=jnp.asarray(1.0),
        U_buf=jnp.full((L,), 3, dtype=jnp.int64),
    )
    theta = ParticleParamsGDM(
        log_sigma_vR=jnp.asarray(-6.0),
        b_0=jnp.asarray(0.2),
        b_1=jnp.asarray(0.4),
        log_M=jnp.asarray(3.5),
    )
    y_t = jnp.asarray(5)
    new_state, log_w = propose_step(jr.key(0), state, theta, y_t, cfg)
    assert isinstance(new_state, ParticleStateGDM)
    assert new_state.U_buf.shape == (L,)
    assert np.isfinite(float(log_w)), f"Expected finite log-weight, got {log_w}"
    # Buffer should be non-negative integer-valued.
    assert jnp.all(new_state.U_buf >= 0)


def test_forward_step_produces_valid_partition():
    """``forward_step`` (truth simulator) produces consistent state."""
    cfg = dataclasses.replace(default_config(), ascertainment_alpha=0.9)
    L = cfg.buffer_len
    state = ParticleStateGDM(
        log_Rt=jnp.asarray(0.5),
        v_R=jnp.asarray(0.0),
        log_I0=jnp.asarray(1.0),
        U_buf=jnp.full((L,), 3, dtype=jnp.int64),
    )
    theta = ParticleParamsGDM(
        log_sigma_vR=jnp.asarray(-6.0),
        b_0=jnp.asarray(0.2),
        b_1=jnp.asarray(0.4),
        log_M=jnp.asarray(3.5),
    )
    new_state, y, O = forward_step(jr.key(0), state, theta, cfg)
    assert O.shape == (L,)
    assert int(jnp.sum(O)) == int(y), "Σ O should equal y_t"
    assert jnp.all(O >= 0), "All partition counts non-negative"
