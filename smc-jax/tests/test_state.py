"""Round-trip tests for the ParticleState <-> flat-vector pack/unpack."""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr

from smc_renewal.config import default_config
from smc_renewal.state import (
    ParticleState,
    pack,
    state_dim,
    unpack,
)


def _random_state(key, buffer_len):
    keys = jr.split(key, 6)
    return ParticleState(
        log_Rt=jr.normal(keys[0]),
        log_sigma_R=jr.normal(keys[1]),
        log_F=jr.normal(keys[2]),
        log_sigma_F=jr.normal(keys[3]),
        log_I0=jr.normal(keys[4]),
        I_buf=jr.uniform(keys[5], (buffer_len,)),
    )


def test_pack_unpack_round_trip():
    cfg = default_config()
    state = _random_state(jr.key(0), cfg.buffer_len)
    flat = pack(state)
    assert flat.shape == (state_dim(cfg.buffer_len),)
    rec = unpack(flat, cfg.buffer_len)
    assert jnp.allclose(state.log_Rt, rec.log_Rt)
    assert jnp.allclose(state.log_sigma_R, rec.log_sigma_R)
    assert jnp.allclose(state.log_F, rec.log_F)
    assert jnp.allclose(state.log_sigma_F, rec.log_sigma_F)
    assert jnp.allclose(state.log_I0, rec.log_I0)
    assert jnp.allclose(state.I_buf, rec.I_buf)


def test_pmfs_normalize_and_have_right_length():
    cfg = default_config()
    for pmf, length in [
        (cfg.generation_interval, cfg.Tg),
        (cfg.delay_pmf, cfg.Td),
    ]:
        assert pmf.shape == (length,)
        assert jnp.allclose(jnp.sum(pmf), 1.0, atol=1e-6)
        assert bool(jnp.all(pmf >= 0))


def test_buffer_len_is_max_of_two():
    cfg = default_config()
    assert cfg.buffer_len == max(cfg.Tg, cfg.Td)
