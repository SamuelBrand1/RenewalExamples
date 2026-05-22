"""Rolling-origin forecast evaluation: fit → forecast → advance → update → repeat.

This is the harness that operationalises the SMC story: each "origin" T_i
ingests one more week of data via a sequential update (NOT a refit), forecasts
1..h_max days ahead, and we score against held-out truth.
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp
import jax.random as jr
from jax import Array

from smc_renewal.config import ModelConfig
from smc_renewal.forecast import forecast_from_cloud
from smc_renewal.pf.runner import run_liu_west
from smc_renewal.pf.update import _slice_final, extend_liu_west
from smc_renewal.scoring import crps_weighted_batch, interval_coverage


class RollingOriginResult(NamedTuple):
    """Per-origin forecast skill metrics.

    Shapes (n_origins, h_max).
    """

    crps: Array
    cov50: Array
    cov95: Array


def rolling_origin_forecast(
    key: Array,
    cfg: ModelConfig,
    y: Array,
    initial_T0: int,
    step_size: int,
    h_max: int,
    n_particles: int = 2000,
    h: float | None = None,
) -> RollingOriginResult:
    """Run a rolling-origin evaluation of the Liu-West PF.

    First origin fits the model on ``y[:initial_T0]`` from scratch, forecasts
    ``h_max`` days, and scores against ``y[initial_T0:initial_T0+h_max]``.
    Subsequent origins extend the particle cloud by ``step_size`` days each,
    forecast and score again.  Returns CRPS and 50%/95% interval coverage per
    origin per horizon.

    Note: this loop is in Python (not ``jax.lax.scan``) because each origin
    produces a variable-length result and the sequential updates are independent
    calls.  Each inner call is itself fully jit-compiled.
    """
    T = y.shape[0]
    origins = []
    t = initial_T0
    while t + h_max <= T:
        origins.append(t)
        t += step_size
    if not origins:
        raise ValueError("series too short for the requested rolling-origin schedule")

    key, sub_init, sub_fcast = jr.split(key, 3)
    # First origin: fit from scratch.
    first_origin = origins[0]
    fit = run_liu_west(
        sub_init, cfg, y[:first_origin], n_particles=n_particles, h=h
    )
    particles, params, lw = _slice_final(fit)

    crps_rows = []
    cov50_rows = []
    cov95_rows = []

    def score_one(particles, params, lw, t_obs, fcast_key):
        f = forecast_from_cloud(fcast_key, cfg, particles, params, lw, h_max)
        y_true = y[t_obs : t_obs + h_max]
        crps = crps_weighted_batch(y_true, f.y, lw)
        c50 = interval_coverage(y_true, f.y, lw, level=0.5)
        c95 = interval_coverage(y_true, f.y, lw, level=0.95)
        return crps, c50, c95

    crps, c50, c95 = score_one(particles, params, lw, first_origin, sub_fcast)
    crps_rows.append(crps)
    cov50_rows.append(c50)
    cov95_rows.append(c95)

    # Subsequent origins: extend.
    for next_origin in origins[1:]:
        prev_origin = next_origin - step_size
        key, k_upd, k_fc = jr.split(key, 3)
        new_window = y[prev_origin:next_origin]
        ext = extend_liu_west(
            k_upd, cfg, new_window, particles, params, lw, h=h
        )
        particles, params, lw = _slice_final(ext)
        crps, c50, c95 = score_one(particles, params, lw, next_origin, k_fc)
        crps_rows.append(crps)
        cov50_rows.append(c50)
        cov95_rows.append(c95)

    return RollingOriginResult(
        crps=jnp.stack(crps_rows, axis=0),
        cov50=jnp.stack(cov50_rows, axis=0),
        cov95=jnp.stack(cov95_rows, axis=0),
    )


def horizon_summaries(result: RollingOriginResult, horizons: list[int]) -> dict:
    """Average CRPS and coverage at the requested horizons (in days, 1-indexed).

    Returns a dict with per-horizon mean CRPS and mean coverage of 50%/95% intervals.
    """
    out = {}
    for h_days in horizons:
        i = h_days - 1
        out[h_days] = {
            "crps_mean": float(jnp.mean(result.crps[:, i])),
            "cov50_mean": float(jnp.mean(result.cov50[:, i])),
            "cov95_mean": float(jnp.mean(result.cov95[:, i])),
        }
    return out
