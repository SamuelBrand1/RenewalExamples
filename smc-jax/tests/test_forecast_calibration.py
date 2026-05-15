"""End-to-end: rolling-origin Liu-West forecasts on synthetic data are calibrated.

Success criterion: averaged across origins and 1/2/3/4-week horizons, the
95% interval covers ground truth on at least 80% of forecast windows
(below the nominal 95% because bootstrap PFs are systematically over-confident,
but should be in the ballpark).
"""

from __future__ import annotations

import jax.numpy as jnp
import jax.random as jr

from smc_renewal.config import default_config
from smc_renewal.rolling_origin import (
    horizon_summaries,
    rolling_origin_forecast,
)
from smc_renewal.state import ParticleParams
from smc_renewal.synthetic import simulate


def test_rolling_origin_forecast_is_calibrated():
    cfg = default_config()
    T = 140
    truth = ParticleParams(
        log_tau_R=jnp.asarray(-4.0),
        log_tau_F=jnp.asarray(-12.0),
        log_phi=jnp.asarray(2.5),
    )
    ds = simulate(jr.key(2), cfg, T, params=truth)
    assert bool(jnp.isfinite(ds.mu_y).all()), "synthetic exploded; retry seed"

    result = rolling_origin_forecast(
        jr.key(7),
        cfg,
        ds.y,
        initial_T0=60,
        step_size=7,
        h_max=28,
        n_particles=1500,
        h=0.1,
    )
    summaries = horizon_summaries(result, horizons=[7, 14, 21, 28])

    # Diagnostic for the test log.
    for hdays, s in summaries.items():
        print(
            f"horizon={hdays}d  CRPS={s['crps_mean']:.2f} "
            f"cov50={s['cov50_mean']:.2f} cov95={s['cov95_mean']:.2f}"
        )

    # Aggregate coverage across the 4 headline horizons.
    mean_cov95 = sum(summaries[h]["cov95_mean"] for h in summaries) / len(summaries)
    assert mean_cov95 >= 0.80, (
        f"mean 95% coverage across horizons = {mean_cov95:.3f} (< 0.80)"
    )
