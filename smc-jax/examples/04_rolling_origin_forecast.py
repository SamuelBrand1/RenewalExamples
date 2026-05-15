"""04 — Rolling-origin forecast evaluation.

The headline operational demo: fit on the first chunk of data, forecast 28
days ahead, *sequentially* ingest the next week (no refit), forecast again,
repeat.  Visualise the forecast fans at a few origins and the resulting
CRPS / interval-coverage curves vs horizon.
"""

from __future__ import annotations

import _common  # noqa: F401
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np

from smc_renewal.config import default_config
from smc_renewal.forecast import forecast_from_cloud
from smc_renewal.pf.runner import run_liu_west
from smc_renewal.pf.update import _slice_final, extend_liu_west
from smc_renewal.rolling_origin import (
    horizon_summaries,
    rolling_origin_forecast,
)
from smc_renewal.scoring import weighted_quantile
from smc_renewal.state import ParticleParams
from smc_renewal.synthetic import simulate


def main():
    _common.set_clean_style()
    cfg = default_config()
    T = 180
    H_MAX = 28
    INITIAL_T0 = 80
    STEP = 14
    N = 2500

    truth = ParticleParams(
        log_tau_R=jnp.asarray(-4.0),
        log_tau_F=jnp.asarray(-12.0),
        log_phi=jnp.asarray(2.5),
    )
    ds = simulate(jr.key(2), cfg, T, params=truth)

    # --- rolling-origin skill curves ---
    print("running rolling-origin forecast harness…")
    eval_result = rolling_origin_forecast(
        jr.key(101),
        cfg, ds.y,
        initial_T0=INITIAL_T0, step_size=STEP, h_max=H_MAX,
        n_particles=N, h=0.1,
    )
    summary = horizon_summaries(eval_result, horizons=[7, 14, 21, 28])
    print("horizon-wise summary:")
    for h, s in summary.items():
        print(f"  {h:>3}d  CRPS={s['crps_mean']:>10.2f}  cov50={s['cov50_mean']:.2f}  cov95={s['cov95_mean']:.2f}")

    # --- generate forecast fans at 3 chosen origins for the visual ---
    # Place origins so the last one's horizon still falls inside the series.
    last_origin = T - H_MAX
    origins_to_plot = [INITIAL_T0, (INITIAL_T0 + last_origin) // 2, last_origin]
    fans = []

    fit = run_liu_west(jr.key(11), cfg, ds.y[:origins_to_plot[0]], n_particles=N, h=0.1)
    particles, params, lw = _slice_final(fit)
    fcast = forecast_from_cloud(jr.key(31), cfg, particles, params, lw, H_MAX)
    fans.append((origins_to_plot[0], fcast))

    for i in range(1, len(origins_to_plot)):
        new_data = ds.y[origins_to_plot[i - 1] : origins_to_plot[i]]
        ext = extend_liu_west(jr.key(40 + i), cfg, new_data, particles, params, lw, h=0.1)
        particles, params, lw = _slice_final(ext)
        fcast = forecast_from_cloud(jr.key(31 + i), cfg, particles, params, lw, H_MAX)
        fans.append((origins_to_plot[i], fcast))

    # --- plotting ---
    fig = plt.figure(figsize=(12, 7.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[2.2, 1.0], hspace=0.4, wspace=0.3)

    for idx, (origin, fcast) in enumerate(fans):
        ax = fig.add_subplot(gs[0, idx])
        tt = np.arange(T)
        ax.plot(tt, ds.y, color="k", lw=0.8, alpha=0.5, label="observed cases (full)")
        ax.scatter(tt[:origin], np.asarray(ds.y[:origin]), s=8, color="k", alpha=0.6)
        # Quantile bands across particles for each horizon day.
        h_t = origin + 1 + np.arange(H_MAX)
        qs = (0.025, 0.25, 0.5, 0.75, 0.975)
        Q = jnp.stack(
            [
                jax.vmap(weighted_quantile, in_axes=(0, None, None))(
                    fcast.y, fcast.log_weights, jnp.asarray(q)
                )
                for q in qs
            ]
        )
        ax.fill_between(h_t, Q[0], Q[4], color="C0", alpha=0.15, label="95% PI")
        ax.fill_between(h_t, Q[1], Q[3], color="C0", alpha=0.30, label="50% PI")
        ax.plot(h_t, Q[2], color="C0", lw=1.2, label="median")
        ax.axvline(origin, color="r", lw=0.7, ls="--")
        ax.set_title(f"forecast from day {origin}")
        ax.set_xlabel("day"); ax.set_ylabel("cases")
        if idx == 0:
            ax.legend(loc="upper left", frameon=False, fontsize=8)

    # Skill curves.
    horizons = np.arange(1, H_MAX + 1)
    crps_mean = np.asarray(eval_result.crps.mean(axis=0))
    cov50_mean = np.asarray(eval_result.cov50.mean(axis=0))
    cov95_mean = np.asarray(eval_result.cov95.mean(axis=0))

    ax = fig.add_subplot(gs[1, 0])
    # CRPS is dominated by tail blowups at long horizons; show on log scale, clip floor.
    ax.semilogy(horizons, np.maximum(crps_mean, 1e-2), color="C0")
    ax.set_title("Mean CRPS vs horizon")
    ax.set_xlabel("horizon (days)"); ax.set_ylabel("CRPS (log)")

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(horizons, cov50_mean, color="C2", label="50% PI")
    ax.axhline(0.5, color="C2", lw=0.5, ls="--")
    ax.plot(horizons, cov95_mean, color="C0", label="95% PI")
    ax.axhline(0.95, color="C0", lw=0.5, ls="--")
    ax.set_ylim(0, 1.05)
    ax.set_title("Interval coverage vs horizon")
    ax.set_xlabel("horizon (days)"); ax.set_ylabel("coverage")
    ax.legend(loc="lower right", frameon=False, fontsize=8)

    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    table_text = "horizon  CRPS    cov50  cov95\n" + "\n".join(
        f"  {h:>3}d  {summary[h]['crps_mean']:>8.2f}  {summary[h]['cov50_mean']:.2f}   {summary[h]['cov95_mean']:.2f}"
        for h in (7, 14, 21, 28)
    )
    ax.text(
        0.0, 0.95, "headline horizons", fontsize=10, fontweight="bold",
        family="monospace", transform=ax.transAxes, va="top",
    )
    ax.text(
        0.0, 0.80, table_text, fontsize=9, family="monospace",
        transform=ax.transAxes, va="top",
    )

    fig.suptitle(
        "Rolling-origin forecast — Liu-West PF with sequential weekly updates", y=1.02,
    )
    out = _common.save(fig, "04_rolling_origin_forecast.png")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
