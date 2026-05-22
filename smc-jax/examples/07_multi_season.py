"""07 — Multi-season demo: nested-RW renewal, two doubly-stochastic processes.

Both log Rt(t) and log F(t) follow the same nested-RW structure (a random
walk driven by an inner volatility that is itself a random walk), but with
**deliberately separated timescales**:

  - **log Rt(t)** varies fast: seasonal swings of ±0.45 within ~120 days.
  - **log F(t)** varies slow: strain-turnover timescale, drifts modestly
    over ~720 days.  F = exp(log_F) > 0 so feedback always damps:
    ``Rt_eff = Rt_raw · exp(-F · conv(I, g))``.

The two are **partially confounded** — both enter μ_y(t) — so on any given
short window the data can't fully tell them apart.  The timescale gap is
what lets the data separate them *most* of the time.  This is where SMC
earns its keep over a point-estimator: the joint particle cloud preserves
the slight-unidentifiability *correlation* in the posterior, rather than
collapsing to a single mode that hides it.

Truth construction: log Rt(t) is **forced** to a seasonal trajectory; log_F(t)
evolves naturally via the nested-RW dynamics with a chosen truth ``log_τ_F``.

The inference task: recover the seasonal Rt swings through time-varying
σ_R(t), recover the slow log_F drift through σ_F(t), and learn something
about τ_F.  The τ_R posterior is harder to identify because Rt is forced
(so its data-generating τ_R isn't really meaningful).

The inference priors are **deliberately wrong** — centered 1-2 prior-SDs
away from truth — to make a methodological point: Liu-West has the prior
**only in the initial-particle sampling**, NOT in the per-step weight
update (unlike PMMH).  So as long as the prior is wide enough to include
truth in its tail, the posterior drifts to data-driven values regardless
of where the prior is centered.  We demonstrate that here.

The companion forecast figure also makes a second, less flattering point:
**this is a bad model for the data**.  The inference model has no seasonal
term, just a random walk on log Rt.  In-sample tracking works fine
(filter and smoother bands cover the truth), but multi-month forecasts
across seasonal transitions are systematically biased — the model
extrapolates current Rt forward and so misses the seasonal trough
entirely.  This is a clean demonstration of model misspecification: the
SMC inference is doing the best it can, but the model lacks the structure
needed to forecast across seasons.  A production pipeline would extend
the latent process (e.g. add a periodic component) to fix this.
"""

from __future__ import annotations

import dataclasses

import _common  # noqa: F401
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import numpyro.distributions as dist
from pfjax import particle_smooth

from smc_renewal.config import default_config
from smc_renewal.forecast import forecast_from_cloud
from smc_renewal.pf.bootstrap import weighted_quantile
from smc_renewal.pf.runner import run_liu_west
from smc_renewal.state import ParticleParams
from smc_renewal.synthetic import sample_initial_state
from smc_renewal.transition import (
    TransitionNoise,
    expected_observation,
    step,
)


def simulate_seasonal_with_natural_F(
    key, cfg, log_Rt_seq, truth_params: ParticleParams,
    truth_log_F_init: float, truth_log_sigma_F_init: float,
    season_log_I0: float = 2.3,
):
    """Forward-simulate the model with log_Rt(t) forced and log_F(t) drawn naturally."""
    T = log_Rt_seq.shape[0]
    k_init, k_noise, k_obs = jr.split(key, 3)
    initial_state = sample_initial_state(k_init, cfg)
    initial_state = initial_state._replace(
        log_I0=jnp.asarray(season_log_I0),
        I_buf=jnp.full((cfg.buffer_len,), jnp.exp(season_log_I0)),
        log_F=jnp.asarray(truth_log_F_init),
        log_sigma_F=jnp.asarray(truth_log_sigma_F_init),
    )
    all_noise = jr.normal(k_noise, (T, 4))

    def body(state, inputs):
        log_Rt_t, noise_row = inputs
        # Force log_Rt; let log_F / log_σ_F walk naturally (zero noise on the
        # Rt-side dims because Rt is forced).
        forced = state._replace(log_Rt=log_Rt_t)
        noise = TransitionNoise(
            eps_R=jnp.asarray(0.0),
            eta_R=jnp.asarray(0.0),
            eps_F=noise_row[2],
            eta_F=noise_row[3],
        )
        new_state = step(forced, truth_params, noise, cfg)
        mu_y = expected_observation(new_state, cfg)
        return new_state, (new_state, mu_y)

    _, (traj, mu_y) = jax.lax.scan(body, initial_state, (log_Rt_seq, all_noise))
    phi = jnp.exp(truth_params.log_phi)
    y = dist.NegativeBinomial2(mean=mu_y, concentration=phi).sample(k_obs)
    return traj, mu_y, y.astype(jnp.float64)


def _wmean(x, lw):
    w = jnp.exp(lw - jnp.max(lw))
    w = w / w.sum()
    return jnp.sum(w * x)


def main():
    _common.set_clean_style()
    # Truth parameters and initial state for the F-dynamics.  Bigger τ values
    # than the prior defaults so σ_R and σ_F have visible drift across a
    # ~4-year horizon.
    #   τ_R = exp(-3) ≈ 0.050  → log_σ_R walks by ~1.9 over T=1440, so σ_R
    #     can vary by exp(±1) ≈ 3× across the series.
    #   τ_F = exp(-5) ≈ 6.7e-3 → log_σ_F walks by ~0.25 over T=1440, so σ_F
    #     drifts modestly across the series.
    #   F ≈ exp(-8) = 3.4e-4 at start; with σ_F ≈ 6.7e-3, log_F walks
    #     ~0.25 over T=1440, so F can range ~2.6e-4 to 4.4e-4.
    truth_params = ParticleParams(
        log_tau_R=jnp.asarray(-3.0),
        log_tau_F=jnp.asarray(-7.0),
        log_phi=jnp.asarray(2.5),
    )
    truth_log_F_init = -8.0
    truth_log_sigma_F_init = -5.0

    # Inference priors **deliberately wrong** — centered ~1-2 prior-SDs away
    # from truth for both log_F dynamics and the τ hyperparameters.  Liu-West's
    # only contact with the prior is sampling the *initial* particle cloud
    # (there is no prior in the per-step weight update, unlike PMMH).  So if
    # the prior is wide enough to include the truth-region, the cloud should
    # drift to it via data+shrink-jitter regardless of prior centering.  This
    # is the right kind of robustness for a method that gets "applied to
    # things we don't know in advance".
    #
    # ``sigma_floor`` is lowered so it doesn't clamp the truth's σ_F upward.
    cfg = dataclasses.replace(
        default_config(),
        sigma_floor=1e-6,
        init_log_F_mean=-10.0,            # truth is -8.0 → off by ~1 prior-SD
        init_log_F_sd=2.0,                # wide
        init_log_sigma_F_mean=-8.0,       # truth is -5.0 → off by ~1.5 prior-SDs
        init_log_sigma_F_sd=2.0,          # wide
        prior_log_tau_R_mean=-4.5,        # truth is -3.0 → off by ~1.5 prior-SDs
        prior_log_tau_R_sd=1.0,           # wide
        prior_log_tau_F_mean=-8.0,        # truth is -5.0 → off by ~2 prior-SDs
        prior_log_tau_F_sd=1.5,           # wide
    )

    # --- seasonal truth: 4 annual seasons (T=1440 days = ~4 years) ---
    season_len = 360
    n_seasons = 4
    T = season_len * n_seasons
    t = jnp.arange(T)
    # Baseline ≈ E[F·conv(I,g)] so infections stay roughly stationary rather
    # than decaying.  With F ~ 3.4e-4 and conv ~ 30-80, F·conv ≈ 0.01-0.03 →
    # baseline = 0.02.
    base = 0.45 * jnp.sin(2 * jnp.pi * t / season_len) + 0.02
    raw = 0.06 * jr.normal(jr.key(33), (T,))
    smoothed = jnp.convolve(raw, jnp.ones(7) / 7, mode="same")
    log_Rt_truth = base + smoothed

    print(f"running multi-season demo: T={T} days, {n_seasons} quasi-seasons")
    print(f"  truth: seasonal log_Rt (imposed) + log_F(t) drawn from nested-RW")
    print(f"  truth log_F init = {truth_log_F_init}, log_σ_F init = {truth_log_sigma_F_init}, "
          f"log_τ_F = {float(truth_params.log_tau_F):+.1f}")
    truth_traj, mu_y_truth, y = simulate_seasonal_with_natural_F(
        jr.key(7), cfg, log_Rt_truth, truth_params,
        truth_log_F_init, truth_log_sigma_F_init,
    )
    log_F_truth = truth_traj.log_F
    log_sigma_F_truth = truth_traj.log_sigma_F

    # --- fit nested-RW model with Liu-West PF ---
    # Steyn-style fixed-lag resampling: when resampling at time t, only
    # permute the last L time-steps of state+param history.  Observations
    # at time t are informative about R at time t-lag (≈ mean delay ~6d +
    # GI ~5d ≈ 14d window), so resampling further back doesn't add
    # information and just induces path degeneracy.
    N = 6000
    h = 0.07
    fixed_lag_L = 21
    print(f"  Liu-West PF: N={N}, h={h}, fixed_lag_L={fixed_lag_L}")
    result = run_liu_west(jr.key(11), cfg, y, n_particles=N, h=h, fixed_lag_L=fixed_lag_L)

    lw = result.log_weights_history
    log_Rt_p = result.particles_history.log_Rt
    log_sigma_R_p = result.particles_history.log_sigma_R
    log_F_p = result.particles_history.log_F
    log_sigma_F_p = result.particles_history.log_sigma_F
    log_tau_R_p = result.params_history.log_tau_R
    log_tau_F_p = result.params_history.log_tau_F

    qs = (0.05, 0.5, 0.95)
    log_Rt_q = jnp.stack(
        [jax.vmap(weighted_quantile)(log_Rt_p, lw, jnp.full((T,), q)) for q in qs]
    )
    log_F_q = jnp.stack(
        [jax.vmap(weighted_quantile)(log_F_p, lw, jnp.full((T,), q)) for q in qs]
    )
    log_sigma_R_q = jnp.stack(
        [jax.vmap(weighted_quantile)(log_sigma_R_p, lw, jnp.full((T,), q)) for q in qs]
    )
    log_sigma_F_q = jnp.stack(
        [jax.vmap(weighted_quantile)(log_sigma_F_p, lw, jnp.full((T,), q)) for q in qs]
    )

    # --- trajectory smoother via pfjax.particle_smooth ---
    n_smooth = 600
    smooth_keys = jr.split(jr.key(99), n_smooth)
    final_lw = lw[-1]

    def _smooth_dim(arr):
        return jax.vmap(
            lambda k: particle_smooth(k, final_lw, arr, result.ancestors)
        )(smooth_keys)

    log_Rt_sm = _smooth_dim(log_Rt_p)
    log_F_sm = _smooth_dim(log_F_p)
    log_sigma_R_sm = _smooth_dim(log_sigma_R_p)
    log_sigma_F_sm = _smooth_dim(log_sigma_F_p)

    log_Rt_qs = jnp.quantile(log_Rt_sm, jnp.array([0.05, 0.5, 0.95]), axis=0)
    log_F_qs = jnp.quantile(log_F_sm, jnp.array([0.05, 0.5, 0.95]), axis=0)
    log_sigma_R_qs = jnp.quantile(log_sigma_R_sm, jnp.array([0.05, 0.5, 0.95]), axis=0)
    log_sigma_F_qs = jnp.quantile(log_sigma_F_sm, jnp.array([0.05, 0.5, 0.95]), axis=0)
    tau_R_q = jnp.stack(
        [jax.vmap(weighted_quantile)(log_tau_R_p, lw, jnp.full((T,), q)) for q in (0.1, 0.5, 0.9)]
    )
    tau_F_q = jnp.stack(
        [jax.vmap(weighted_quantile)(log_tau_F_p, lw, jnp.full((T,), q)) for q in (0.1, 0.5, 0.9)]
    )

    cov_logRt = float(jnp.mean((log_Rt_truth >= log_Rt_q[0]) & (log_Rt_truth <= log_Rt_q[2])))
    cov_logF = float(jnp.mean((log_F_truth >= log_F_q[0]) & (log_F_truth <= log_F_q[2])))
    cov_logRt_s = float(jnp.mean((log_Rt_truth >= log_Rt_qs[0]) & (log_Rt_truth <= log_Rt_qs[2])))
    cov_logF_s = float(jnp.mean((log_F_truth >= log_F_qs[0]) & (log_F_truth <= log_F_qs[2])))
    print(f"  log_Rt 90% filter coverage:   {cov_logRt:.3f}")
    print(f"  log_Rt 90% smoother coverage: {cov_logRt_s:.3f}")
    print(f"  log_F  90% filter coverage:   {cov_logF:.3f}")
    print(f"  log_F  90% smoother coverage: {cov_logF_s:.3f}")
    print(f"  final log_τ_R post mean: {float(_wmean(log_tau_R_p[-1], lw[-1])):+.3f}  "
          f"(truth {float(truth_params.log_tau_R):+.3f})")
    print(f"  final log_τ_F post mean: {float(_wmean(log_tau_F_p[-1], lw[-1])):+.3f}  "
          f"(truth {float(truth_params.log_tau_F):+.3f})")

    # --- 5-panel plot ---
    fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True,
                             gridspec_kw={"hspace": 0.5})
    season_boundaries = [i * season_len for i in range(1, n_seasons)]

    ax = axes[0]
    ax.plot(t, mu_y_truth, color="C0", lw=1.0, label="true μ_y(t)")
    ax.scatter(t, np.asarray(y), s=6, color="k", alpha=0.55, label="observed y")
    for b in season_boundaries:
        ax.axvline(b, color="gray", lw=0.5, ls=":")
    ax.set_ylabel("cases")
    ax.set_title("Synthetic data — six quasi-seasons of seasonal Rt with drifting F")
    ax.legend(loc="upper left", frameon=False, fontsize=9)

    ax = axes[1]
    ax.plot(t, log_Rt_truth, color="k", lw=1.4, label="truth")
    ax.fill_between(t, log_Rt_q[0], log_Rt_q[2], color="C0", alpha=0.18, label="filter 90%")
    ax.plot(t, log_Rt_q[1], color="C0", lw=1.0, label="filter median")
    ax.fill_between(t, log_Rt_qs[0], log_Rt_qs[2], color="C4", alpha=0.20, label="smoother 90%")
    ax.plot(t, log_Rt_qs[1], color="C4", lw=1.0, label="smoother median")
    ax.axhline(0, color="k", lw=0.3)
    for b in season_boundaries:
        ax.axvline(b, color="gray", lw=0.5, ls=":")
    ax.set_ylabel("log Rt")
    ax.set_title("Liu-West tracking of seasonal log Rt(t) — filter vs smoother")
    ax.legend(loc="upper right", frameon=False, fontsize=8, ncol=3)

    ax = axes[2]
    ax.plot(t, log_F_truth, color="k", lw=1.4, label="truth (natural RW)")
    ax.fill_between(t, log_F_q[0], log_F_q[2], color="C3", alpha=0.18, label="filter 90%")
    ax.plot(t, log_F_q[1], color="C3", lw=1.0, label="filter median")
    ax.fill_between(t, log_F_qs[0], log_F_qs[2], color="C4", alpha=0.20, label="smoother 90%")
    ax.plot(t, log_F_qs[1], color="C4", lw=1.0, label="smoother median")
    for b in season_boundaries:
        ax.axvline(b, color="gray", lw=0.5, ls=":")
    ax.set_ylabel("log F(t)")
    ax.set_title(f"Liu-West tracking of log F(t) — drawn from nested-RW with truth log_τ_F={float(truth_params.log_tau_F):+.1f}")
    ax.legend(loc="upper right", frameon=False, fontsize=8, ncol=3)

    ax = axes[3]
    ax.fill_between(t, jnp.exp(log_sigma_R_q[0]), jnp.exp(log_sigma_R_q[2]),
                    color="C1", alpha=0.18, label="σ_R filter 90%")
    ax.plot(t, jnp.exp(log_sigma_R_q[1]), color="C1", lw=1.0, label="σ_R filter med")
    ax.fill_between(t, jnp.exp(log_sigma_R_qs[0]), jnp.exp(log_sigma_R_qs[2]),
                    color="C4", alpha=0.18, label="σ_R smooth 90%")
    ax.plot(t, jnp.exp(log_sigma_R_qs[1]), color="C4", lw=1.0, ls="--", label="σ_R smooth med")
    ax.set_ylabel("σ_R(t)")
    ax.set_yscale("log")
    ax2 = ax.twinx()
    ax2.fill_between(t, jnp.exp(log_sigma_F_q[0]), jnp.exp(log_sigma_F_q[2]),
                     color="C5", alpha=0.18, label="σ_F filter 90%")
    ax2.plot(t, jnp.exp(log_sigma_F_q[1]), color="C5", lw=1.0, label="σ_F filter med")
    ax2.fill_between(t, jnp.exp(log_sigma_F_qs[0]), jnp.exp(log_sigma_F_qs[2]),
                     color="C6", alpha=0.18, label="σ_F smooth 90%")
    ax2.plot(t, jnp.exp(log_sigma_F_qs[1]), color="C6", lw=1.0, ls="--", label="σ_F smooth med")
    ax2.set_ylabel("σ_F(t)")
    ax2.set_yscale("log")
    for b in season_boundaries:
        ax.axvline(b, color="gray", lw=0.5, ls=":")
    ax.set_title("Inferred volatilities σ_R(t) (orange/C4) and σ_F(t) (purple/C6) — filter & smoother, log scales")
    ax.legend(loc="upper left", frameon=False, fontsize=7)
    ax2.legend(loc="upper right", frameon=False, fontsize=7)

    ax = axes[4]
    ax.fill_between(t, tau_R_q[0], tau_R_q[2], color="C2", alpha=0.22, label="log_τ_R 80%")
    ax.plot(t, tau_R_q[1], color="C2", lw=1.0, label="log_τ_R median")
    ax.axhline(float(truth_params.log_tau_R), color="C2", lw=0.8, ls="--", label="log_τ_R truth")
    ax.fill_between(t, tau_F_q[0], tau_F_q[2], color="C5", alpha=0.18, label="log_τ_F 80%")
    ax.plot(t, tau_F_q[1], color="C5", lw=1.0, label="log_τ_F median")
    ax.axhline(float(truth_params.log_tau_F), color="C5", lw=0.8, ls="--", label="log_τ_F truth")
    for b in season_boundaries:
        ax.axvline(b, color="gray", lw=0.5, ls=":")
    ax.set_xlabel("day")
    ax.set_ylabel("log τ")
    ax.set_title("Liu-West posteriors on volatility-of-volatility (τ_R, τ_F) over time")
    ax.legend(loc="center right", frameon=False, fontsize=8, ncol=2)

    fig.suptitle(
        "Multi-season demo — Liu-West PF on seasonal Rt + drifting F",
        y=0.998,
    )
    out = _common.save(fig, "07_multi_season.png")
    print(f"saved: {out}")

    # --- sequential forecasts at four rolling origins ---
    # Reuse the already-fitted cloud at each origin time; no refit needed.
    # Origins are placed on the **ascending phase** of each wave — where the
    # truth (and the inferred model) think log_Rt > 0 and so the forecast
    # median actually has direction (exponential growth from current state),
    # avoiding the "flat median because origin Rt ≈ 1" pathology.  Horizon is
    # kept short (~2 months) so the forecast still reflects RW-Rt dispersion
    # rather than being dominated by the F-feedback equilibrium.
    h_max = 60
    origins = [60 + i * season_len for i in range(n_seasons)]
    origins = [o for o in origins if o + h_max <= T]
    print(f"\nsequential forecasts at origins {origins}, horizon {h_max}d:")

    fig2, ax = plt.subplots(1, 1, figsize=(12, 5.5))
    # Log y-axis: forecast upper tails can hit 10^5 from a rising-phase origin
    # while observed cases are in the 10²–10³ range; linear scale would
    # squash everything.  Clip y >= 1 to avoid log(0) artifacts.
    y_clipped = np.maximum(np.asarray(y), 0.5)
    ax.plot(t, np.maximum(np.asarray(mu_y_truth), 0.5), color="k", lw=0.7,
            alpha=0.6, label="true μ_y(t)")
    ax.scatter(t, y_clipped, s=4, color="k", alpha=0.3)
    for b in season_boundaries:
        ax.axvline(b, color="gray", lw=0.5, ls=":")

    forecast_colors = ["C2", "C3", "C4", "C8"]
    for i, (t0, color) in enumerate(zip(origins, forecast_colors)):
        # Cloud at end of day t0-1 (after seeing y[:t0]).
        particles_t0 = jax.tree.map(lambda x: x[t0 - 1], result.particles_history)
        params_t0 = jax.tree.map(lambda x: x[t0 - 1], result.params_history)
        lw_t0 = result.log_weights_history[t0 - 1]

        fc = forecast_from_cloud(
            jr.fold_in(jr.key(99), i), cfg, particles_t0, params_t0, lw_t0, h_max
        )
        h_t = t0 + np.arange(1, h_max + 1)
        # Weighted quantiles per horizon day.
        qs = (0.025, 0.25, 0.5, 0.75, 0.975)
        Q = jnp.stack([
            jax.vmap(weighted_quantile, in_axes=(0, None, None))(
                fc.y, fc.log_weights, jnp.asarray(q)
            ) for q in qs
        ])
        # Clip to >= 0.5 for log axis.
        ax.fill_between(h_t, jnp.maximum(Q[0], 0.5), jnp.maximum(Q[4], 0.5),
                        color=color, alpha=0.10)
        ax.fill_between(h_t, jnp.maximum(Q[1], 0.5), jnp.maximum(Q[3], 0.5),
                        color=color, alpha=0.22)
        ax.plot(h_t, jnp.maximum(Q[2], 0.5), color=color, lw=1.4,
                label=f"forecast from day {t0}")
        ax.axvline(t0, color=color, lw=0.6, ls="--", alpha=0.5)

    ax.set_xlabel("day")
    ax.set_ylabel("cases (log scale)")
    ax.set_yscale("log")
    ax.set_ylim(0.5, 1e6)
    ax.set_title(
        f"Sequential forecasts (50% and 95% PIs, {h_max}-day horizon) at four origins "
        "across the 4-year window"
    )
    ax.legend(loc="upper right", frameon=False, fontsize=9, ncol=2)
    out2 = _common.save(fig2, "07_multi_season_forecasts.png")
    print(f"saved: {out2}")


if __name__ == "__main__":
    main()
