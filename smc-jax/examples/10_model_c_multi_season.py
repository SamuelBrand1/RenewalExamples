"""10 — Model C (integrated-Brownian-motion log Rt) on multi-season data.

Same synthetic truth as examples 07 (Model A) and 09 (Model B) — a forced
seasonal log Rt plus a naturally-drifting log F.  Here the inference model
adds a velocity state for log Rt and log F, so the median forecast actually
extrapolates whichever direction the inferred trend is going (rather than
freezing at the current level as in Models A and B).

We make the same 5-panel diagnostic figure plus the four-origins forecast
figure for direct visual comparison with example 09.
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
from smc_renewal.pf.bootstrap import weighted_quantile
from smc_renewal.pf.model_trend import (
    ParticleParamsTrend,
    ParticleStateTrend,
    TransitionNoiseTrend,
    expected_observation_trend,
    step_trend,
)
from smc_renewal.pf.runner_trend import run_liu_west_trend
from smc_renewal.state import ParticleParams
from smc_renewal.synthetic import sample_initial_state
from smc_renewal.transition import (
    TransitionNoise,
    expected_observation,
    step,
)


def simulate_seasonal_with_natural_F(
    key, cfg, log_Rt_seq, truth_params, truth_log_F_init, truth_log_sigma_F_init,
    season_log_I0=2.3,
):
    """Same truth as examples 07/09 — Model A's truth."""
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


def forecast_from_cloud_trend(
    key, cfg, particles: ParticleStateTrend, params: ParticleParamsTrend,
    log_weights, h_max,
):
    """Forward-simulate h_max days from Model C's cloud."""
    N = log_weights.shape[0]
    k_noise, k_obs = jr.split(key, 2)
    all_noise = jr.normal(k_noise, (h_max, N, 2))

    def body(state, n_t):
        noises = TransitionNoiseTrend(eta_R=n_t[:, 0], eta_F=n_t[:, 1])
        new_state = jax.vmap(lambda s, p, n: step_trend(s, p, n, cfg))(
            state, params, noises
        )
        mu_y = jax.vmap(lambda s: expected_observation_trend(s, cfg))(new_state)
        return new_state, mu_y

    _, mu_y_traj = jax.lax.scan(body, particles, all_noise)
    phi = jnp.exp(params.log_phi)
    y_traj = dist.NegativeBinomial2(mean=mu_y_traj, concentration=phi[None, :]).sample(k_obs)
    return mu_y_traj, y_traj.astype(jnp.float64)


def _wmean(x, lw):
    w = jnp.exp(lw - jnp.max(lw))
    w = w / w.sum()
    return jnp.sum(w * x)


def main():
    _common.set_clean_style()

    truth_params = ParticleParams(
        log_tau_R=jnp.asarray(-3.0),
        log_tau_F=jnp.asarray(-7.0),
        log_phi=jnp.asarray(2.5),
    )
    truth_log_F_init = -8.0
    truth_log_sigma_F_init = -5.0

    cfg = dataclasses.replace(
        default_config(),
        sigma_floor=1e-6,
        init_log_Rt_sd=0.5,
        init_log_F_mean=truth_log_F_init,
        init_log_F_sd=1.0,
        # Initial velocity priors: small (no strong "currently trending" prior).
        init_v_R_mean=0.0,
        init_v_R_sd=0.02,
        init_v_F_mean=0.0,
        init_v_F_sd=1e-4,
        # Volatility of velocity: per-step std on Δv.  Set σ_vR so that v_R
        # can swing meaningfully over a season (~360 days): σ_vR · √360 ≈ 0.04
        # ⇒ σ_vR ≈ 0.002, log_σ_vR ≈ -6.  Wide prior covers ±1 log-unit.
        init_log_sigma_vR_mean=-6.0,
        init_log_sigma_vR_sd=1.0,
        init_log_sigma_vF_mean=-10.0,
        init_log_sigma_vF_sd=1.0,
        prior_log_phi_mean=2.5,
        prior_log_phi_sd=0.5,
    )

    season_len = 360
    n_seasons = 4
    T = season_len * n_seasons
    t = jnp.arange(T)
    base = 0.45 * jnp.sin(2 * jnp.pi * t / season_len) + 0.02
    raw = 0.06 * jr.normal(jr.key(33), (T,))
    smoothed = jnp.convolve(raw, jnp.ones(7) / 7, mode="same")
    log_Rt_truth = base + smoothed

    print(f"Model C on multi-season data: T={T} days, {n_seasons} seasons")
    truth_traj, mu_y_truth, y = simulate_seasonal_with_natural_F(
        jr.key(7), cfg, log_Rt_truth, truth_params,
        truth_log_F_init, truth_log_sigma_F_init,
    )
    log_F_truth = truth_traj.log_F

    N = 6000
    h = 0.07
    fixed_lag_L = 21
    print(f"  Liu-West PF (trend variant): N={N}, h={h}, fixed_lag_L={fixed_lag_L}")
    result = run_liu_west_trend(
        jr.key(11), cfg, y, n_particles=N, h=h, fixed_lag_L=fixed_lag_L
    )

    lw = result.log_weights_history
    log_Rt_p = result.particles_history.log_Rt
    v_R_p = result.particles_history.v_R
    log_F_p = result.particles_history.log_F
    v_F_p = result.particles_history.v_F
    log_sigma_vR_p = result.params_history.log_sigma_vR
    log_sigma_vF_p = result.params_history.log_sigma_vF
    log_phi_p = result.params_history.log_phi

    qs = (0.05, 0.5, 0.95)
    log_Rt_q = jnp.stack(
        [jax.vmap(weighted_quantile)(log_Rt_p, lw, jnp.full((T,), q)) for q in qs]
    )
    v_R_q = jnp.stack(
        [jax.vmap(weighted_quantile)(v_R_p, lw, jnp.full((T,), q)) for q in qs]
    )
    log_F_q = jnp.stack(
        [jax.vmap(weighted_quantile)(log_F_p, lw, jnp.full((T,), q)) for q in qs]
    )

    # Smoother via particle_smooth.
    n_smooth = 600
    smooth_keys = jr.split(jr.key(99), n_smooth)
    final_lw = lw[-1]
    smooth = lambda arr: jax.vmap(
        lambda k: particle_smooth(k, final_lw, arr, result.ancestors)
    )(smooth_keys)
    log_Rt_qs = jnp.quantile(smooth(log_Rt_p), jnp.array([0.05, 0.5, 0.95]), axis=0)
    v_R_qs = jnp.quantile(smooth(v_R_p), jnp.array([0.05, 0.5, 0.95]), axis=0)
    log_F_qs = jnp.quantile(smooth(log_F_p), jnp.array([0.05, 0.5, 0.95]), axis=0)

    cov_logRt = float(jnp.mean((log_Rt_truth >= log_Rt_q[0]) & (log_Rt_truth <= log_Rt_q[2])))
    cov_logRt_s = float(jnp.mean((log_Rt_truth >= log_Rt_qs[0]) & (log_Rt_truth <= log_Rt_qs[2])))
    cov_logF = float(jnp.mean((log_F_truth >= log_F_q[0]) & (log_F_truth <= log_F_q[2])))
    cov_logF_s = float(jnp.mean((log_F_truth >= log_F_qs[0]) & (log_F_truth <= log_F_qs[2])))

    # True v_R (derivative of the seasonal log_Rt trajectory).  We compare
    # against this approximately via finite differences of log_Rt_truth.
    v_R_truth = jnp.gradient(log_Rt_truth)

    print(f"  log_Rt 90% filter coverage:   {cov_logRt:.3f}")
    print(f"  log_Rt 90% smoother coverage: {cov_logRt_s:.3f}")
    print(f"  log_F  90% filter coverage:   {cov_logF:.3f}")
    print(f"  log_F  90% smoother coverage: {cov_logF_s:.3f}")
    print(f"  final log_σ_vR post mean: {float(_wmean(log_sigma_vR_p[-1], final_lw)):+.3f}")
    print(f"  final log_σ_vF post mean: {float(_wmean(log_sigma_vF_p[-1], final_lw)):+.3f}")
    print(f"  final log_φ    post mean: {float(_wmean(log_phi_p[-1], final_lw)):+.3f}  "
          f"(truth {float(truth_params.log_phi):+.3f})")

    # ---- main figure: 5 panels ----
    fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True,
                             gridspec_kw={"hspace": 0.5})
    season_boundaries = [i * season_len for i in range(1, n_seasons)]

    ax = axes[0]
    # Cases on log scale with 0.5 offset so the trough days (y=0) stay
    # visible — matches the convention in the forecast figure.
    ax.plot(t, jnp.maximum(mu_y_truth, 0.5), color="C0", lw=1.0, label="true μ_y(t)")
    ax.scatter(t, np.maximum(np.asarray(y), 0.5), s=4, color="k", alpha=0.5,
               label="observed y")
    for b in season_boundaries:
        ax.axvline(b, color="gray", lw=0.5, ls=":")
    ax.set_yscale("log")
    ax.set_ylabel("cases (log scale)")
    ax.set_title("Model C (integrated BM on log Rt and log F) — same data as 07, 09")
    ax.legend(loc="upper right", frameon=False, fontsize=9)

    ax = axes[1]
    ax.plot(t, log_Rt_truth, color="k", lw=1.4, label="truth")
    ax.fill_between(t, log_Rt_q[0], log_Rt_q[2], color="C0", alpha=0.18, label="filter 90%")
    ax.plot(t, log_Rt_q[1], color="C0", lw=1.0, label="filter median")
    ax.fill_between(t, log_Rt_qs[0], log_Rt_qs[2], color="C4", alpha=0.20, label="smoother 90%")
    ax.plot(t, log_Rt_qs[1], color="C4", lw=1.0, label="smoother median")
    for b in season_boundaries:
        ax.axvline(b, color="gray", lw=0.5, ls=":")
    ax.set_ylabel("log Rt")
    ax.set_title(f"log Rt tracking — filter cov {cov_logRt:.2f}, smoother cov {cov_logRt_s:.2f}")
    ax.legend(loc="upper right", frameon=False, fontsize=8, ncol=3)

    ax = axes[2]
    ax.plot(t, v_R_truth, color="k", lw=1.2, label="truth (∂log Rt / ∂t)")
    ax.fill_between(t, v_R_q[0], v_R_q[2], color="C0", alpha=0.18, label="filter 90%")
    ax.plot(t, v_R_q[1], color="C0", lw=1.0, label="filter median")
    ax.fill_between(t, v_R_qs[0], v_R_qs[2], color="C4", alpha=0.20, label="smoother 90%")
    ax.plot(t, v_R_qs[1], color="C4", lw=1.0, label="smoother median")
    ax.axhline(0, color="k", lw=0.3)
    for b in season_boundaries:
        ax.axvline(b, color="gray", lw=0.5, ls=":")
    ax.set_ylabel("v_R(t) = d/dt log Rt")
    ax.set_title("Inferred velocity (slope) of log Rt — Model C's distinctive state coordinate")
    ax.legend(loc="upper right", frameon=False, fontsize=8, ncol=3)

    ax = axes[3]
    ax.plot(t, log_F_truth, color="k", lw=1.4, label="truth")
    ax.fill_between(t, log_F_q[0], log_F_q[2], color="C3", alpha=0.18, label="filter 90%")
    ax.plot(t, log_F_q[1], color="C3", lw=1.0, label="filter median")
    ax.fill_between(t, log_F_qs[0], log_F_qs[2], color="C4", alpha=0.20, label="smoother 90%")
    ax.plot(t, log_F_qs[1], color="C4", lw=1.0, label="smoother median")
    for b in season_boundaries:
        ax.axvline(b, color="gray", lw=0.5, ls=":")
    ax.set_ylabel("log F(t)")
    ax.set_title(f"log F tracking — filter cov {cov_logF:.2f}, smoother cov {cov_logF_s:.2f}")
    ax.legend(loc="upper right", frameon=False, fontsize=8, ncol=3)

    # Bottom row: posterior on the three Liu-West parameters, each in its own
    # subaxis, with the prior overlaid as a dashed Gaussian curve.  Comparing
    # prior vs posterior makes the magnitude of the Liu-West update explicit.
    axes[4].remove()
    gs_bottom = fig.add_gridspec(5, 1)[4, :].subgridspec(1, 3, wspace=0.35)
    wF = jnp.exp(final_lw - jnp.max(final_lw)); wF = wF / wF.sum()
    panels = [
        ("log σ_vR", log_sigma_vR_p[-1], "C1",
         float(cfg.init_log_sigma_vR_mean), float(cfg.init_log_sigma_vR_sd), None),
        ("log σ_vF", log_sigma_vF_p[-1], "C5",
         float(cfg.init_log_sigma_vF_mean), float(cfg.init_log_sigma_vF_sd), None),
        ("log φ", log_phi_p[-1], "C2",
         float(cfg.prior_log_phi_mean), float(cfg.prior_log_phi_sd),
         float(truth_params.log_phi)),
    ]
    for i, (name, arr, color, p_mu, p_sd, truth_val) in enumerate(panels):
        a = fig.add_subplot(gs_bottom[0, i])
        arr_np = np.asarray(arr)
        a.hist(arr_np, bins=30, weights=np.asarray(wF),
               density=True, color=color, alpha=0.7, label="posterior")
        # Prior pdf on a span that comfortably covers both prior and posterior.
        lo = min(float(arr_np.min()), p_mu - 3 * p_sd)
        hi = max(float(arr_np.max()), p_mu + 3 * p_sd)
        xs = np.linspace(lo, hi, 200)
        prior_pdf = np.exp(-0.5 * ((xs - p_mu) / p_sd) ** 2) / (p_sd * np.sqrt(2 * np.pi))
        a.plot(xs, prior_pdf, color="C7", lw=1.2, ls="--", label="prior")
        if truth_val is not None:
            a.axvline(truth_val, color="k", lw=1.2, label="truth")
        a.set_title(f"final-step posterior on {name}", fontsize=10)
        a.tick_params(axis="y", labelleft=False, left=False)
        if i == 0:
            a.legend(loc="upper right", frameon=False, fontsize=7)

    fig.suptitle(
        "Model C — Liu-West on (log σ_vR, log σ_vF, log φ); state carries velocities (v_R, v_F)",
        y=0.998,
    )
    out = _common.save(fig, "10_model_c_multi_season.png")
    print(f"saved: {out}")

    # ---- sequential forecasts ----
    h_max = 60
    origins = [60 + i * season_len for i in range(n_seasons)]
    origins = [o for o in origins if o + h_max <= T]
    print(f"\nsequential forecasts at origins {origins}, horizon {h_max}d:")

    fig2, ax = plt.subplots(1, 1, figsize=(12, 5.5))
    ax.plot(t, jnp.maximum(jnp.asarray(mu_y_truth), 0.5), color="k", lw=0.7,
            alpha=0.6, label="true μ_y(t)")
    ax.scatter(t, np.maximum(np.asarray(y), 0.5), s=4, color="k", alpha=0.3)
    for b in season_boundaries:
        ax.axvline(b, color="gray", lw=0.5, ls=":")

    forecast_colors = ["C2", "C3", "C4", "C8"]
    for i, (t0, color) in enumerate(zip(origins, forecast_colors)):
        particles_t0 = jax.tree.map(lambda x: x[t0 - 1], result.particles_history)
        params_t0 = jax.tree.map(lambda x: x[t0 - 1], result.params_history)
        lw_t0 = result.log_weights_history[t0 - 1]

        _mu, y_fc = forecast_from_cloud_trend(
            jr.fold_in(jr.key(99), i), cfg, particles_t0, params_t0, lw_t0, h_max
        )
        h_t = t0 + np.arange(1, h_max + 1)
        qs5 = (0.025, 0.25, 0.5, 0.75, 0.975)
        Q = jnp.stack([
            jax.vmap(weighted_quantile, in_axes=(0, None, None))(
                y_fc, lw_t0, jnp.asarray(q)
            ) for q in qs5
        ])
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
        f"Model C sequential forecasts (50% and 95% PIs, {h_max}-day horizon) "
        f"— median *extrapolates current trend*"
    )
    ax.legend(loc="upper right", frameon=False, fontsize=9, ncol=2)
    out2 = _common.save(fig2, "10_model_c_forecasts.png")
    print(f"saved: {out2}")


if __name__ == "__main__":
    main()
