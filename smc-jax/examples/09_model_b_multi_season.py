"""09 — Model B (Liu-West on log_σ) on the multi-season data from example 07.

Structurally a re-run of example 07 but with the cleaner inference design:
``log σ_R, log σ_F`` are Liu-West parameters directly (drifting via
shrink-jitter), not states walking under static ``τ``.  Same synthetic
truth (forced seasonal ``log Rt``, natural-RW ``log F``) so the figures
are directly comparable to ``07_multi_season.png`` / ``07_multi_season_forecasts.png``.
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
from smc_renewal.pf.model_sigma import (
    ParticleParamsSigma,
    ParticleStateSigma,
    TransitionNoiseSigma,
    expected_observation_sigma,
    step_sigma,
)
from smc_renewal.pf.runner_sigma import run_liu_west_sigma
from smc_renewal.state import ParticleParams
from smc_renewal.synthetic import sample_initial_state
from smc_renewal.transition import (
    TransitionNoise,
    expected_observation,
    step,
)


# ---- shared truth generation (same as example 07) -----------------------------

def simulate_seasonal_with_natural_F(
    key, cfg, log_Rt_seq, truth_params: ParticleParams,
    truth_log_F_init: float, truth_log_sigma_F_init: float,
    season_log_I0: float = 2.3,
):
    """Forward-simulate with log_Rt forced and log_F drawn naturally (Model A truth)."""
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


# ---- Model-B forecast helper -------------------------------------------------

def forecast_from_cloud_sigma(
    key, cfg, particles: ParticleStateSigma, params: ParticleParamsSigma,
    log_weights, h_max,
):
    """Forward-simulate h_max days from Model B's particle cloud."""
    N = log_weights.shape[0]
    k_noise, k_obs = jr.split(key, 2)
    all_noise = jr.normal(k_noise, (h_max, N, 2))

    def body(state, n_t):
        noises = TransitionNoiseSigma(eps_R=n_t[:, 0], eps_F=n_t[:, 1])
        new_state = jax.vmap(lambda s, p, n: step_sigma(s, p, n, cfg))(
            state, params, noises
        )
        mu_y = jax.vmap(lambda s: expected_observation_sigma(s, cfg))(new_state)
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

    # Same truth setup as example 07.
    truth_params = ParticleParams(
        log_tau_R=jnp.asarray(-3.0),
        log_tau_F=jnp.asarray(-7.0),
        log_phi=jnp.asarray(2.5),
    )
    truth_log_F_init = -8.0
    truth_log_sigma_F_init = -5.0

    # Cfg priors:
    #   - σ_R prior: centred on truth-implied scale (sin amp 0.45 / period 360
    #     → typical d/dt log Rt of order 0.008, so σ_R ~ 0.03, log_σ_R ~ -3.5).
    #   - σ_F prior: centred on truth's initial log_σ_F (= -5).
    #   - h (Liu-West shrinkage) tuned so σ-cloud can drift modestly over T=1440.
    cfg = dataclasses.replace(
        default_config(),
        sigma_floor=1e-6,
        init_log_Rt_sd=0.5,
        init_log_F_mean=truth_log_F_init,
        init_log_F_sd=1.0,
        init_log_sigma_R_mean=-3.5,
        init_log_sigma_R_sd=0.8,
        init_log_sigma_F_mean=-5.0,
        init_log_sigma_F_sd=1.0,
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

    print(f"Model B on multi-season data: T={T} days, {n_seasons} seasons")
    truth_traj, mu_y_truth, y = simulate_seasonal_with_natural_F(
        jr.key(7), cfg, log_Rt_truth, truth_params,
        truth_log_F_init, truth_log_sigma_F_init,
    )
    log_F_truth = truth_traj.log_F
    log_sigma_F_truth = truth_traj.log_sigma_F  # for visual reference

    N = 6000
    h = 0.07
    fixed_lag_L = 21
    print(f"  Liu-West PF (σ-variant): N={N}, h={h}, fixed_lag_L={fixed_lag_L}")
    result = run_liu_west_sigma(
        jr.key(11), cfg, y, n_particles=N, h=h, fixed_lag_L=fixed_lag_L
    )

    lw = result.log_weights_history
    log_Rt_p = result.particles_history.log_Rt
    log_F_p = result.particles_history.log_F
    log_sigma_R_p = result.params_history.log_sigma_R  # now a LW PARAM, not state
    log_sigma_F_p = result.params_history.log_sigma_F
    log_phi_p = result.params_history.log_phi

    # Quantiles for filter bands.
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

    # Smoother via particle_smooth (works on any per-leaf array with ancestors).
    n_smooth = 600
    smooth_keys = jr.split(jr.key(99), n_smooth)
    final_lw = lw[-1]
    smooth = lambda arr: jax.vmap(
        lambda k: particle_smooth(k, final_lw, arr, result.ancestors)
    )(smooth_keys)
    log_Rt_qs = jnp.quantile(smooth(log_Rt_p), jnp.array([0.05, 0.5, 0.95]), axis=0)
    log_F_qs = jnp.quantile(smooth(log_F_p), jnp.array([0.05, 0.5, 0.95]), axis=0)

    cov_logRt = float(jnp.mean((log_Rt_truth >= log_Rt_q[0]) & (log_Rt_truth <= log_Rt_q[2])))
    cov_logRt_s = float(jnp.mean((log_Rt_truth >= log_Rt_qs[0]) & (log_Rt_truth <= log_Rt_qs[2])))
    cov_logF = float(jnp.mean((log_F_truth >= log_F_q[0]) & (log_F_truth <= log_F_q[2])))
    cov_logF_s = float(jnp.mean((log_F_truth >= log_F_qs[0]) & (log_F_truth <= log_F_qs[2])))
    print(f"  log_Rt 90% filter coverage:   {cov_logRt:.3f}")
    print(f"  log_Rt 90% smoother coverage: {cov_logRt_s:.3f}")
    print(f"  log_F  90% filter coverage:   {cov_logF:.3f}")
    print(f"  log_F  90% smoother coverage: {cov_logF_s:.3f}")
    print(f"  final log_σ_R post mean:  {float(_wmean(log_sigma_R_p[-1], final_lw)):+.3f}")
    print(f"  final log_σ_F post mean:  {float(_wmean(log_sigma_F_p[-1], final_lw)):+.3f}  "
          f"(truth log_σ_F at T = {float(log_sigma_F_truth[-1]):+.3f})")
    print(f"  final log_φ post mean:    {float(_wmean(log_phi_p[-1], final_lw)):+.3f}  "
          f"(truth {float(truth_params.log_phi):+.3f})")

    # ---- main figure: 5 panels, mirror example 07 layout ----
    fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True,
                             gridspec_kw={"hspace": 0.5})
    season_boundaries = [i * season_len for i in range(1, n_seasons)]

    ax = axes[0]
    ax.plot(t, mu_y_truth, color="C0", lw=1.0, label="true μ_y(t)")
    ax.scatter(t, np.asarray(y), s=4, color="k", alpha=0.5, label="observed y")
    for b in season_boundaries:
        ax.axvline(b, color="gray", lw=0.5, ls=":")
    ax.set_ylabel("cases")
    ax.set_title("Model B (LW on log_σ) — same synthetic data as example 07")
    ax.legend(loc="upper left", frameon=False, fontsize=9)

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

    ax = axes[3]
    ax.fill_between(t, jnp.exp(log_sigma_R_q[0]), jnp.exp(log_sigma_R_q[2]),
                    color="C1", alpha=0.22, label="σ_R 90% (LW param)")
    ax.plot(t, jnp.exp(log_sigma_R_q[1]), color="C1", lw=1.0, label="σ_R median")
    ax.set_ylabel("σ_R(t)")
    ax.set_yscale("log")
    ax2 = ax.twinx()
    ax2.fill_between(t, jnp.exp(log_sigma_F_q[0]), jnp.exp(log_sigma_F_q[2]),
                     color="C5", alpha=0.22, label="σ_F 90% (LW param)")
    ax2.plot(t, jnp.exp(log_sigma_F_q[1]), color="C5", lw=1.0, label="σ_F median")
    # Truth σ_F for reference (Model A's σ_F(t) state, not directly the same
    # parameter, but the same physical quantity).
    ax2.plot(t, jnp.exp(log_sigma_F_truth), color="k", lw=0.8, ls=":",
             label="truth σ_F(t)")
    ax2.set_ylabel("σ_F(t)")
    ax2.set_yscale("log")
    for b in season_boundaries:
        ax.axvline(b, color="gray", lw=0.5, ls=":")
    ax.set_title("Liu-West σ-cloud trajectories (NB: these are PARAMETERS in Model B, not states)")
    ax.legend(loc="upper left", frameon=False, fontsize=7)
    ax2.legend(loc="upper right", frameon=False, fontsize=7)

    # Bottom row split into three sub-axes so each parameter shows on its
    # own x-scale (previously all three shared one axis, which masked the
    # tightly-identified log_φ as a thin spike next to wider log_σ histograms).
    axes[4].remove()
    gs_bottom = fig.add_gridspec(5, 3, height_ratios=[1, 1, 1, 1, 1],
                                 hspace=0.5, wspace=0.30)[4, :].subgridspec(1, 3, wspace=0.35)
    wF = jnp.exp(final_lw - jnp.max(final_lw)); wF = wF / wF.sum()
    panels = [
        ("log σ_R", log_sigma_R_p[-1], "C1", None),
        ("log σ_F", log_sigma_F_p[-1], "C5", None),
        ("log φ", log_phi_p[-1], "C2", float(truth_params.log_phi)),
    ]
    for i, (name, arr, color, truth_val) in enumerate(panels):
        a = fig.add_subplot(gs_bottom[0, i])
        a.hist(np.asarray(arr), bins=30, weights=np.asarray(wF),
               density=True, color=color, alpha=0.7)
        if truth_val is not None:
            a.axvline(truth_val, color="k", lw=1.2, ls="--", label="truth")
            a.legend(loc="upper right", frameon=False, fontsize=8)
        a.set_title(f"final-step posterior on {name}", fontsize=10)
        a.tick_params(axis="y", labelleft=False, left=False)

    fig.suptitle(
        "Model B — Liu-West on log_σ — same multi-season data as example 07",
        y=0.998,
    )
    out = _common.save(fig, "09_model_b_multi_season.png")
    print(f"saved: {out}")

    # ---- sequential forecasts at ascending-phase origins ----
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

        _mu, y_fc = forecast_from_cloud_sigma(
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
        f"Model B sequential forecasts (50% and 95% PIs, {h_max}-day horizon) "
        f"at four origins"
    )
    ax.legend(loc="upper right", frameon=False, fontsize=9, ncol=2)
    out2 = _common.save(fig2, "09_model_b_forecasts.png")
    print(f"saved: {out2}")


if __name__ == "__main__":
    main()
