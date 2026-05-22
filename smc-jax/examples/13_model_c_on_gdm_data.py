"""13 — Model C (NegBin obs) applied to Model E's GDM-generated outbreak.

Loads the locked synthetic dataset from ``examples/data/12_truth.npz`` (the
single-seed 50-day Model E outbreak with GDM cohort-partition observations,
α = 1, no F-feedback, no immigration, b_0 = −0.5) and fits **Model C**
(velocity-driven log Rt + log F, continuous renewal, delay-conv + NegBin
observation) to it.

To isolate the comparison to the *observation model* (not the delay
distribution itself), Model C is given a delay PMF that matches the
GDM's marginal lag distribution.  So the only structural differences are:

  • Discrete Poisson I_t → continuous deterministic renewal.
  • GDM cohort partition (with per-day Beta over-dispersion + cohort-budget
    coupling) → NegBin around the delay-convolved mean.

The question this example tries to answer is: when the data-generating
mechanism is cohort-partition GDM, does a NegBin-overdispersion model
still recover the latent dynamics, or does the structural mismatch bite?

No editorialising in the code itself — the figure and printed posteriors
just report what Model C makes of the GDM data.
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

import _common  # noqa: F401
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
from jax.scipy.stats import norm

import numpyro.distributions as dist

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


TRUTH_NPZ = Path(__file__).parent / "data" / "12_truth.npz"


def gdm_marginal_delay_pmf(b_0: float, b_1: float, L: int) -> jnp.ndarray:
    """Marginal P(delay = s) for s = 0, 1, …, L−1 under the GDM stick-breaking
    used in ``observation_gdm.gdm_beta_params``.

    p_0 = 0 (min delay 1 day); for s ≥ 1, p_s = Φ(b_0 + b_1 · (s − 1)) is the
    fraction of remaining cases reported at stage s.  This collapses the GDM
    over the q_s auxiliaries to give a single deterministic delay PMF.
    """
    stages = np.arange(L)
    p_s = np.empty(L)
    p_s[0] = 0.0
    p_s[1:] = norm.cdf(b_0 + b_1 * (stages[1:] - 1.0))
    pmf = np.empty(L)
    remaining = 1.0
    for s in range(L):
        pmf[s] = remaining * p_s[s]
        remaining *= 1.0 - p_s[s]
    return jnp.asarray(pmf / pmf.sum(), dtype=jnp.float64)


def _wmean(x, lw):
    w = jnp.exp(lw - jnp.max(lw))
    w = w / w.sum()
    return jnp.sum(w * x)


def main():
    _common.set_clean_style()

    if not TRUTH_NPZ.exists():
        raise FileNotFoundError(
            f"Locked synthetic dataset not found at {TRUTH_NPZ}. "
            f"Run examples/12_model_d_gdm.py first to generate it."
        )

    data = np.load(TRUTH_NPZ)
    y = jnp.asarray(data["y"], dtype=jnp.float64)
    log_Rt_truth = jnp.asarray(data["log_Rt_truth"])
    I_truth = jnp.asarray(data["I_truth"])
    T = int(data["T"])
    truth_b_0 = float(data["truth_b_0"])
    truth_b_1 = float(data["truth_b_1"])
    truth_log_M = float(data["truth_log_M"])
    truth_log_Rt_init = float(data["truth_log_Rt_init"])
    truth_v_R_init = float(data["truth_v_R_init"])
    truth_log_sigma_vR = float(data["truth_log_sigma_vR"])
    ascertainment_alpha = float(data["ascertainment_alpha"])
    print(f"Loaded locked synthetic data from {TRUTH_NPZ}")
    print(f"  T={T}, peak y_t={int(y.max())}, total cases={int(y.sum())}")
    print(f"  truth: log_Rt_init={truth_log_Rt_init:.2f}, v_R_init={truth_v_R_init:+.3f},")
    print(f"         b_0={truth_b_0:+.2f}, b_1={truth_b_1:+.2f}, log_M={truth_log_M:+.2f}")
    print(f"         α={ascertainment_alpha:.2f}")

    # --- Build Model C cfg ---
    # Use the GDM's marginal delay PMF as Model C's deterministic delay PMF.
    # This isolates the comparison to the observation-model layer.
    base = default_config()
    L_delay = int(base.delay_pmf.shape[0])
    matched_delay = gdm_marginal_delay_pmf(truth_b_0, truth_b_1, L_delay)
    print(f"  matched delay PMF (mean lag ≈ "
          f"{float((matched_delay * jnp.arange(L_delay)).sum()):.2f} days)")

    cfg = dataclasses.replace(
        base,
        delay_pmf=matched_delay,
        sigma_floor=1e-6,
        # Match Model E's prior on log_Rt and v_R so the LW cloud has
        # similar initialisation — only the model class differs.
        init_log_Rt_mean=0.3,
        init_log_Rt_sd=0.4,
        init_v_R_mean=0.0,
        init_v_R_sd=0.04,
        init_log_sigma_vR_mean=-5.0,
        init_log_sigma_vR_sd=1.0,
        # F-feedback priors: Model C has a log_F state.  Truth has F=0,
        # so prior centred very negative + tight should let the model
        # learn F ≈ 0 cleanly.
        init_log_F_mean=-12.0,
        init_log_F_sd=1.0,
        init_v_F_mean=0.0,
        init_v_F_sd=1e-4,
        init_log_sigma_vF_mean=-10.0,
        init_log_sigma_vF_sd=1.0,
        init_log_I0_mean=1.5,
        init_log_I0_sd=0.5,
        # NegBin overdispersion prior — wide-ish, let the model decide.
        prior_log_phi_mean=2.0,
        prior_log_phi_sd=1.0,
    )

    # --- Inference ---
    N = 8000
    h = 0.04
    fixed_lag_L = 14
    print(f"\nLiu-West PF (Model C, NegBin obs): N={N}, h={h}, fixed_lag_L={fixed_lag_L}")
    result = run_liu_west_trend(
        jr.key(11), cfg, y,
        n_particles=N, h=h, fixed_lag_L=fixed_lag_L,
    )

    lw = result.log_weights_history
    log_Rt_p = result.particles_history.log_Rt
    log_F_p = result.particles_history.log_F
    I_p = result.particles_history.I_buf[:, :, 0].astype(jnp.float64)
    log_sigma_vR_p = result.params_history.log_sigma_vR
    log_sigma_vF_p = result.params_history.log_sigma_vF
    log_phi_p = result.params_history.log_phi

    qs = (0.05, 0.5, 0.95)
    log_Rt_q = jnp.stack(
        [jax.vmap(weighted_quantile)(log_Rt_p, lw, jnp.full((T,), q)) for q in qs]
    )
    log_F_q = jnp.stack(
        [jax.vmap(weighted_quantile)(log_F_p, lw, jnp.full((T,), q)) for q in qs]
    )
    I_q = jnp.stack(
        [jax.vmap(weighted_quantile)(I_p, lw, jnp.full((T,), q)) for q in qs]
    )

    final_lw = lw[-1]
    cov_logRt = float(jnp.mean((log_Rt_truth >= log_Rt_q[0]) & (log_Rt_truth <= log_Rt_q[2])))
    cov_I = float(jnp.mean((I_truth >= I_q[0]) & (I_truth <= I_q[2])))
    rmse_logRt = float(jnp.sqrt(jnp.mean((log_Rt_q[1] - log_Rt_truth) ** 2)))
    rmse_I = float(jnp.sqrt(jnp.mean((I_q[1] - I_truth) ** 2)))
    print(f"  log_Rt 90% filter cov   {cov_logRt:.3f}   filter-median RMSE {rmse_logRt:.3f}")
    print(f"  I(t)    90% filter cov   {cov_I:.3f}   filter-median RMSE {rmse_I:.2f}")
    print(f"  min ESS = {float(result.ess_history.min()):.1f} (out of {N})")
    print(f"  final log_σ_vR post mean: {float(_wmean(log_sigma_vR_p[-1], final_lw)):+.3f}  (truth {truth_log_sigma_vR:+.3f})")
    print(f"  final log_σ_vF post mean: {float(_wmean(log_sigma_vF_p[-1], final_lw)):+.3f}  (Model E truth has no v_F)")
    print(f"  final log_φ    post mean: {float(_wmean(log_phi_p[-1], final_lw)):+.3f}   (truth has no NegBin — over-dispersion is structural)")

    # --- Figure ---
    t = np.arange(T)
    fig = plt.figure(figsize=(13, 13), constrained_layout=False)
    gs = fig.add_gridspec(5, 1, hspace=0.55,
                          height_ratios=[1, 1, 1, 1, 0.9])

    ax = fig.add_subplot(gs[0])
    ax.plot(t, np.asarray(y).astype(int), "o-", color="k", ms=4, alpha=0.7,
            label="observed y_t (from Model E / GDM)")
    ax.set_ylabel("daily cases")
    ax.set_title("Locked Model E synthetic data — fed to Model C (NegBin observation model)")
    ax.legend(loc="upper right", frameon=False, fontsize=9)

    ax = fig.add_subplot(gs[1])
    ax.plot(t, log_Rt_truth, color="k", lw=1.4, label="Model E truth")
    ax.fill_between(t, log_Rt_q[0], log_Rt_q[2], color="C3", alpha=0.2, label="Model C filter 90%")
    ax.plot(t, log_Rt_q[1], color="C3", lw=1.0, label="Model C filter median")
    ax.set_ylabel("log Rt")
    ax.set_title(
        f"log Rt — Model C filter on GDM-generated data  "
        f"(90% cov {cov_logRt:.2f}, median RMSE {rmse_logRt:.3f})"
    )
    ax.legend(loc="upper right", frameon=False, fontsize=8, ncol=3)

    ax = fig.add_subplot(gs[2])
    ax.plot(t, log_F_q[1], color="C4", lw=1.0, label="Model C filter median")
    ax.fill_between(t, log_F_q[0], log_F_q[2], color="C4", alpha=0.2, label="Model C filter 90%")
    ax.axhline(-12.0, color="k", lw=0.8, ls=":", label="prior centre (F ≈ 0)")
    ax.set_ylabel("log F(t)")
    ax.set_title("log F(t) — Model C's redundant F state (truth has no F-feedback)")
    ax.legend(loc="upper right", frameon=False, fontsize=8, ncol=3)

    ax = fig.add_subplot(gs[3])
    ax.plot(t, np.maximum(np.asarray(I_truth), 0.5), color="k", lw=1.0, label="Model E truth I(t)")
    ax.fill_between(t, np.maximum(np.asarray(I_q[0]), 0.5), np.maximum(np.asarray(I_q[2]), 0.5),
                    color="C2", alpha=0.2, label="Model C filter 90%")
    ax.plot(t, np.maximum(np.asarray(I_q[1]), 0.5), color="C2", lw=1.0, label="Model C filter median")
    ax.set_ylabel("I(t)")
    ax.set_title(
        f"Latent infections — Model C (continuous renewal) tracking GDM-Poisson truth  "
        f"(90% cov {cov_I:.2f}, median RMSE {rmse_I:.1f})"
    )
    ax.set_xlabel("day")
    ax.legend(loc="upper right", frameon=False, fontsize=8, ncol=3)

    # Bottom row: 3-axis posterior strip
    gs_bot = gs[4].subgridspec(1, 3, wspace=0.4)
    wF = jnp.exp(final_lw - jnp.max(final_lw)); wF = wF / wF.sum()
    panels = [
        ("log σ_vR", log_sigma_vR_p[-1], "C1",
         float(cfg.init_log_sigma_vR_mean), float(cfg.init_log_sigma_vR_sd),
         truth_log_sigma_vR),
        ("log σ_vF", log_sigma_vF_p[-1], "C6",
         float(cfg.init_log_sigma_vF_mean), float(cfg.init_log_sigma_vF_sd),
         None),
        ("log φ (NegBin)", log_phi_p[-1], "C2",
         float(cfg.prior_log_phi_mean), float(cfg.prior_log_phi_sd), None),
    ]
    for i, (name, arr, color, p_mu, p_sd, truth_val) in enumerate(panels):
        a = fig.add_subplot(gs_bot[0, i])
        arr_np = np.asarray(arr)
        a.hist(arr_np, bins=30, weights=np.asarray(wF), density=True,
               color=color, alpha=0.7, label="posterior")
        lo = min(float(arr_np.min()), p_mu - 3 * p_sd)
        hi = max(float(arr_np.max()), p_mu + 3 * p_sd)
        xs = np.linspace(lo, hi, 200)
        prior_pdf = np.exp(-0.5 * ((xs - p_mu) / p_sd) ** 2) / (p_sd * np.sqrt(2 * np.pi))
        a.plot(xs, prior_pdf, color="C7", lw=1.2, ls="--", label="prior")
        if truth_val is not None:
            a.axvline(truth_val, color="k", lw=1.2, label="truth")
        a.set_title(name, fontsize=10)
        a.tick_params(axis="y", labelleft=False, left=False)
        if i == 0:
            a.legend(loc="upper right", frameon=False, fontsize=7)

    fig.suptitle(
        "Model C (NegBin observation) on Model E's GDM-generated outbreak — "
        "structural-mismatch diagnostic",
        y=0.998,
    )
    out = _common.save(fig, "13_model_c_on_gdm_data.png")
    print(f"\nsaved: {out}")

    # --- Forecast figure: extrapolate Model C's cloud, compare to truth ---
    def forecast_from_cloud_trend(key, cfg, particles, params, h_max):
        N_ = particles.log_Rt.shape[0]
        k_noise, k_obs = jr.split(key, 2)
        all_noise = jr.normal(k_noise, (h_max, N_, 2))

        def body(state, n_t):
            noises = TransitionNoiseTrend(eta_R=n_t[:, 0], eta_F=n_t[:, 1])
            new_state = jax.vmap(lambda s, p, n: step_trend(s, p, n, cfg))(
                state, params, noises
            )
            mu_y = jax.vmap(lambda s: expected_observation_trend(s, cfg))(new_state)
            return new_state, mu_y

        _, mu_y_traj = jax.lax.scan(body, particles, all_noise)
        phi = jnp.exp(params.log_phi)
        y_traj = dist.NegativeBinomial2(
            mean=mu_y_traj, concentration=phi[None, :]
        ).sample(k_obs)
        return mu_y_traj, y_traj.astype(jnp.float64)

    h_max = 14
    origins = [d for d in (15, 25, 35) if d + h_max <= T]
    print(f"\nsequential forecasts at origins {origins}, horizon {h_max}d:")

    fig2, ax = plt.subplots(1, 1, figsize=(12, 5.5))
    ax.plot(t, np.maximum(np.asarray(y).astype(float), 0.5), "o-", color="k",
            lw=0.7, ms=3, alpha=0.6, label="observed y_t")

    forecast_colors = ["C2", "C3", "C4"]
    for i, (t0, color) in enumerate(zip(origins, forecast_colors)):
        particles_t0 = jax.tree.map(lambda x: x[t0 - 1], result.particles_history)
        params_t0 = jax.tree.map(lambda x: x[t0 - 1], result.params_history)
        lw_t0 = result.log_weights_history[t0 - 1]

        _mu, y_fc = forecast_from_cloud_trend(
            jr.fold_in(jr.key(99), i), cfg, particles_t0, params_t0, h_max,
        )
        h_t = t0 + np.arange(1, h_max + 1)
        qs5 = (0.025, 0.25, 0.5, 0.75, 0.975)
        Q = jnp.stack([
            jax.vmap(weighted_quantile, in_axes=(0, None, None))(
                y_fc, lw_t0, jnp.asarray(q)
            ) for q in qs5
        ])
        ax.fill_between(h_t, jnp.maximum(Q[0], 0.5), jnp.maximum(Q[4], 0.5),
                        color=color, alpha=0.12)
        ax.fill_between(h_t, jnp.maximum(Q[1], 0.5), jnp.maximum(Q[3], 0.5),
                        color=color, alpha=0.24)
        ax.plot(h_t, jnp.maximum(Q[2], 0.5), color=color, lw=1.4,
                label=f"forecast from day {t0}")
        ax.axvline(t0, color=color, lw=0.6, ls="--", alpha=0.5)

    ax.set_xlabel("day")
    ax.set_ylabel("daily cases (log scale)")
    ax.set_yscale("log")
    ax.set_ylim(0.5, 1e5)
    ax.set_title(
        f"Model C sequential forecasts (50% and 95% PIs, {h_max}-day horizon) "
        f"— extrapolating a *wrong* Rt trajectory"
    )
    ax.legend(loc="upper right", frameon=False, fontsize=9, ncol=2)
    out2 = _common.save(fig2, "13_model_c_on_gdm_data_forecasts.png")
    print(f"saved: {out2}")


if __name__ == "__main__":
    main()
