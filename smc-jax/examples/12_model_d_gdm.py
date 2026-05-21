"""12 — Model E: outbreak analysis with GDM observation delay + guided PF proposal.

Short single-outbreak (T=70 days) demonstrating that high-ascertainment
regimes break the conditional-independence assumption of examples 01–11's
observation model.  Replaces the simplistic ``μ_y(t) = Σ d_s · I[t−s]``
delay convolution + NegBin with a **Generalised Dirichlet Multinomial**
(Stoner et al) cohort partition: each cohort's eventually-observable cases
get stick-broken across reporting stages via independent Beta-Binomials
with means parameterised on the probit scale,
``Φ⁻¹(p_s) = b_0 + b_1 · s``.

Because the observation is now coupled with the latent partition, a
bootstrap PF fails (almost no particle samples a partition that hits the
observed ``y_t`` exactly).  We instead use a **multivariate hypergeometric
guided proposal** for the cohort partition.  The IS weight against the
target BetaBin product has a clean closed form where the binomial
coefficients cancel:

    Δlog w  =  Σ_s [log B(O_s+α_s, U[s]−O_s+β_s) − log B(α_s, β_s)]
             +  log C(ΣU, y_t)

The Liu-West cloud is 6-D:
``(log σ_vR, log σ_vF, log μ, b_0, b_1, log_M)``.
"""

from __future__ import annotations

import dataclasses

import _common  # noqa: F401
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np

from smc_renewal.config import default_config
from smc_renewal.pf.bootstrap import weighted_quantile
from smc_renewal.pf.model_gdm import (
    ParticleParamsGDM,
    ParticleStateGDM,
    forward_step,
)
from smc_renewal.pf.runner_gdm import run_liu_west_gdm


def simulate_outbreak_gdm(
    key,
    cfg,
    T: int,
    truth_log_Rt_init: float,
    truth_log_F_init: float,
    truth_log_I0: float,
    truth_log_sigma_vR: float,
    truth_log_sigma_vF: float,
    truth_log_mu: float,
    truth_b_0: float,
    truth_b_1: float,
    truth_log_M: float,
):
    """Forward-simulate Model E for ``T`` days using its own data-generating
    mechanism (GDM partition).

    log_Rt and log_F are not forced — they walk naturally from the given
    initial values.  F-feedback bends the curve as the outbreak grows; no
    seasonal forcing.  Returns the latent trajectory, expected daily cases
    (= sum of cohort BetaBin means), the actual partition history, and the
    observed daily counts.
    """
    truth_params = ParticleParamsGDM(
        log_sigma_vR=jnp.asarray(truth_log_sigma_vR),
        log_sigma_vF=jnp.asarray(truth_log_sigma_vF),
        log_mu=jnp.asarray(truth_log_mu),
        b_0=jnp.asarray(truth_b_0),
        b_1=jnp.asarray(truth_b_1),
        log_M=jnp.asarray(truth_log_M),
    )
    I0 = jnp.maximum(jnp.round(jnp.exp(jnp.asarray(truth_log_I0))), 1.0).astype(jnp.int64)
    init_state = ParticleStateGDM(
        log_Rt=jnp.asarray(truth_log_Rt_init),
        v_R=jnp.asarray(0.0),
        log_F=jnp.asarray(truth_log_F_init),
        v_F=jnp.asarray(0.0),
        log_I0=jnp.asarray(truth_log_I0),
        I_buf=jnp.full((cfg.buffer_len,), I0),
        # Fresh outbreak: no phantom unreported cases from pre-history.
        U_buf=jnp.zeros((cfg.buffer_len,), dtype=jnp.int64),
    )

    step_keys = jr.split(key, T)

    def body(state, k):
        new_state, y_t, O = forward_step(k, state, truth_params, cfg)
        return new_state, (new_state, y_t, O)

    _, (traj, y, O_traj) = jax.lax.scan(body, init_state, step_keys)
    return traj, y, O_traj


def _wmean(x, lw):
    w = jnp.exp(lw - jnp.max(lw))
    w = w / w.sum()
    return jnp.sum(w * x)


def main():
    _common.set_clean_style()

    # --- truth values ---
    truth_log_Rt_init = 0.55      # Rt ≈ 1.73, outbreak ascent
    truth_log_F_init = -6.0       # F ≈ 2.5e-3, modest feedback that bends curve
    truth_log_I0 = 1.0            # I0 ≈ 2.7 cases (small seed)
    truth_log_sigma_vR = -7.0     # σ_vR ≈ 9e-4 / day
    truth_log_sigma_vF = -10.0    # σ_vF ≈ 4.5e-5 / day (slow F drift)
    truth_log_mu = 0.0            # μ ≈ 1 imported case / day
    truth_b_0 = 0.2               # p_0 ≈ 0.58 (58 % same-day)
    truth_b_1 = 0.4               # p_s climbs with stage → fast clearance
    truth_log_M = 3.5             # M ≈ 33, moderate Beta concentration

    cfg = dataclasses.replace(
        default_config(),
        sigma_floor=1e-6,
        ascertainment_alpha=0.9,
        # priors offset from truth to give the model something to learn
        init_log_Rt_mean=0.2,         # truth 0.55
        init_log_Rt_sd=0.4,
        init_v_R_mean=0.0,
        init_v_R_sd=0.02,
        init_log_sigma_vR_mean=-6.0,
        init_log_sigma_vR_sd=1.0,
        init_log_F_mean=-6.0,         # prior centred on truth
        init_log_F_sd=1.0,
        init_v_F_mean=0.0,
        init_v_F_sd=1e-4,
        init_log_sigma_vF_mean=-9.0,
        init_log_sigma_vF_sd=1.0,
        init_log_I0_mean=1.0,
        init_log_I0_sd=0.5,
        init_log_mu_mean=0.0,
        init_log_mu_sd=0.6,
        # GDM delay priors
        init_b0_mean=0.0,
        init_b0_sd=0.5,
        init_b1_mean=0.2,
        init_b1_sd=0.3,
        init_log_M_mean=3.0,
        init_log_M_sd=0.6,
    )

    T = 70
    print(
        f"Model E outbreak: T={T} days, "
        f"truth log_Rt_init={truth_log_Rt_init:.2f}, "
        f"log_F_init={truth_log_F_init:.2f}, "
        f"α={cfg.ascertainment_alpha:.2f}"
    )

    truth_traj, y, O_traj = simulate_outbreak_gdm(
        jr.key(7), cfg, T,
        truth_log_Rt_init, truth_log_F_init, truth_log_I0,
        truth_log_sigma_vR, truth_log_sigma_vF, truth_log_mu,
        truth_b_0, truth_b_1, truth_log_M,
    )
    # Truth quantities
    I_truth = truth_traj.I_buf[:, 0].astype(jnp.float64)
    log_Rt_truth = truth_traj.log_Rt
    log_F_truth = truth_traj.log_F
    y_np = np.asarray(y).astype(int)
    print(f"  peak y_t = {int(y_np.max())}, total cases = {int(y_np.sum())}")

    # --- inference ---
    N = 8000
    h = 0.04
    fixed_lag_L = 14
    print(f"  Liu-West PF (Model E): N={N}, h={h}, fixed_lag_L={fixed_lag_L}")
    result = run_liu_west_gdm(
        jr.key(11), cfg, y,
        n_particles=N, h=h, fixed_lag_L=fixed_lag_L,
    )

    lw = result.log_weights_history
    log_Rt_p = result.particles_history.log_Rt
    log_F_p = result.particles_history.log_F
    I_p = result.particles_history.I_buf[:, :, 0].astype(jnp.float64)
    log_sigma_vR_p = result.params_history.log_sigma_vR
    log_sigma_vF_p = result.params_history.log_sigma_vF
    log_mu_p = result.params_history.log_mu
    b_0_p = result.params_history.b_0
    b_1_p = result.params_history.b_1
    log_M_p = result.params_history.log_M

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
    cov_logF = float(jnp.mean((log_F_truth >= log_F_q[0]) & (log_F_truth <= log_F_q[2])))
    cov_I = float(jnp.mean((I_truth >= I_q[0]) & (I_truth <= I_q[2])))
    print(f"  log_Rt 90% filter cov   {cov_logRt:.3f}")
    print(f"  log_F  90% filter cov   {cov_logF:.3f}")
    print(f"  I(t)    90% filter cov   {cov_I:.3f}")
    print(f"  min ESS = {float(result.ess_history.min()):.1f} (out of {N})")
    print(f"  final log_σ_vR post mean: {float(_wmean(log_sigma_vR_p[-1], final_lw)):+.3f}  (truth {truth_log_sigma_vR:+.3f})")
    print(f"  final log_σ_vF post mean: {float(_wmean(log_sigma_vF_p[-1], final_lw)):+.3f}  (truth {truth_log_sigma_vF:+.3f})")
    print(f"  final log_μ    post mean: {float(_wmean(log_mu_p[-1], final_lw)):+.3f}  (truth {truth_log_mu:+.3f})")
    print(f"  final b_0      post mean: {float(_wmean(b_0_p[-1], final_lw)):+.3f}  (truth {truth_b_0:+.3f})")
    print(f"  final b_1      post mean: {float(_wmean(b_1_p[-1], final_lw)):+.3f}  (truth {truth_b_1:+.3f})")
    print(f"  final log_M    post mean: {float(_wmean(log_M_p[-1], final_lw)):+.3f}  (truth {truth_log_M:+.3f})")

    # --- figure 1: 6 panels ---
    t = np.arange(T)
    fig = plt.figure(figsize=(13, 15), constrained_layout=False)
    gs = fig.add_gridspec(6, 1, hspace=0.55,
                          height_ratios=[1, 1, 1, 1, 0.9, 0.9])

    ax = fig.add_subplot(gs[0])
    ax.plot(t, y_np, "o-", color="k", ms=4, alpha=0.7, label="observed y_t")
    ax.set_ylabel("daily cases")
    ax.set_title("Model E — outbreak analysis (GDM observation delay)")
    ax.legend(loc="upper right", frameon=False, fontsize=9)

    ax = fig.add_subplot(gs[1])
    ax.plot(t, log_Rt_truth, color="k", lw=1.4, label="truth")
    ax.fill_between(t, log_Rt_q[0], log_Rt_q[2], color="C0", alpha=0.2, label="filter 90%")
    ax.plot(t, log_Rt_q[1], color="C0", lw=1.0, label="filter median")
    ax.set_ylabel("log Rt")
    ax.set_title(f"log Rt tracking — filter 90% cov {cov_logRt:.2f}")
    ax.legend(loc="upper right", frameon=False, fontsize=8, ncol=3)

    ax = fig.add_subplot(gs[2])
    ax.plot(t, log_F_truth, color="k", lw=1.4, label="truth")
    ax.fill_between(t, log_F_q[0], log_F_q[2], color="C3", alpha=0.2, label="filter 90%")
    ax.plot(t, log_F_q[1], color="C3", lw=1.0, label="filter median")
    ax.set_ylabel("log F(t)")
    ax.set_title(f"log F tracking — filter 90% cov {cov_logF:.2f}")
    ax.legend(loc="upper right", frameon=False, fontsize=8, ncol=3)

    ax = fig.add_subplot(gs[3])
    ax.plot(t, np.maximum(np.asarray(I_truth), 0.5), color="k", lw=1.0, label="truth I(t)")
    ax.fill_between(t, np.maximum(np.asarray(I_q[0]), 0.5), np.maximum(np.asarray(I_q[2]), 0.5),
                    color="C2", alpha=0.2, label="filter 90%")
    ax.plot(t, np.maximum(np.asarray(I_q[1]), 0.5), color="C2", lw=1.0, label="filter median")
    ax.set_ylabel("I(t)")
    ax.set_title(f"Latent infections — filter 90% cov {cov_I:.2f}")
    ax.set_xlabel("day")
    ax.legend(loc="upper right", frameon=False, fontsize=8, ncol=3)

    # Panel 4: cohort partition decomposition (truth-side) for selected days.
    ax = fig.add_subplot(gs[4])
    pick_days = [d for d in (10, 20, 30, 45, 60) if d < T]
    L = cfg.buffer_len
    bottoms = np.zeros(len(pick_days))
    stage_cmap = plt.cm.viridis(np.linspace(0.1, 0.9, L))
    for s in range(L):
        heights = np.array([int(O_traj[d, s]) for d in pick_days])
        ax.bar(
            range(len(pick_days)), heights, bottom=bottoms,
            color=stage_cmap[s], edgecolor="white", linewidth=0.3,
            label=f"lag {s}" if s in (0, 1, 2, 3, 6, 9, 13) else None,
        )
        bottoms += heights
    ax.set_xticks(range(len(pick_days)))
    ax.set_xticklabels([f"day {d}" for d in pick_days])
    ax.set_ylabel("cohort contribution to y_t")
    ax.set_title("Decomposition of y_t into per-cohort lag contributions (truth)")
    ax.legend(loc="upper right", frameon=False, fontsize=7, ncol=3)

    # Panel 5: 6-axis posterior strip
    gs_bot = gs[5].subgridspec(1, 6, wspace=0.45)
    wF = jnp.exp(final_lw - jnp.max(final_lw)); wF = wF / wF.sum()
    panels = [
        ("log σ_vR", log_sigma_vR_p[-1], "C1",
         float(cfg.init_log_sigma_vR_mean), float(cfg.init_log_sigma_vR_sd),
         truth_log_sigma_vR),
        ("log σ_vF", log_sigma_vF_p[-1], "C6",
         float(cfg.init_log_sigma_vF_mean), float(cfg.init_log_sigma_vF_sd),
         truth_log_sigma_vF),
        ("log μ", log_mu_p[-1], "C5",
         float(cfg.init_log_mu_mean), float(cfg.init_log_mu_sd), truth_log_mu),
        ("b_0", b_0_p[-1], "C8",
         float(cfg.init_b0_mean), float(cfg.init_b0_sd), truth_b_0),
        ("b_1", b_1_p[-1], "C9",
         float(cfg.init_b1_mean), float(cfg.init_b1_sd), truth_b_1),
        ("log M", log_M_p[-1], "C2",
         float(cfg.init_log_M_mean), float(cfg.init_log_M_sd), truth_log_M),
    ]
    for i, (name, arr, color, p_mu, p_sd, truth_val) in enumerate(panels):
        a = fig.add_subplot(gs_bot[0, i])
        arr_np = np.asarray(arr)
        a.hist(
            arr_np, bins=30, weights=np.asarray(wF), density=True,
            color=color, alpha=0.7, label="posterior",
        )
        lo = min(float(arr_np.min()), p_mu - 3 * p_sd)
        hi = max(float(arr_np.max()), p_mu + 3 * p_sd)
        xs = np.linspace(lo, hi, 200)
        prior_pdf = np.exp(-0.5 * ((xs - p_mu) / p_sd) ** 2) / (p_sd * np.sqrt(2 * np.pi))
        a.plot(xs, prior_pdf, color="C7", lw=1.2, ls="--", label="prior")
        a.axvline(truth_val, color="k", lw=1.2, label="truth")
        a.set_title(f"{name}", fontsize=10)
        a.tick_params(axis="y", labelleft=False, left=False)
        if i == 0:
            a.legend(loc="upper right", frameon=False, fontsize=7)

    fig.suptitle(
        "Model E — discrete renewal + GDM observation delay + guided PF proposal",
        y=0.998,
    )
    out = _common.save(fig, "12_model_d_gdm.png")
    print(f"saved: {out}")

    # --- figure 2: sequential forecast fan ---
    h_max = 14
    origins = [d for d in (25, 40, 55) if d + h_max <= T]
    print(f"\nsequential forecasts at origins {origins}, horizon {h_max}d:")

    def forecast_from_cloud_gdm(key, cfg, particles, params, h_max):
        N_ = particles.log_Rt.shape[0]
        keys_outer = jr.split(key, h_max)

        def body(state, k_t):
            keys_per = jr.split(k_t, N_)
            new_state, y_t, _O = jax.vmap(
                lambda kk, s, p: forward_step(kk, s, p, cfg)
            )(keys_per, state, params)
            return new_state, y_t

        _, y_traj = jax.lax.scan(body, particles, keys_outer)
        return y_traj  # shape (h_max, N)

    fig2, ax = plt.subplots(1, 1, figsize=(12, 5.5))
    ax.plot(t, y_np, "o-", color="k", lw=0.7, ms=3, alpha=0.6, label="observed y_t")

    forecast_colors = ["C2", "C3", "C4"]
    for i, (t0, color) in enumerate(zip(origins, forecast_colors)):
        particles_t0 = jax.tree.map(lambda x: x[t0 - 1], result.particles_history)
        params_t0 = jax.tree.map(lambda x: x[t0 - 1], result.params_history)
        lw_t0 = result.log_weights_history[t0 - 1]

        y_fc = forecast_from_cloud_gdm(
            jr.fold_in(jr.key(99), i), cfg, particles_t0, params_t0, h_max,
        )
        h_t = t0 + np.arange(1, h_max + 1)
        qs5 = (0.025, 0.25, 0.5, 0.75, 0.975)
        Q = jnp.stack([
            jax.vmap(weighted_quantile, in_axes=(0, None, None))(
                y_fc, lw_t0, jnp.asarray(q)
            ) for q in qs5
        ])
        ax.fill_between(h_t, Q[0], Q[4], color=color, alpha=0.12)
        ax.fill_between(h_t, Q[1], Q[3], color=color, alpha=0.24)
        ax.plot(h_t, Q[2], color=color, lw=1.4, label=f"forecast from day {t0}")
        ax.axvline(t0, color=color, lw=0.6, ls="--", alpha=0.5)

    ax.set_xlabel("day")
    ax.set_ylabel("daily cases")
    ax.set_title(
        f"Model E sequential forecasts (50% and 95% PIs, {h_max}-day horizon)"
    )
    ax.legend(loc="upper right", frameon=False, fontsize=9, ncol=2)
    out2 = _common.save(fig2, "12_model_d_gdm_forecasts.png")
    print(f"saved: {out2}")


if __name__ == "__main__":
    main()
