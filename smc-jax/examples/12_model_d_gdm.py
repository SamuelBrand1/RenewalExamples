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
from pathlib import Path

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

# Locked synthetic dataset path — example 13 (NegBin model on the same data)
# loads from here.
DATA_DIR = Path(__file__).parent / "data"
TRUTH_NPZ = DATA_DIR / "12_truth.npz"


def simulate_outbreak_gdm(
    key,
    cfg,
    T: int,
    truth_log_Rt_init: float,
    truth_v_R_init: float,
    truth_log_I0: float,
    truth_log_sigma_vR: float,
    truth_b_0: float,
    truth_b_1: float,
    truth_log_M: float,
):
    """Forward-simulate Model E for ``T`` days using its own data-generating
    mechanism (GDM partition).  No immigration, no F-feedback.

    A negative ``truth_v_R_init`` is what bends the outbreak — `log Rt`
    drifts downward as the integrated velocity accumulates, taking Rt from
    above 1 (early growth) through 1 (peak) to below 1 (decline).  No
    susceptibility-depletion mechanism is needed.

    Returns the latent trajectory, the cohort-partition history ``O_traj``,
    and the observed daily counts.
    """
    truth_params = ParticleParamsGDM(
        log_sigma_vR=jnp.asarray(truth_log_sigma_vR),
        b_0=jnp.asarray(truth_b_0),
        b_1=jnp.asarray(truth_b_1),
        log_M=jnp.asarray(truth_log_M),
    )
    I0 = jnp.maximum(jnp.round(jnp.exp(jnp.asarray(truth_log_I0))), 1.0).astype(jnp.int64)
    init_state = ParticleStateGDM(
        log_Rt=jnp.asarray(truth_log_Rt_init),
        v_R=jnp.asarray(truth_v_R_init),
        log_I0=jnp.asarray(truth_log_I0),
        # Contact-tracing renewal reads λ_t from U_buf — seed it with
        # ``I0`` per age-slot ("I0 infectious people per day across the past
        # L days, none yet observed at t=0").
        U_buf=jnp.full((cfg.buffer_len,), I0),
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
    # Short outbreak (50 days) with HIGH per-cohort reporting over-dispersion
    # (small M).  This is the regime where the GDM observation model and
    # the auxiliary-q Wallenius proposal are most clearly justified — naive
    # NegBin-on-delay-conv blurs the cohort-budget coupling, and a
    # fixed-p Wallenius proposal would have catastrophic IS variance.
    # Tuned for a contact-tracing outbreak: tracing depletes the U-buffer as
    # cases are reported, which bends the curve naturally.  Need Rt large
    # enough to survive the depletion drag, but not so large that the
    # outbreak explodes past sampler bounds.
    truth_log_Rt_init = 0.75      # Rt ≈ 2.12 — high enough that with the
                                  # very small σ_vR below, log Rt is
                                  # essentially guaranteed to stay above 0
                                  # over the 25-day observation window
    truth_v_R_init = 0.0
    truth_log_I0 = 1.0            # I0 ≈ 2.7 per past-day slot
    truth_log_sigma_vR = -7.0     # σ_vR ≈ 9e-4 / day — small enough that
                                  # accumulated drift over 25 days has SD
                                  # ≈ 0.06, keeping log Rt safely above 0
    truth_b_0 = -1.5              # moderately slow reporting: p_1 = Φ(−1.5) ≈ 0.067
    truth_b_1 = 0.4               # mean reporting lag ≈ 3.5 days
    truth_log_M = 3.5             # M ≈ 33 — tight enough Beta that IS
                                  # variance stays manageable; cohort noise
                                  # still visibly above Binomial baseline

    cfg = dataclasses.replace(
        default_config(),
        sigma_floor=1e-6,
        ascertainment_alpha=1.0,      # 100 % ascertainment — Hantavirus-style.
                                       # Only "missing" cases come from tail
                                       # truncation (negligible at L = 14).
        # priors offset from truth to give the model something to learn
        init_log_Rt_mean=0.4,         # truth 0.75
        init_log_Rt_sd=0.4,
        init_v_R_mean=0.0,            # truth 0 (Rt held essentially constant)
        init_v_R_sd=0.04,
        init_log_sigma_vR_mean=-7.0,  # truth -10
        init_log_sigma_vR_sd=1.0,
        init_log_I0_mean=1.0,         # truth 1.0
        init_log_I0_sd=0.5,
        # GDM delay priors.  Prior centred at b_0 = -1.0 (≈ 16% reported
        # at lag 1) — what we'd reasonably expect a priori for a "tracing
        # catches infections" workflow, not chasing truth.  Wide sd so the
        # cloud can reach the truth of b_0 = -1.5 (slower reporting).
        init_b0_mean=-1.0,
        init_b0_sd=1.0,
        init_b1_mean=0.2,
        init_b1_sd=0.3,
        init_log_M_mean=3.0,          # truth 3.5
        init_log_M_sd=0.7,
    )

    T = 50
    print(
        f"Model E outbreak: T={T} days, "
        f"truth log_Rt_init={truth_log_Rt_init:.2f}, "
        f"α={cfg.ascertainment_alpha:.2f} (no immigration, no F-feedback, "
        f"min delay 1 day)"
    )

    # Resample seeds until the outbreak (a) doesn't fizzle and (b) keeps
    # log Rt strictly above 0 through day 25 — so that any bending in
    # cases attributable to "Rt fell" is structurally impossible in the
    # truth, leaving contact-tracing depletion as the only mechanism.
    for trial_seed in range(50):
        truth_traj, y, O_traj = simulate_outbreak_gdm(
            jr.key(trial_seed), cfg, T,
            truth_log_Rt_init, truth_v_R_init, truth_log_I0,
            truth_log_sigma_vR,
            truth_b_0, truth_b_1, truth_log_M,
        )
        I_total_early = int(jnp.sum(truth_traj.U_buf[:8, 0]))
        I_total = int(jnp.sum(truth_traj.U_buf[:, 0]))
        min_logRt_first25 = float(jnp.min(truth_traj.log_Rt[:25]))
        if (
            I_total_early >= 20
            and I_total >= 200
            and min_logRt_first25 > 0.0
        ):
            print(f"  truth seed = {trial_seed},  "
                  f"min log Rt over days 1-25 = {min_logRt_first25:+.3f}")
            break
    else:
        raise RuntimeError(
            "no outbreak-taking-off-with-Rt-above-1 found across 50 seeds — "
            "consider raising truth_log_Rt_init or lowering truth_log_sigma_vR"
        )
    # Truth quantities
    I_truth = truth_traj.U_buf[:, 0].astype(jnp.float64)
    log_Rt_truth = truth_traj.log_Rt
    y_np = np.asarray(y).astype(int)
    print(f"  peak y_t = {int(y_np.max())}, total cases = {int(y_np.sum())}")

    # Lock in the synthetic dataset so example 13 (and any future comparison)
    # operates on byte-identical observations + ground truth.
    DATA_DIR.mkdir(exist_ok=True)
    np.savez(
        TRUTH_NPZ,
        y=y_np,
        I_truth=np.asarray(I_truth),
        log_Rt_truth=np.asarray(log_Rt_truth),
        O_traj=np.asarray(O_traj),
        T=T,
        truth_log_Rt_init=truth_log_Rt_init,
        truth_v_R_init=truth_v_R_init,
        truth_log_I0=truth_log_I0,
        truth_log_sigma_vR=truth_log_sigma_vR,
        truth_b_0=truth_b_0,
        truth_b_1=truth_b_1,
        truth_log_M=truth_log_M,
        ascertainment_alpha=cfg.ascertainment_alpha,
    )
    print(f"  locked synthetic data → {TRUTH_NPZ}")

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
    # I(t) trace = freshest cohort slot post-step.  Under min-delay-1-day
    # and α = 1, no observations apply to age-0, so U_buf[..., 0] at the
    # end of step t equals the sampled N_t for that step.
    I_p = result.particles_history.U_buf[:, :, 0].astype(jnp.float64)
    log_sigma_vR_p = result.params_history.log_sigma_vR
    b_0_p = result.params_history.b_0
    b_1_p = result.params_history.b_1
    log_M_p = result.params_history.log_M

    qs = (0.05, 0.5, 0.95)
    log_Rt_q = jnp.stack(
        [jax.vmap(weighted_quantile)(log_Rt_p, lw, jnp.full((T,), q)) for q in qs]
    )
    I_q = jnp.stack(
        [jax.vmap(weighted_quantile)(I_p, lw, jnp.full((T,), q)) for q in qs]
    )

    final_lw = lw[-1]
    cov_logRt = float(jnp.mean((log_Rt_truth >= log_Rt_q[0]) & (log_Rt_truth <= log_Rt_q[2])))
    cov_I = float(jnp.mean((I_truth >= I_q[0]) & (I_truth <= I_q[2])))
    print(f"  log_Rt 90% filter cov   {cov_logRt:.3f}")
    print(f"  I(t)    90% filter cov   {cov_I:.3f}")
    print(f"  min ESS = {float(result.ess_history.min()):.1f} (out of {N})")
    print(f"  final log_σ_vR post mean: {float(_wmean(log_sigma_vR_p[-1], final_lw)):+.3f}  (truth {truth_log_sigma_vR:+.3f})")
    print(f"  final b_0      post mean: {float(_wmean(b_0_p[-1], final_lw)):+.3f}  (truth {truth_b_0:+.3f})")
    print(f"  final b_1      post mean: {float(_wmean(b_1_p[-1], final_lw)):+.3f}  (truth {truth_b_1:+.3f})")
    print(f"  final log_M    post mean: {float(_wmean(log_M_p[-1], final_lw)):+.3f}  (truth {truth_log_M:+.3f})")

    # --- figure 1: 5 panels ---
    t = np.arange(T)
    fig = plt.figure(figsize=(13, 13), constrained_layout=False)
    gs = fig.add_gridspec(5, 1, hspace=0.55,
                          height_ratios=[1, 1, 1, 0.9, 0.9])

    ax = fig.add_subplot(gs[0])
    ax.plot(t, y_np, "o-", color="k", ms=4, alpha=0.7, label="observed y_t")
    ax.set_ylabel("daily cases")
    ax.set_title("Model E — contact-tracing outbreak (GDM observation delay)")
    ax.legend(loc="upper right", frameon=False, fontsize=9)

    ax = fig.add_subplot(gs[1])
    ax.plot(t, log_Rt_truth, color="k", lw=1.4, label="truth")
    ax.fill_between(t, log_Rt_q[0], log_Rt_q[2], color="C0", alpha=0.2, label="filter 90%")
    ax.plot(t, log_Rt_q[1], color="C0", lw=1.0, label="filter median")
    ax.set_ylabel("log Rt")
    ax.set_title(f"log Rt tracking — filter 90% cov {cov_logRt:.2f}")
    ax.legend(loc="upper right", frameon=False, fontsize=8, ncol=3)

    ax = fig.add_subplot(gs[2])
    ax.plot(t, np.maximum(np.asarray(I_truth), 0.5), color="k", lw=1.0, label="truth I(t)")
    ax.fill_between(t, np.maximum(np.asarray(I_q[0]), 0.5), np.maximum(np.asarray(I_q[2]), 0.5),
                    color="C2", alpha=0.2, label="filter 90%")
    ax.plot(t, np.maximum(np.asarray(I_q[1]), 0.5), color="C2", lw=1.0, label="filter median")
    ax.set_ylabel("I(t)")
    ax.set_title(f"Latent infections — filter 90% cov {cov_I:.2f}")
    ax.set_xlabel("day")
    ax.legend(loc="upper right", frameon=False, fontsize=8, ncol=3)

    # Panel 3: cohort partition decomposition (truth-side) for selected days.
    ax = fig.add_subplot(gs[3])
    pick_days = [d for d in (8, 16, 24, 34, 44) if d < T]
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

    # Panel 4: 4-axis posterior strip
    gs_bot = gs[4].subgridspec(1, 4, wspace=0.45)
    wF = jnp.exp(final_lw - jnp.max(final_lw)); wF = wF / wF.sum()
    panels = [
        ("log σ_vR", log_sigma_vR_p[-1], "C1",
         float(cfg.init_log_sigma_vR_mean), float(cfg.init_log_sigma_vR_sd),
         truth_log_sigma_vR),
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
        "Model E — contact-tracing renewal + GDM observation delay + guided PF proposal",
        y=0.998,
    )
    out = _common.save(fig, "12_model_d_gdm.png")
    print(f"saved: {out}")

    # --- figure 2: sequential forecast fan ---
    h_max = 14
    origins = [d for d in (15, 25, 35) if d + h_max <= T]
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
    ax.plot(t, np.maximum(y_np.astype(float), 0.5), "o-", color="k", lw=0.7,
            ms=3, alpha=0.6, label="observed y_t")

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
        f"Model E sequential forecasts (50% and 95% PIs, {h_max}-day horizon)"
    )
    ax.legend(loc="upper right", frameon=False, fontsize=9, ncol=2)
    out2 = _common.save(fig2, "12_model_d_gdm_forecasts.png")
    print(f"saved: {out2}")


if __name__ == "__main__":
    main()
