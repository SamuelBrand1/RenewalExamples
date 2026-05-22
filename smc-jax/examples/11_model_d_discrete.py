"""11 — Model D (discrete Poisson renewal + F-feedback + immigration) on multi-season data.

Sibling of example 10 (Model C).  Same integrated-BM dynamics on both
``log Rt`` and ``log F``, same F-feedback ``exp(-F · conv(I, g))``, but the
renewal core is now stochastic:

    λ_t = μ + Rt · exp(-F · conv(I, g)) · conv(I, g)
    I_t ~ Poisson(λ_t)

A small immigration rate μ (inferred as a Liu-West parameter) prevents
extinction.  Bootstrap PF handles the discrete-count latent state just as
comfortably as the deterministic-renewal variants — demonstrating that is
the methods point.

Liu-West parameters: 4-D cloud ``(log σ_vR, log σ_vF, log μ, log φ)``.
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
from smc_renewal.pf.model_discrete import (
    ParticleParamsDiscrete,
    ParticleStateDiscrete,
    expected_observation_discrete,
    step_discrete,
)
from smc_renewal.pf.runner_discrete import run_liu_west_discrete


# ---- truth simulation: Model D's own dynamics with forced log_Rt -----------

def simulate_seasonal_discrete(
    key, cfg, log_Rt_seq, truth_log_F_init: float,
    truth_log_sigma_vR: float, truth_log_sigma_vF: float,
    truth_log_mu: float, truth_log_phi: float,
    season_log_I0: float = 2.3,
):
    """Forward-simulate Model D with log_Rt forced and log_F walked.

    log_Rt is overridden to the forced seasonal trajectory at every step
    (this is how we get the multi-season aesthetics).  log_F walks naturally
    via its velocity ``v_F`` driven by ``σ_vF = exp(truth_log_sigma_vF)``.
    v_R "shadow-walks" with ``σ_vR = exp(truth_log_sigma_vR)`` even though
    log_Rt is forced — this gives a well-defined truth value for σ_vR that
    the inference can be compared against.  I_t is Poisson-sampled at each
    step (Model D's discrete renewal).
    """
    T = log_Rt_seq.shape[0]
    k_init, k_steps, k_obs = jr.split(key, 3)

    sigma_vR_truth = jnp.exp(jnp.asarray(truth_log_sigma_vR))
    sigma_vF_truth = jnp.exp(jnp.asarray(truth_log_sigma_vF))

    init_state = ParticleStateDiscrete(
        log_Rt=jnp.asarray(log_Rt_seq[0]),
        v_R=jnp.asarray(0.0),
        log_F=jnp.asarray(truth_log_F_init),
        v_F=jnp.asarray(0.0),
        log_I0=jnp.asarray(season_log_I0),
        I_buf=jnp.full((cfg.buffer_len,), jnp.exp(season_log_I0)),
    )
    truth_params = ParticleParamsDiscrete(
        log_sigma_vR=jnp.asarray(truth_log_sigma_vR),
        log_sigma_vF=jnp.asarray(truth_log_sigma_vF),
        log_mu=jnp.asarray(truth_log_mu),
        log_phi=jnp.asarray(truth_log_phi),
    )

    step_keys = jr.split(k_steps, T)
    eta_R_seq = jr.normal(jr.fold_in(key, 1), (T,))
    eta_F_seq = jr.normal(jr.fold_in(key, 2), (T,))

    def body(state, inputs):
        log_Rt_t, eta_R, eta_F, k = inputs
        # Force log_Rt to the seasonal trajectory; let v_R walk freely
        # ("shadow-walk") with truth σ_vR so the truth has a defined value
        # for that parameter.  v_F and log_F walk naturally.
        forced = state._replace(log_Rt=log_Rt_t)
        new_state = step_discrete(forced, truth_params, eta_R, eta_F, k, cfg)
        mu_y = expected_observation_discrete(new_state, cfg)
        return new_state, (new_state, mu_y)

    _, (traj, mu_y) = jax.lax.scan(
        body, init_state, (log_Rt_seq, eta_R_seq, eta_F_seq, step_keys)
    )
    phi = jnp.exp(jnp.asarray(truth_log_phi))
    y = dist.NegativeBinomial2(mean=mu_y, concentration=phi).sample(k_obs)
    return traj, mu_y, y.astype(jnp.float64)


def _wmean(x, lw):
    w = jnp.exp(lw - jnp.max(lw))
    w = w / w.sum()
    return jnp.sum(w * x)


def main():
    _common.set_clean_style()

    # Truth values.  v_R and v_F walk with these σ values, giving the truth
    # well-defined values for all 4 Liu-West parameters.
    truth_log_F_init = jnp.asarray(-8.0)   # F ≈ 3.4e-4 — example-09/10 scale
    truth_log_sigma_vR = jnp.asarray(-7.0)   # σ_vR ≈ 9e-4 / day
    truth_log_sigma_vF = jnp.asarray(-10.0)  # σ_vF ≈ 4.5e-5 / day (slow F drift)
    truth_log_mu = jnp.log(2.0)              # μ ≈ 2 immigrating cases/day
    truth_log_phi = jnp.asarray(2.5)

    cfg = dataclasses.replace(
        default_config(),
        sigma_floor=1e-6,
        init_log_Rt_sd=0.5,
        init_v_R_mean=0.0,
        init_v_R_sd=0.02,
        init_log_sigma_vR_mean=-6.0,
        init_log_sigma_vR_sd=1.0,
        init_log_F_mean=-8.0,           # prior centred on truth log_F
        init_log_F_sd=1.0,
        init_v_F_mean=0.0,
        init_v_F_sd=1e-4,
        init_log_sigma_vF_mean=-9.0,    # prior matches Model C convention
        init_log_sigma_vF_sd=1.0,
        init_log_mu_mean=0.0,           # prior μ ≈ 1.0 (truth = 2.0, off by ~0.7)
        init_log_mu_sd=0.8,
        prior_log_phi_mean=2.5,
        prior_log_phi_sd=0.5,
    )

    # 4-year horizon with seasonal log_Rt — same amplitude as examples 09/10.
    # F-feedback keeps the discrete renewal bounded, so no runaway growth.
    season_len = 360
    n_seasons = 4
    T = season_len * n_seasons
    t = jnp.arange(T)
    base = 0.45 * jnp.sin(2 * jnp.pi * t / season_len) + 0.02
    raw = 0.06 * jr.normal(jr.key(33), (T,))
    smoothed = jnp.convolve(raw, jnp.ones(7) / 7, mode="same")
    log_Rt_truth = base + smoothed

    print(f"Model D on multi-season data: T={T} days, {n_seasons} seasons")
    print(f"  truth: forced seasonal log_Rt + walked log_F (init {float(truth_log_F_init):.1f}) "
          f"+ Poisson I_t + μ={float(jnp.exp(truth_log_mu)):.2f}")
    print(f"  truth σ_vR={float(jnp.exp(truth_log_sigma_vR)):.2e}, "
          f"σ_vF={float(jnp.exp(truth_log_sigma_vF)):.2e}")
    truth_traj, mu_y_truth, y = simulate_seasonal_discrete(
        jr.key(7), cfg, log_Rt_truth,
        float(truth_log_F_init),
        float(truth_log_sigma_vR), float(truth_log_sigma_vF),
        float(truth_log_mu), float(truth_log_phi),
    )
    I_truth = truth_traj.I_buf[:, 0]
    log_F_truth = truth_traj.log_F

    N = 6000
    h = 0.07
    fixed_lag_L = 21
    print(f"  Liu-West PF (discrete variant): N={N}, h={h}, fixed_lag_L={fixed_lag_L}")
    result = run_liu_west_discrete(
        jr.key(11), cfg, y, n_particles=N, h=h, fixed_lag_L=fixed_lag_L
    )

    lw = result.log_weights_history
    log_Rt_p = result.particles_history.log_Rt
    log_F_p = result.particles_history.log_F
    I_p = result.particles_history.I_buf[:, :, 0]
    log_sigma_vR_p = result.params_history.log_sigma_vR
    log_sigma_vF_p = result.params_history.log_sigma_vF
    log_mu_p = result.params_history.log_mu
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

    # Smoother via particle_smooth.
    n_smooth = 600
    smooth_keys = jr.split(jr.key(99), n_smooth)
    final_lw = lw[-1]
    smooth = lambda arr: jax.vmap(
        lambda k: particle_smooth(k, final_lw, arr, result.ancestors)
    )(smooth_keys)
    log_Rt_qs = jnp.quantile(smooth(log_Rt_p), jnp.array([0.05, 0.5, 0.95]), axis=0)
    log_F_qs = jnp.quantile(smooth(log_F_p), jnp.array([0.05, 0.5, 0.95]), axis=0)
    I_qs = jnp.quantile(smooth(I_p), jnp.array([0.05, 0.5, 0.95]), axis=0)

    cov_logRt = float(jnp.mean((log_Rt_truth >= log_Rt_q[0]) & (log_Rt_truth <= log_Rt_q[2])))
    cov_logRt_s = float(jnp.mean((log_Rt_truth >= log_Rt_qs[0]) & (log_Rt_truth <= log_Rt_qs[2])))
    cov_logF = float(jnp.mean((log_F_truth >= log_F_q[0]) & (log_F_truth <= log_F_q[2])))
    cov_logF_s = float(jnp.mean((log_F_truth >= log_F_qs[0]) & (log_F_truth <= log_F_qs[2])))
    cov_I = float(jnp.mean((I_truth >= I_q[0]) & (I_truth <= I_q[2])))
    cov_I_s = float(jnp.mean((I_truth >= I_qs[0]) & (I_truth <= I_qs[2])))

    print(f"  log_Rt 90% filter cov   {cov_logRt:.3f}, smoother cov {cov_logRt_s:.3f}")
    print(f"  log_F  90% filter cov   {cov_logF:.3f}, smoother cov {cov_logF_s:.3f}")
    print(f"  I(t)    90% filter cov   {cov_I:.3f}, smoother cov {cov_I_s:.3f}")
    print(f"  final log_σ_vR post mean: {float(_wmean(log_sigma_vR_p[-1], final_lw)):+.3f}")
    print(f"  final log_σ_vF post mean: {float(_wmean(log_sigma_vF_p[-1], final_lw)):+.3f}")
    print(f"  final log_μ    post mean: {float(_wmean(log_mu_p[-1], final_lw)):+.3f}  "
          f"(truth {float(truth_log_mu):+.3f})")
    print(f"  final log_φ    post mean: {float(_wmean(log_phi_p[-1], final_lw)):+.3f}  "
          f"(truth {float(truth_log_phi):+.3f})")

    # ---- main figure: 5 panels ----
    fig, axes = plt.subplots(5, 1, figsize=(12, 14), sharex=True,
                             gridspec_kw={"hspace": 0.5})
    season_boundaries = [i * season_len for i in range(1, n_seasons)]

    ax = axes[0]
    ax.plot(t, jnp.maximum(mu_y_truth, 0.5), color="C0", lw=1.0, label="true μ_y(t)")
    ax.scatter(t, np.maximum(np.asarray(y), 0.5), s=4, color="k", alpha=0.5,
               label="observed y")
    for b in season_boundaries:
        ax.axvline(b, color="gray", lw=0.5, ls=":")
    ax.set_yscale("log")
    ax.set_ylabel("cases (log scale)")
    ax.set_title("Model D (discrete Poisson renewal + immigration μ)")
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
    ax.set_title(f"log Rt tracking — filter 90% cov {cov_logRt:.2f}, smoother 90% cov {cov_logRt_s:.2f}")
    ax.legend(loc="upper right", frameon=False, fontsize=8, ncol=3)

    ax = axes[2]
    ax.plot(t, log_F_truth, color="k", lw=1.4, label="truth (constant)")
    ax.fill_between(t, log_F_q[0], log_F_q[2], color="C3", alpha=0.18, label="filter 90%")
    ax.plot(t, log_F_q[1], color="C3", lw=1.0, label="filter median")
    ax.fill_between(t, log_F_qs[0], log_F_qs[2], color="C4", alpha=0.20, label="smoother 90%")
    ax.plot(t, log_F_qs[1], color="C4", lw=1.0, label="smoother median")
    for b in season_boundaries:
        ax.axvline(b, color="gray", lw=0.5, ls=":")
    ax.set_ylabel("log F(t)")
    ax.set_title(f"log F tracking — filter 90% cov {cov_logF:.2f}, smoother 90% cov {cov_logF_s:.2f}")
    ax.legend(loc="upper right", frameon=False, fontsize=8, ncol=3)

    ax = axes[3]
    ax.plot(t, jnp.maximum(I_truth, 0.5), color="k", lw=1.0, label="truth I(t) (Poisson sample)")
    ax.fill_between(t, jnp.maximum(I_q[0], 0.5), jnp.maximum(I_q[2], 0.5),
                    color="C2", alpha=0.18, label="filter 90%")
    ax.plot(t, jnp.maximum(I_q[1], 0.5), color="C2", lw=1.0, label="filter median")
    ax.fill_between(t, jnp.maximum(I_qs[0], 0.5), jnp.maximum(I_qs[2], 0.5),
                    color="C4", alpha=0.20, label="smoother 90%")
    ax.plot(t, jnp.maximum(I_qs[1], 0.5), color="C4", lw=1.0, label="smoother median")
    for b in season_boundaries:
        ax.axvline(b, color="gray", lw=0.5, ls=":")
    ax.set_yscale("log")
    ax.set_ylabel("I(t) (log scale)")
    ax.set_title(f"Latent infections I(t) — discrete Poisson draws  "
                 f"(filter 90% cov {cov_I:.2f}, smoother {cov_I_s:.2f})")
    ax.legend(loc="upper right", frameon=False, fontsize=8, ncol=3)

    # Bottom row: four sub-axes (one per Liu-West parameter), posterior +
    # prior overlay.  Truth lines for μ and φ; σ_vR / σ_vF have no truth
    # value because the truth pinned both velocities to 0.
    axes[4].remove()
    gs_bottom = fig.add_gridspec(5, 1)[4, :].subgridspec(1, 4, wspace=0.35)
    wF = jnp.exp(final_lw - jnp.max(final_lw)); wF = wF / wF.sum()
    panels = [
        ("log σ_vR", log_sigma_vR_p[-1], "C1",
         float(cfg.init_log_sigma_vR_mean), float(cfg.init_log_sigma_vR_sd),
         float(truth_log_sigma_vR)),
        ("log σ_vF", log_sigma_vF_p[-1], "C6",
         float(cfg.init_log_sigma_vF_mean), float(cfg.init_log_sigma_vF_sd),
         float(truth_log_sigma_vF)),
        ("log μ", log_mu_p[-1], "C5",
         float(cfg.init_log_mu_mean), float(cfg.init_log_mu_sd), float(truth_log_mu)),
        ("log φ", log_phi_p[-1], "C2",
         float(cfg.prior_log_phi_mean), float(cfg.prior_log_phi_sd), float(truth_log_phi)),
    ]
    for i, (name, arr, color, p_mu, p_sd, truth_val) in enumerate(panels):
        a = fig.add_subplot(gs_bottom[0, i])
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
        a.set_title(f"final-step posterior on {name}", fontsize=10)
        a.tick_params(axis="y", labelleft=False, left=False)
        if i == 0:
            a.legend(loc="upper right", frameon=False, fontsize=7)

    fig.suptitle(
        "Model D — discrete (Poisson) renewal with immigration μ; "
        "Liu-West on (log σ_vR, log μ, log φ)",
        y=0.998,
    )
    out = _common.save(fig, "11_model_d_discrete.png")
    print(f"saved: {out}")

    # ---- sequential forecasts ----
    h_max = 60
    origins = [60 + i * season_len for i in range(n_seasons)]
    origins = [o for o in origins if o + h_max <= T]
    print(f"\nsequential forecasts at origins {origins}, horizon {h_max}d:")

    def forecast_from_cloud_discrete(key, cfg, particles, params, lw_orig, h_max):
        N_ = lw_orig.shape[0]
        k_noise, k_pois_outer, k_obs = jr.split(key, 3)
        eta_seq = jr.normal(k_noise, (h_max, N_, 2))    # (eta_R, eta_F)
        pois_keys = jr.split(k_pois_outer, h_max * N_).reshape(h_max, N_)

        def body(state, ix):
            eta_t, k_t = ix

            def per_particle(s, p, e, k):
                return step_discrete(s, p, e[0], e[1], k, cfg)

            new_state = jax.vmap(per_particle)(state, params, eta_t, k_t)
            mu_y = jax.vmap(lambda s: expected_observation_discrete(s, cfg))(new_state)
            return new_state, mu_y

        _, mu_y_traj = jax.lax.scan(body, particles, (eta_seq, pois_keys))
        phi = jnp.exp(params.log_phi)
        y_traj = dist.NegativeBinomial2(
            mean=mu_y_traj, concentration=phi[None, :]
        ).sample(k_obs)
        return mu_y_traj, y_traj.astype(jnp.float64)

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

        _mu, y_fc = forecast_from_cloud_discrete(
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
        f"Model D sequential forecasts (50% and 95% PIs, {h_max}-day horizon) "
        f"— integrated trend + Poisson scatter"
    )
    ax.legend(loc="upper right", frameon=False, fontsize=9, ncol=2)
    out2 = _common.save(fig2, "11_model_d_forecasts.png")
    print(f"saved: {out2}")


if __name__ == "__main__":
    main()
