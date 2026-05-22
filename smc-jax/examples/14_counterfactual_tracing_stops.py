"""14 — Counterfactual: what if contact tracing stopped on day 25?

Loads the locked example-12 dataset (50-day single-seed outbreak under
contact tracing) and *re-runs the world* from day 25 with one twist:
**contact tracing stops** — newly observed cases are still counted, but
they no longer leave the renewal pool, so the depletion that was bending
the curve disappears.

We then ask each model what it forecasts for the next 14 days under this
counterfactual:

  1. **Truth (counterfactual)**: re-simulate the truth from day 25
     forward with no removal from the renewal pool.  Latent infections
     explode.
  2. **Model E forecast (counterfactual)**: extrapolate Model E's
     particle cloud at day 25 with contact-tracing depletion *turned off*.
     Should track the truth explosion (Model E knows the mechanism).
  3. **Model C forecast (status quo)**: Model C has no contact-tracing
     concept, so its forecast does not change — it predicts continued
     decline because its inferred Rt is falling.

Saves the counterfactual trajectory to `examples/data/14_counterfactual.npz`
(does *not* modify `12_truth.npz`).
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
import numpyro.distributions as dist
from jax.scipy.stats import norm

from smc_renewal.config import default_config
from smc_renewal.pf.bootstrap import weighted_quantile
from smc_renewal.pf.model_gdm import (
    ParticleParamsGDM,
    ParticleStateGDM,
    forward_step,
)
from smc_renewal.pf.model_trend import (
    ParticleParamsTrend,
    ParticleStateTrend,
    TransitionNoiseTrend,
    expected_observation_trend,
    step_trend,
)
from smc_renewal.pf.runner_gdm import run_liu_west_gdm
from smc_renewal.pf.runner_trend import run_liu_west_trend
from smc_renewal.transition import _pad_pmf


TRUTH_NPZ = Path(__file__).parent / "data" / "12_truth.npz"
CF_NPZ = Path(__file__).parent / "data" / "14_counterfactual.npz"


def gdm_marginal_delay_pmf(b_0: float, b_1: float, L: int) -> jnp.ndarray:
    """GDM marginal-lag PMF (same helper as example 13)."""
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


def no_tracing_forward_sim(
    key, log_Rt0, v_R0, I_buf0, sigma_vR, h_max, cfg,
):
    """Forward-simulate ``h_max`` days with **no contact-tracing depletion**.

    Renewal kernel reads from the cumulative-infectious buffer ``I_buf`` and
    no entry ever decreases — every new infection stays in the renewal
    window for L days, then ages out via the generation interval.

    Returns the ``N_t`` time series (shape (h_max,)).
    """
    g_pad = _pad_pmf(cfg.generation_interval, cfg.buffer_len)
    keys = jr.split(key, h_max)

    def body(state, k):
        log_Rt, v_R, I_buf = state
        k_eta, k_pois = jr.split(k, 2)
        eta = jr.normal(k_eta)
        log_Rt_new = log_Rt + v_R
        v_R_new = v_R + sigma_vR * eta
        g_conv = jnp.dot(g_pad, I_buf.astype(jnp.float64))
        Rt_eff = jnp.exp(jnp.clip(log_Rt_new, -20.0, 20.0))
        lambda_t = jnp.clip(Rt_eff * g_conv, 1e-12, 1e12)
        N_t = jr.poisson(k_pois, lambda_t).astype(jnp.int64)
        I_buf_new = jnp.concatenate([N_t[None], I_buf[:-1]])
        return (log_Rt_new, v_R_new, I_buf_new), N_t

    _, N_traj = jax.lax.scan(body, (log_Rt0, v_R0, I_buf0), keys)
    return N_traj


def no_tracing_forecast_cloud(key, particles, params, h_max, cfg):
    """Vectorised over particles: no-contact-tracing forward sim from each
    particle's day-25 state.  ``particles.U_buf`` is used as the initial
    in-circulation buffer (it's what's still infectious at day 25; pre-day-25
    tracing removals stay removed)."""
    g_pad = _pad_pmf(cfg.generation_interval, cfg.buffer_len)
    N_ = particles.log_Rt.shape[0]
    keys_outer = jr.split(key, h_max)

    def body(state, k_t):
        log_Rt, v_R, I_buf = state
        keys_per = jr.split(k_t, N_)
        sigma_vR = jnp.exp(params.log_sigma_vR)

        def per_p(k, lR, vR, Ib, sR):
            keta, kp = jr.split(k, 2)
            eta = jr.normal(keta)
            lR_new = lR + vR
            vR_new = vR + sR * eta
            g_conv = jnp.dot(g_pad, Ib.astype(jnp.float64))
            Rt_eff = jnp.exp(jnp.clip(lR_new, -20.0, 20.0))
            lam = jnp.clip(Rt_eff * g_conv, 1e-12, 1e12)
            N_t = jr.poisson(kp, lam).astype(jnp.int64)
            Ib_new = jnp.concatenate([N_t[None], Ib[:-1]])
            return lR_new, vR_new, Ib_new, N_t

        lR_n, vR_n, Ib_n, N_t = jax.vmap(per_p)(
            keys_per, log_Rt, v_R, I_buf, sigma_vR
        )
        return (lR_n, vR_n, Ib_n), N_t

    init_state = (particles.log_Rt, particles.v_R, particles.U_buf.astype(jnp.int64))
    _, N_traj = jax.lax.scan(body, init_state, keys_outer)
    return N_traj  # (h_max, N)


def model_c_forecast_cloud(key, particles, params, h_max, cfg):
    """Model C's standard forecast (delay-conv + NegBin).  Returns y_t
    trajectory (h_max, N)."""
    k_noise, k_obs = jr.split(key, 2)
    N_ = particles.log_Rt.shape[0]
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
    return y_traj.astype(jnp.float64)


def _wmean(x, lw):
    w = jnp.exp(lw - jnp.max(lw))
    w = w / w.sum()
    return jnp.sum(w * x)


def main():
    _common.set_clean_style()

    if not TRUTH_NPZ.exists():
        raise FileNotFoundError(
            f"Run examples/12_model_d_gdm.py first to generate {TRUTH_NPZ}."
        )

    # ----- Load locked Model E truth -----
    data = np.load(TRUTH_NPZ)
    y = jnp.asarray(data["y"], dtype=jnp.float64)
    log_Rt_truth = jnp.asarray(data["log_Rt_truth"])
    I_truth = jnp.asarray(data["I_truth"])
    T = int(data["T"])
    truth_log_Rt_init = float(data["truth_log_Rt_init"])
    truth_v_R_init = float(data["truth_v_R_init"])
    truth_log_I0 = float(data["truth_log_I0"])
    truth_log_sigma_vR = float(data["truth_log_sigma_vR"])
    truth_b_0 = float(data["truth_b_0"])
    truth_b_1 = float(data["truth_b_1"])
    truth_log_M = float(data["truth_log_M"])
    ascertainment_alpha = float(data["ascertainment_alpha"])
    print(f"Loaded locked Model E truth from {TRUTH_NPZ}")
    print(f"  T={T}, peak y_t={int(y.max())}")

    hinge_day = 25       # tracing stops at end of day 25
    h_max = 14           # forecast horizon

    # ----- Re-simulate the truth deterministically to recover day-25 state -----
    cfg_e = dataclasses.replace(default_config(), ascertainment_alpha=ascertainment_alpha)
    truth_params_e = ParticleParamsGDM(
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
        U_buf=jnp.full((cfg_e.buffer_len,), I0),
    )
    # truth seed used in example 12 was jr.key(0) when seed=0 was the first
    # accepted in the for-loop.  Replay it.
    truth_keys = jr.split(jr.key(0), T)

    def truth_body(state, k):
        new_state, y_step, O_step = forward_step(k, state, truth_params_e, cfg_e)
        return new_state, new_state

    _, truth_traj_state = jax.lax.scan(truth_body, init_state, truth_keys)
    log_Rt_at_hinge = truth_traj_state.log_Rt[hinge_day - 1]
    v_R_at_hinge = truth_traj_state.v_R[hinge_day - 1]
    U_buf_at_hinge = truth_traj_state.U_buf[hinge_day - 1]
    print(f"  state at end of day {hinge_day}: log Rt={float(log_Rt_at_hinge):+.3f}, "
          f"v_R={float(v_R_at_hinge):+.4f}, Σ U_buf={int(jnp.sum(U_buf_at_hinge))}")

    # ----- Counterfactual truth: no contact tracing from day 25 onwards -----
    sigma_vR_truth = jnp.exp(jnp.asarray(truth_log_sigma_vR))
    truth_cf_keys = jr.fold_in(jr.key(0), 9999)
    N_cf_truth = no_tracing_forward_sim(
        truth_cf_keys, log_Rt_at_hinge, v_R_at_hinge, U_buf_at_hinge.astype(jnp.int64),
        sigma_vR_truth, h_max, cfg_e,
    )
    print(f"  counterfactual truth N_t: day {hinge_day+1}={int(N_cf_truth[0])} → "
          f"day {hinge_day+h_max}={int(N_cf_truth[-1])}")

    np.savez(
        CF_NPZ,
        N_cf_truth=np.asarray(N_cf_truth),
        hinge_day=hinge_day,
        h_max=h_max,
        log_Rt_at_hinge=float(log_Rt_at_hinge),
        v_R_at_hinge=float(v_R_at_hinge),
        U_buf_at_hinge=np.asarray(U_buf_at_hinge),
    )
    print(f"  saved counterfactual truth → {CF_NPZ}")

    # ----- Re-run Model E PF on the locked data, get cloud at day-25 -----
    cfg_e_full = dataclasses.replace(
        cfg_e,
        sigma_floor=1e-6,
        init_log_Rt_mean=0.3, init_log_Rt_sd=0.4,
        init_v_R_mean=0.0, init_v_R_sd=0.04,
        init_log_sigma_vR_mean=-5.0, init_log_sigma_vR_sd=1.0,
        init_log_I0_mean=1.0, init_log_I0_sd=0.5,
        init_b0_mean=-1.0, init_b0_sd=1.0,
        init_b1_mean=0.2, init_b1_sd=0.3,
        init_log_M_mean=3.0, init_log_M_sd=0.7,
    )
    print(f"\nRunning Model E PF on locked data (N=8000)…")
    result_e = run_liu_west_gdm(
        jr.key(11), cfg_e_full, y, n_particles=8000, h=0.04, fixed_lag_L=14,
    )
    particles_e = jax.tree.map(lambda x: x[hinge_day - 1], result_e.particles_history)
    params_e = jax.tree.map(lambda x: x[hinge_day - 1], result_e.params_history)
    lw_e = result_e.log_weights_history[hinge_day - 1]
    print(f"  Model E day-{hinge_day} cloud extracted; ESS = "
          f"{float(jnp.exp(-jax.scipy.special.logsumexp(2.0*(lw_e - jax.scipy.special.logsumexp(lw_e))))):.1f}")

    # ----- Run Model C PF on locked data, get cloud at day-25 -----
    L_delay = int(default_config().delay_pmf.shape[0])
    matched_delay = gdm_marginal_delay_pmf(truth_b_0, truth_b_1, L_delay)
    cfg_c = dataclasses.replace(
        default_config(),
        delay_pmf=matched_delay,
        sigma_floor=1e-6,
        init_log_Rt_mean=0.3, init_log_Rt_sd=0.4,
        init_v_R_mean=0.0, init_v_R_sd=0.04,
        init_log_sigma_vR_mean=-5.0, init_log_sigma_vR_sd=1.0,
        init_log_F_mean=-12.0, init_log_F_sd=1.0,
        init_v_F_mean=0.0, init_v_F_sd=1e-4,
        init_log_sigma_vF_mean=-10.0, init_log_sigma_vF_sd=1.0,
        init_log_I0_mean=1.5, init_log_I0_sd=0.5,
        prior_log_phi_mean=2.0, prior_log_phi_sd=1.0,
    )
    print(f"Running Model C PF on locked data (N=8000)…")
    result_c = run_liu_west_trend(
        jr.key(11), cfg_c, y, n_particles=8000, h=0.04, fixed_lag_L=14,
    )
    particles_c = jax.tree.map(lambda x: x[hinge_day - 1], result_c.particles_history)
    params_c = jax.tree.map(lambda x: x[hinge_day - 1], result_c.params_history)
    lw_c = result_c.log_weights_history[hinge_day - 1]

    # ----- Forecasts -----
    print(f"\nForecasting {h_max} days from day {hinge_day}…")
    # Model E under counterfactual (no tracing from day 25):
    N_fc_e = no_tracing_forecast_cloud(
        jr.key(77), particles_e, params_e, h_max, cfg_e_full,
    )  # (h_max, N)
    # Model C under its own (status-quo) dynamics:
    y_fc_c = model_c_forecast_cloud(
        jr.key(78), particles_c, params_c, h_max, cfg_c,
    )  # (h_max, N)

    qs5 = (0.025, 0.25, 0.5, 0.75, 0.975)
    Q_e = jnp.stack([
        jax.vmap(weighted_quantile, in_axes=(0, None, None))(N_fc_e, lw_e, jnp.asarray(q))
        for q in qs5
    ])
    Q_c = jnp.stack([
        jax.vmap(weighted_quantile, in_axes=(0, None, None))(y_fc_c, lw_c, jnp.asarray(q))
        for q in qs5
    ])

    # ----- Plot -----
    fig, ax = plt.subplots(1, 1, figsize=(13, 6))
    t_obs = np.arange(T)
    t_cf = hinge_day + np.arange(1, h_max + 1)
    y_np = np.asarray(y).astype(int)

    # Observed up to hinge
    ax.plot(t_obs[:hinge_day], np.maximum(y_np[:hinge_day], 0.5), "o-",
            color="k", lw=0.7, ms=3, alpha=0.7,
            label="observed y_t (days 1–25, tracing active)")

    # Truth counterfactual after hinge
    ax.plot(t_cf, np.maximum(np.asarray(N_cf_truth), 0.5), "o-",
            color="k", lw=1.4, ms=4,
            label="truth if tracing stops at day 25 (latent N_t)")

    # Model E counterfactual forecast
    ax.fill_between(t_cf, jnp.maximum(Q_e[0], 0.5), jnp.maximum(Q_e[4], 0.5),
                    color="C2", alpha=0.15)
    ax.fill_between(t_cf, jnp.maximum(Q_e[1], 0.5), jnp.maximum(Q_e[3], 0.5),
                    color="C2", alpha=0.30)
    ax.plot(t_cf, jnp.maximum(Q_e[2], 0.5), color="C2", lw=1.8,
            label="Model E counterfactual forecast (knows tracing)")

    # Model C forecast (status quo)
    ax.fill_between(t_cf, jnp.maximum(Q_c[0], 0.5), jnp.maximum(Q_c[4], 0.5),
                    color="C3", alpha=0.15)
    ax.fill_between(t_cf, jnp.maximum(Q_c[1], 0.5), jnp.maximum(Q_c[3], 0.5),
                    color="C3", alpha=0.30)
    ax.plot(t_cf, jnp.maximum(Q_c[2], 0.5), color="C3", lw=1.8,
            label="Model C forecast (no tracing in model)")

    ax.axvline(hinge_day, color="gray", lw=1.0, ls="--", alpha=0.7)
    ax.text(hinge_day + 0.3, 4e4, "tracing\nstops",
            ha="left", va="top", fontsize=9, color="gray")

    # Annotate the day-39 upper-95% bounds — the operational "what's the
    # worst case?" signal each model sends to public health.
    upper_e = float(Q_e[4, -1])
    upper_c = float(Q_c[4, -1])
    truth_end = float(N_cf_truth[-1])
    ax.annotate(
        f"Model E 95% upper: {upper_e:,.0f}",
        xy=(t_cf[-1] + 0.2, upper_e),
        xytext=(t_cf[-1] - 8, 3e5),
        fontsize=9, color="C2",
        arrowprops=dict(arrowstyle="-", color="C2", alpha=0.6, lw=0.7),
    )
    ax.annotate(
        f"truth: {truth_end:.0f}",
        xy=(t_cf[-1] + 0.2, truth_end),
        xytext=(t_cf[-1] - 4, 9e3),
        fontsize=9, color="k",
        arrowprops=dict(arrowstyle="-", color="k", alpha=0.5, lw=0.7),
    )
    ax.annotate(
        f"Model C 95% upper: {upper_c:.0f}",
        xy=(t_cf[-1] + 0.2, upper_c),
        xytext=(t_cf[-1] - 8, 50),
        fontsize=9, color="C3",
        arrowprops=dict(arrowstyle="-", color="C3", alpha=0.6, lw=0.7),
    )

    ax.set_yscale("log")
    ax.set_ylim(0.5, 1e6)
    ax.set_xlabel("day")
    ax.set_ylabel("daily cases (log scale)")
    ax.set_title(
        f"Counterfactual: what if contact tracing stopped at day {hinge_day}?  "
        f"(14-day horizon, 50% and 95% PIs)"
    )
    ax.legend(loc="upper left", frameon=False, fontsize=9)

    out = _common.save(fig, "14_counterfactual_tracing_stops.png")
    print(f"\nsaved: {out}")
    print(f"  truth at day {hinge_day + h_max}: {int(N_cf_truth[-1])} cases/day")
    print(f"  Model E forecast 95% upper at day {hinge_day + h_max}: {upper_e:.0f}")
    print(f"  Model C forecast 95% upper at day {hinge_day + h_max}: {upper_c:.0f}")


if __name__ == "__main__":
    main()
