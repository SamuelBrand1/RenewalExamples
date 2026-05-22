"""06 — Rolling-origin forecast using SMC² + cuthbert EKF, same data as 04.

Companion to ``04_rolling_origin_forecast.py``.  Uses the same reference
series and forecast horizons; swaps the Liu-West bootstrap PF for SMC²-
with-Gaussian-inner.

Procedure per origin t:
    1. Fit (or sequentially extend) SMC²+EKF on ``y[:t]`` → ``N_theta`` θ-particles.
    2. For each θ-particle: run the EKF on ``y[:t]``, draw ONE latent state
       from the terminal filtered Gaussian, then forward-simulate ``h_max``
       days via ``transition.step`` with fresh noise.
    3. Sample NegBin observations; score against ``y[t:t+h_max]``.

The forecast cloud has size ``N_theta`` (here 128) — substantially smaller
than the PF's ``N_particles=2500`` in example 04 — so CRPS estimates here
are noisier per origin.
"""

from __future__ import annotations

import _common  # noqa: F401
import jax
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np
import numpyro.distributions as dist

from smc_renewal.config import default_config
from smc_renewal.scoring import (
    crps_weighted_batch,
    interval_coverage,
    weighted_quantile,
)
from smc_renewal.smc2.ekf_cuthbert import filter_trajectory
from smc_renewal.smc2.ekf_cuthbert import marginal_log_likelihood as ekf_loglik
from smc_renewal.smc2.runner import extend_fit, fit_smc2
from smc_renewal.state import (
    ParticleParams,
    ParticleState,
    state_dim,
    unpack,
)
from smc_renewal.synthetic import simulate
from smc_renewal.transition import (
    TransitionNoise,
    expected_observation,
    step,
)


def _sample_state_from_ekf(
    key, cfg, y_so_far, params: ParticleParams
) -> ParticleState:
    """Run EKF on ``y_so_far``, sample ONE state from the terminal Gaussian."""
    means, chol_covs = filter_trajectory(cfg, y_so_far, params)
    m = means[-1]
    L = chol_covs[-1]
    D = state_dim(cfg.buffer_len)
    z = jr.normal(key, (D,))
    flat = m + L @ z
    # Buffer entries can go negative under Gaussian sampling — clamp to 0.
    return ParticleState(
        log_Rt=flat[0],
        log_sigma_R=flat[1],
        log_F=flat[2],
        log_sigma_F=flat[3],
        log_I0=flat[4],
        I_buf=jnp.maximum(flat[5:], 0.0),
    )


def _forecast_from_theta_cloud(
    key, cfg, y_so_far, theta_particles, h_max
):
    """For each θ-particle: sample a state, forecast h_max days, sample y."""
    N = theta_particles.shape[0]
    k_state, k_noise, k_obs = jr.split(key, 3)
    state_keys = jr.split(k_state, N)

    params_per = ParticleParams(
        log_tau_R=theta_particles[:, 0],
        log_tau_F=theta_particles[:, 1],
        log_phi=theta_particles[:, 2],
    )

    def sample_one(k, th_packed):
        th = ParticleParams(log_tau_R=th_packed[0], log_tau_F=th_packed[1], log_phi=th_packed[2])
        return _sample_state_from_ekf(k, cfg, y_so_far, th)

    states0 = jax.vmap(sample_one)(state_keys, theta_particles)

    # Forward-simulate h_max days, vmapped over N θ-particles.
    all_noise = jr.normal(k_noise, (h_max, N, 4))

    def body(state, n_t):
        noises = TransitionNoise(
            eps_R=n_t[:, 0], eta_R=n_t[:, 1], eps_F=n_t[:, 2], eta_F=n_t[:, 3]
        )
        new_state = jax.vmap(lambda s, p, n: step(s, p, n, cfg))(state, params_per, noises)
        mu_y = jax.vmap(lambda s: expected_observation(s, cfg))(new_state)
        return new_state, mu_y

    _, mu_y_traj = jax.lax.scan(body, states0, all_noise)
    phi = jnp.exp(params_per.log_phi)
    y_traj = dist.NegativeBinomial2(mean=mu_y_traj, concentration=phi[None, :]).sample(k_obs)
    return mu_y_traj, y_traj.astype(jnp.float64)


def main():
    _common.set_clean_style()
    cfg = default_config()
    T = 180
    H_MAX = 28
    INITIAL_T0 = 80
    STEP = 14
    N_THETA = 128

    truth = ParticleParams(
        log_tau_R=jnp.asarray(-4.0),
        log_tau_F=jnp.asarray(-12.0),
        log_phi=jnp.asarray(2.5),
    )
    ds = simulate(jr.key(2), cfg, T, params=truth)

    # Build rolling origins.
    origins = []
    t = INITIAL_T0
    while t + H_MAX <= T:
        origins.append(t)
        t += STEP
    origins_to_plot = [origins[0], origins[len(origins) // 2], origins[-1]]
    print(f"running SMC²+EKF on {len(origins)} origins: {origins}")

    # Fit once at the first origin, then extend.
    key = jr.key(101)
    key, k_fit = jr.split(key, 2)
    cache = fit_smc2(
        k_fit, cfg, ds.y[:origins[0]],
        n_particles=N_THETA, num_mcmc_steps=3, rwmh_scale=0.4,
        max_outer_steps=15, marginal_loglik_fn=ekf_loglik,
    )

    fans = {}  # origin -> y_traj (h_max, N)
    crps_rows, cov50_rows, cov95_rows = [], [], []

    for i, t0 in enumerate(origins):
        if i > 0:
            key, k_ext = jr.split(key, 2)
            cache = extend_fit(
                k_ext, cfg, ds.y[:t0], cache,
                marginal_loglik_fn=ekf_loglik,
            )
        key, k_fc = jr.split(key, 2)
        mu_y_traj, y_traj = _forecast_from_theta_cloud(
            k_fc, cfg, ds.y[:t0], cache.result.particles, H_MAX
        )
        if t0 in origins_to_plot:
            fans[t0] = y_traj

        y_true = ds.y[t0:t0 + H_MAX]
        lw = jnp.full((N_THETA,), -jnp.log(N_THETA))  # equal-weight cloud
        crps = crps_weighted_batch(y_true, y_traj, lw)
        c50 = interval_coverage(y_true, y_traj, lw, level=0.5)
        c95 = interval_coverage(y_true, y_traj, lw, level=0.95)
        crps_rows.append(crps); cov50_rows.append(c50); cov95_rows.append(c95)

    crps = jnp.stack(crps_rows); c50 = jnp.stack(cov50_rows); c95 = jnp.stack(cov95_rows)
    summary = {h: dict(
        crps_mean=float(jnp.mean(crps[:, h - 1])),
        cov50_mean=float(jnp.mean(c50[:, h - 1])),
        cov95_mean=float(jnp.mean(c95[:, h - 1])),
    ) for h in (7, 14, 21, 28)}

    print("horizon-wise summary:")
    for h, s in summary.items():
        print(f"  {h:>3}d  CRPS={s['crps_mean']:>10.2f}  cov50={s['cov50_mean']:.2f}  cov95={s['cov95_mean']:.2f}")

    # --- plotting (same layout as example 04) ---
    fig = plt.figure(figsize=(12, 7.5))
    gs = fig.add_gridspec(2, 3, height_ratios=[2.2, 1.0], hspace=0.4, wspace=0.3)
    for idx, t0 in enumerate(origins_to_plot):
        ax = fig.add_subplot(gs[0, idx])
        tt = np.arange(T)
        ax.plot(tt, ds.y, color="k", lw=0.8, alpha=0.5, label="observed cases (full)")
        ax.scatter(tt[:t0], np.asarray(ds.y[:t0]), s=8, color="k", alpha=0.6)
        h_t = t0 + 1 + np.arange(H_MAX)
        qs = (0.025, 0.25, 0.5, 0.75, 0.975)
        lw = jnp.full((N_THETA,), -jnp.log(N_THETA))
        Q = jnp.stack([
            jax.vmap(weighted_quantile, in_axes=(0, None, None))(fans[t0], lw, jnp.asarray(q))
            for q in qs
        ])
        ax.fill_between(h_t, Q[0], Q[4], color="C4", alpha=0.15, label="95% PI")
        ax.fill_between(h_t, Q[1], Q[3], color="C4", alpha=0.30, label="50% PI")
        ax.plot(h_t, Q[2], color="C4", lw=1.2, label="median")
        ax.axvline(t0, color="r", lw=0.7, ls="--")
        ax.set_title(f"forecast from day {t0}")
        ax.set_xlabel("day"); ax.set_ylabel("cases")
        if idx == 0:
            ax.legend(loc="upper left", frameon=False, fontsize=8)

    horizons = np.arange(1, H_MAX + 1)
    ax = fig.add_subplot(gs[1, 0])
    ax.semilogy(horizons, np.maximum(np.asarray(crps.mean(0)), 1e-2), color="C4")
    ax.set_title("Mean CRPS vs horizon")
    ax.set_xlabel("horizon (days)"); ax.set_ylabel("CRPS (log)")

    ax = fig.add_subplot(gs[1, 1])
    ax.plot(horizons, np.asarray(c50.mean(0)), color="C2", label="50% PI")
    ax.axhline(0.5, color="C2", lw=0.5, ls="--")
    ax.plot(horizons, np.asarray(c95.mean(0)), color="C4", label="95% PI")
    ax.axhline(0.95, color="C4", lw=0.5, ls="--")
    ax.set_ylim(0, 1.05)
    ax.set_title("Interval coverage vs horizon")
    ax.set_xlabel("horizon (days)"); ax.set_ylabel("coverage")
    ax.legend(loc="lower right", frameon=False, fontsize=8)

    ax = fig.add_subplot(gs[1, 2])
    ax.axis("off")
    table = "horizon  CRPS    cov50  cov95\n" + "\n".join(
        f"  {h:>3}d  {summary[h]['crps_mean']:>8.2f}  {summary[h]['cov50_mean']:.2f}   {summary[h]['cov95_mean']:.2f}"
        for h in (7, 14, 21, 28)
    )
    ax.text(0.0, 0.95, "headline horizons", fontsize=10, fontweight="bold",
            family="monospace", transform=ax.transAxes, va="top")
    ax.text(0.0, 0.80, table, fontsize=9, family="monospace",
            transform=ax.transAxes, va="top")

    fig.suptitle(
        "Rolling-origin forecast — SMC² + cuthbert EKF, weekly sequential updates",
        y=1.02,
    )
    out = _common.save(fig, "06_smc2_ekf_rolling_origin.png")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
