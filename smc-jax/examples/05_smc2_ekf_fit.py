"""05 — SMC² with cuthbert EKF inner, on the same synthetic data as example 02.

Companion to ``02_pf_fit.py``.  The same reference series (``seed=2``, T=180)
is fit with the SMC²-with-Gaussian-inner approximation instead of the
Liu-West bootstrap PF.  Visualises:

    1. EKF filtered band for log Rt(t) and σ_R(t) at the SMC² posterior-mean θ
       (with truth overlaid).
    2. SMC² adaptive-tempering trajectory (β over outer steps).
    3. Marginal posteriors over (log_tau_R, log_tau_F, log_phi) — note the
       collapse to the prior on the tau dimensions: this is the Gaussian-
       filter blind spot from PYRENEW_FRICTION.md, made visible.

The Gaussian filter cannot identify ``log_tau_R`` or ``log_tau_F`` (the
nested-volatility parameters) because the chain ``tau_R → log_sigma_R
variance → sigma_R via exp → log_Rt`` decouples at the linearization point.
Compare to example 02, where the Liu-West PF posterior is concentrated
near the truth on all three parameters.
"""

from __future__ import annotations

import _common  # noqa: F401
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np

from smc_renewal.config import default_config
from smc_renewal.smc2.ekf_cuthbert import filter_trajectory
from smc_renewal.smc2.ekf_cuthbert import marginal_log_likelihood as ekf_loglik
from smc_renewal.smc2.runner import fit_smc2
from smc_renewal.state import ParticleParams, unpack
from smc_renewal.synthetic import simulate


def main():
    _common.set_clean_style()
    cfg = default_config()
    T = 180
    truth = ParticleParams(
        log_tau_R=jnp.asarray(-4.0),
        log_tau_F=jnp.asarray(-12.0),
        log_phi=jnp.asarray(2.5),
    )
    ds = simulate(jr.key(2), cfg, T, params=truth)
    print(f"running SMC² + cuthbert EKF: T={T}, N=128, ≤20 outer steps")

    cache = fit_smc2(
        jr.key(11), cfg, ds.y,
        n_particles=128, num_mcmc_steps=4, rwmh_scale=0.5,
        max_outer_steps=20,
        marginal_loglik_fn=ekf_loglik,
    )
    print(f"  SMC² ran {cache.result.n_steps} adaptive-tempering steps")

    theta_particles = cache.result.particles  # (128, 3), equal weights
    names = ["log_tau_R", "log_tau_F", "log_phi"]
    truths = [-4.0, -12.0, 2.5]
    prior_means = [cfg.prior_log_tau_R_mean, cfg.prior_log_tau_F_mean, cfg.prior_log_phi_mean]
    prior_sds = [cfg.prior_log_tau_R_sd, cfg.prior_log_tau_F_sd, cfg.prior_log_phi_sd]

    # Run a single EKF at the SMC² posterior mean θ for the filtered Rt trajectory.
    th_mean = jnp.mean(theta_particles, axis=0)
    th_mean_params = ParticleParams(
        log_tau_R=th_mean[0], log_tau_F=th_mean[1], log_phi=th_mean[2]
    )
    means, chol_covs = filter_trajectory(cfg, ds.y, th_mean_params)
    # cuthbert returns T+1 entries (initial + T observations); drop the prior at index 0.
    means_t = means[1:]               # (T, D)
    chol_covs_t = chol_covs[1:]       # (T, D, D)
    # log_Rt is at flat index 0; log_sigma_R at index 1.
    sigma_log_Rt = jnp.sqrt(jnp.einsum("tij,tij->t", chol_covs_t[:, 0:1, :], chol_covs_t[:, 0:1, :]))
    sigma_log_sR = jnp.sqrt(jnp.einsum("tij,tij->t", chol_covs_t[:, 1:2, :], chol_covs_t[:, 1:2, :]))
    log_Rt_mean = means_t[:, 0]
    log_sigma_R_mean = means_t[:, 1]
    log_Rt_lo = log_Rt_mean - 1.645 * sigma_log_Rt
    log_Rt_hi = log_Rt_mean + 1.645 * sigma_log_Rt
    log_sigma_R_lo = log_sigma_R_mean - 1.645 * sigma_log_sR
    log_sigma_R_hi = log_sigma_R_mean + 1.645 * sigma_log_sR

    in_band = (ds.log_Rt >= log_Rt_lo) & (ds.log_Rt <= log_Rt_hi)
    print(f"EKF log_Rt 90% band coverage: {float(jnp.mean(in_band)):.3f}")

    t = jnp.arange(T)
    fig = plt.figure(figsize=(12, 8.5))
    gs = fig.add_gridspec(3, 3, height_ratios=[1.4, 1.0, 1.0], hspace=0.55, wspace=0.30)

    ax = fig.add_subplot(gs[0, :2])
    ax.fill_between(t, log_Rt_lo, log_Rt_hi, color="C4", alpha=0.25, label="EKF 90%")
    ax.plot(t, log_Rt_mean, color="C4", lw=1.0, label="EKF mean")
    ax.plot(t, ds.log_Rt, color="k", lw=1.4, label="truth")
    ax.set_title("log Rt(t) — EKF at SMC² posterior mean θ")
    ax.set_xlabel("day"); ax.set_ylabel("log Rt")
    ax.legend(loc="lower left", frameon=False, fontsize=9)

    ax = fig.add_subplot(gs[0, 2])
    ax.fill_between(t, jnp.exp(log_sigma_R_lo), jnp.exp(log_sigma_R_hi), color="C1", alpha=0.25)
    ax.plot(t, jnp.exp(log_sigma_R_mean), color="C1", label="EKF mean")
    ax.plot(t, jnp.exp(ds.log_sigma_R), color="k", lw=1.2, label="truth")
    ax.set_title("σ_R(t) — EKF")
    ax.set_xlabel("day"); ax.set_ylabel("σ_R")
    ax.legend(loc="upper left", frameon=False, fontsize=8)

    # Tempering trajectory note (we don't expose intermediate β's; this panel
    # just records the final outcome).
    ax = fig.add_subplot(gs[1, :])
    ax.axis("off")
    ax.text(
        0.0, 0.85,
        "SMC² adaptive tempering",
        fontsize=11, fontweight="bold", transform=ax.transAxes,
    )
    ax.text(
        0.0, 0.55,
        f"  outer steps:    {cache.result.n_steps}\n"
        f"  log marginal:   {float(cache.result.log_marginal):.3f}\n"
        f"  θ-particles:    {theta_particles.shape[0]}\n"
        f"  inner kernel:   cuthbert.gaussian.moments EKF",
        fontsize=10, family="monospace", transform=ax.transAxes, va="top",
    )
    ax.text(
        0.55, 0.85,
        "Posterior means vs truth",
        fontsize=11, fontweight="bold", transform=ax.transAxes,
    )
    posterior_lines = []
    for i, (name, tr) in enumerate(zip(names, truths)):
        m = float(jnp.mean(theta_particles[:, i]))
        sd = float(jnp.std(theta_particles[:, i]))
        flag = " [≈ prior]" if i < 2 else ""
        posterior_lines.append(f"  {name:<12} truth={tr:>+7.3f}  post={m:>+7.3f}±{sd:.3f}{flag}")
    ax.text(
        0.55, 0.55, "\n".join(posterior_lines),
        fontsize=10, family="monospace", transform=ax.transAxes, va="top",
    )

    # Posterior histograms with prior + truth overlaid.
    for i, (name, tr, mu0, sd0) in enumerate(zip(names, truths, prior_means, prior_sds)):
        a = fig.add_subplot(gs[2, i])
        a.hist(np.asarray(theta_particles[:, i]), bins=25, density=True,
               color="C4", alpha=0.6, label="SMC²+EKF post.")
        # Overlay the prior pdf.
        xs = np.linspace(tr - 4 * sd0, tr + 4 * sd0, 200)
        prior_pdf = np.exp(-0.5 * ((xs - mu0) / sd0) ** 2) / (sd0 * np.sqrt(2 * np.pi))
        a.plot(xs, prior_pdf, color="C7", lw=1.0, ls="--", label="prior")
        a.axvline(tr, color="k", lw=1.2, label="truth")
        a.set_title(name)
        a.tick_params(axis="y", labelleft=False, left=False)
        if i == 0:
            a.legend(loc="upper right", frameon=False, fontsize=8)

    fig.suptitle(
        "SMC² + cuthbert EKF — diagnostics on the same data as example 02",
        y=0.995,
    )
    out = _common.save(fig, "05_smc2_ekf_fit.png")
    print(f"saved: {out}")


if __name__ == "__main__":
    main()
