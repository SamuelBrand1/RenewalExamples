"""03 — SMC² + UKF, compared head-to-head with the Liu-West PF.

Both methods fit the SAME synthetic time series; we plot their marginal
posteriors over (log_tau_R, log_tau_F, log_phi) on the same axes.  The
two should broadly agree, but SMC² uses ~30× fewer parameter particles
(it only carries θ-particles; the latent state is integrated out by the UKF).
"""

from __future__ import annotations

import _common  # noqa: F401
import jax.numpy as jnp
import jax.random as jr
import matplotlib.pyplot as plt
import numpy as np

from smc_renewal.config import default_config
from smc_renewal.pf.runner import run_liu_west
from smc_renewal.smc2.ekf_cuthbert import marginal_log_likelihood as ekf_loglik
from smc_renewal.smc2.runner import fit_smc2
from smc_renewal.state import ParticleParams
from smc_renewal.synthetic import simulate


def main():
    _common.set_clean_style()
    cfg = default_config()
    T = 150
    truth = ParticleParams(
        log_tau_R=jnp.asarray(-4.0),
        log_tau_F=jnp.asarray(-12.0),
        log_phi=jnp.asarray(2.5),
    )
    ds = simulate(jr.key(2), cfg, T, params=truth)

    # PF: many state-and-param particles.
    print(f"Liu-West PF (T={T}, N=3000)…")
    pf = run_liu_west(jr.key(11), cfg, ds.y, n_particles=3000, h=0.1)
    pf_lw = pf.log_weights_history[-1]
    pf_w = jnp.exp(pf_lw - jnp.max(pf_lw)); pf_w = pf_w / pf_w.sum()
    pf_theta = jnp.stack(
        [pf.params_history.log_tau_R[-1], pf.params_history.log_tau_F[-1], pf.params_history.log_phi[-1]],
        axis=-1,
    )

    # SMC^2: just θ-particles with a Gaussian filter inside.  Here we use the
    # library-backed Extended Kalman Filter from `cuthbert.gaussian.moments`
    # (auto-diff linearization).  The hand-rolled UKF in `smc_renewal.smc2.ukf`
    # remains the default; pass `marginal_loglik_fn=...` to swap.
    print(f"SMC² + cuthbert EKF (T={T}, N=128, ≤20 outer steps)…")
    smc = fit_smc2(
        jr.key(11), cfg, ds.y,
        n_particles=128, num_mcmc_steps=4, rwmh_scale=0.5,
        marginal_loglik_fn=ekf_loglik,
    )
    smc_theta = smc.result.particles  # (128, 3) — equal weights post-final-resample
    print(f"  SMC² ran {smc.result.n_steps} adaptive-tempering steps")

    names = ["log_tau_R", "log_tau_F", "log_phi"]
    truths = [-4.0, -12.0, 2.5]

    fig, axes = plt.subplots(1, 3, figsize=(11, 3.5))
    for i, (name, tr) in enumerate(zip(names, truths)):
        a = axes[i]
        # Weighted PF histogram.
        a.hist(
            np.asarray(pf_theta[:, i]), bins=30, weights=np.asarray(pf_w),
            density=True, color="C0", alpha=0.55, label="Liu-West PF",
        )
        # Equal-weighted SMC² histogram (already resampled to equal weights).
        a.hist(
            np.asarray(smc_theta[:, i]), bins=30, density=True,
            histtype="step", color="C3", lw=1.6, label="SMC² (UKF inner)",
        )
        a.axvline(tr, color="k", lw=1.2, ls="--", label="truth")
        a.set_title(name)
        a.tick_params(axis="y", labelleft=False, left=False)
        if i == 0:
            a.legend(loc="upper left", frameon=False, fontsize=8)

    fig.suptitle(
        f"Posterior over θ — Liu-West PF (N=3000) vs SMC² (N=128 θ-particles)  ·  T={T} days",
        y=1.05,
    )
    out = _common.save(fig, "03_smc2_vs_pf.png")
    print(f"saved: {out}")

    print("\nPosterior means:")
    print(f"{'param':<14}{'truth':>9}{'PF':>10}{'SMC²':>10}")
    for i, (n, t) in enumerate(zip(names, truths)):
        pf_m = float(jnp.sum(pf_w * pf_theta[:, i]))
        sm_m = float(jnp.mean(smc_theta[:, i]))
        print(f"  {n:<12}{t:>9.3f}{pf_m:>10.3f}{sm_m:>10.3f}")


if __name__ == "__main__":
    main()
