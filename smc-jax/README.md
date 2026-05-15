# smc-renewal

SMC-based sequential forecasting for a renewal model with **nested-RW
volatility**.  Demo of two particle-filter approaches as alternatives to the
team's current weekly NUTS refits of pyrenew models.

## Model

State (daily index `t`):

```
log Rt[t]       = log Rt[t-1]      + sigma_R[t-1] * eps_R[t]
log sigma_R[t]  = log sigma_R[t-1] + tau_R         * eta_R[t]
gamma[t]        = gamma[t-1]       + sigma_F[t-1] * eps_F[t]
log sigma_F[t]  = log sigma_F[t-1] + tau_F         * eta_F[t]
```

Renewal + infection feedback (using
[`pyrenew.latent.infection_functions.compute_infections_from_rt_with_feedback`](https://github.com/CDCgov/PyRenew)):

```
I(t)      = Rt_adj(t) * sum_{tau=1..Tg} I(t-tau) * g(tau)
Rt_adj(t) = Rt(t)     * exp( gamma(t) * sum_{tau=1..Tf} I(t-tau) * f(tau) )
```

Observation:

```
mu_y[t] = sum_{tau=0..Td-1} I(t-tau) * d(tau)
y[t]    ~ NegBin( mu_y[t] , phi )
```

Top-level inferred parameters: `(tau_R, tau_F, log_phi)`.  `log_I0` and the
initial latent values are sampled per particle.  Ascertainment fixed at 1.

## Model variants

Four PF variants live under `pf/`, sharing the same renewal + feedback
dynamics and `NegBin` observation model but differing in how the volatility
of `log Rt` (and `log F`) is represented.  All four go through the same
shared scan loop in `pf/_runner_core.py`; each variant supplies only its
model class and an initial-parameter sampler.

| | A (nested RW) | B (direct σ) | C (trend / integrated BM) | D (discrete + immigration) |
| --- | --- | --- | --- | --- |
| **What evolves** | log σ random walks; log Rt walks at rate σ | log σ is a Liu-West param (no σ in state) | Velocity v_R does a random walk; log Rt evolves as `log Rt + v_R` | Same trend dynamics as C, but `I(t)` is Poisson-sampled with an immigration rate μ |
| **State** | log Rt, log σ_R, log F, log σ_F, log I0, I_buf | log Rt, log F, log I0, I_buf | log Rt, v_R, log F, v_F, log I0, I_buf | log Rt, v_R, log F, v_F, log I0, I_buf (integer infections) |
| **Liu-West cloud** | (log τ_R, log τ_F, log φ) | (log σ_R, log σ_F, log φ) | (log σ_vR, log σ_vF, log φ) | (log σ_vR, log σ_vF, log μ, log φ) |
| **Entry point** | `pf.runner.run_liu_west` | `pf.runner_sigma.run_liu_west_sigma` | `pf.runner_trend.run_liu_west_trend` | `pf.runner_discrete.run_liu_west_discrete` |
| **Demo** | examples 01–07 | example 08 (vs A), 09 | example 10 | example 11 |

Model A is the model described in detail above and is what `smc2/` targets.
The sequential-update path (`extend_liu_west`, `rolling_origin_forecast`) is
currently Model-A only; the shared core in `pf/_runner_core.py` is variant-
agnostic, so adding sequential updates for B/C/D is a thin wrapper.

## What's here

- `transition.py` — pure-JAX per-step transition, verified against
  pyrenew's `compute_infections_from_rt_with_feedback` to ~1e-5.
- `synthetic.py` — seedable forward-simulator.
- `pf/` — Liu-West-augmented bootstrap particle filter built on
  [`pfjax`](https://github.com/mlysy/pfjax) primitives (`BaseModel`,
  `particle_filter`, `particle_smooth`). Includes filtering, **trajectory
  smoothing**, **Steyn-style fixed-lag resampling** (via the `fixed_lag_L`
  kwarg in `run_liu_west` — observations at time t inform R at time t-lag,
  so we cap how far back resampling re-permutes the trajectory history),
  and first-class `extend_liu_west` for sequential updates.
- `smc2/` — adaptive-tempered SMC² over `theta`. The marginal log-likelihood
  inside is pluggable:
  - `smc2.ukf` — hand-rolled UKF (default).
  - `smc2.ekf_cuthbert` — library-backed EKF via
    [`cuthbert`](https://github.com/state-space-models/cuthbert)'s
    `gaussian.moments` filter; agrees with the UKF to ~0.2% on the marginal
    log-lik for our model.
  - `extend_smc2` for sequential updates.
- `rolling_origin.py` — rolling-origin forecast harness that ingests new data
  with sequential updates (not refits) and scores at 1–4 week horizons via CRPS
  + 50/95% interval coverage.

## Run

```bash
uv sync
uv run pytest
uv run python examples/01_synthetic_demo.py
```

See `examples/` for end-to-end scripts covering synthetic data, PF and SMC²
fits, rolling-origin forecasting, and per-model demos.

## Status

- Liu-West PF: posterior recovery on synthetic data within ~5% of truth on
  `(log_tau_R, log_tau_F, log_phi)`.
- SMC² + UKF or EKF: posterior on `log_phi` is well-identified, but **both
  Gaussian filters are essentially blind to `log_tau_R` and `log_tau_F`** —
  the higher-order nested-volatility parameters get linearized away (no direct
  observation of `log sigma_R` means the chain `tau_R → log_sigma_R variance
  → sigma_R via exp` decouples at the EKF/UKF linearization point). The Liu-
  West PF does identify these — its strength on hierarchical-volatility models.
- Sequential update verified: full fit through `T` matches fit-through-`T/2`
  then sequential update to `T` within sampling noise.
- Rolling-origin forecast coverage on synthetic data is calibrated
  (95% interval covers ≥80% of held-out days, averaged across 1–4 week horizons).
- NUTS comparison: deliberately out of scope.
