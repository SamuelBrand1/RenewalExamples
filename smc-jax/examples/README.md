# Examples

Each script is standalone — run with `uv run python examples/<name>.py`.
Plots land in `examples/figures/`.

The examples are roughly ordered (i) intuition → (ii) single-method demos →
(iii) cross-method comparisons → (iv) longer-horizon experiments + alternative
model formulations.

## 01 — Synthetic data tour

Generates one ground-truth trajectory from Model A and plots the four random
walks (`log Rt`, `σ_R`, `F`, `σ_F`), the latent infections, and the noisy
NB-observed cases on one figure. Builds intuition for what the model does
before any inference.

```bash
uv run python examples/01_synthetic_demo.py
```

Output: `figures/01_synthetic_tour.png` · ~5 s

## 02 — Liu-West PF on Model A, with filter + smoother diagnostics

Fit the bootstrap PF (4000 particles, `h=0.1`) to a 180-day synthetic series
and visualise (a) **filter and smoother** credible bands for `log Rt(t)` vs
truth — using `pfjax.particle_smooth` for the smoother — (b) posterior band
for `σ_R(t)`, (c) ESS over time, (d) marginal θ posteriors with truth lines.

```bash
uv run python examples/02_pf_fit.py
```

Output: `figures/02_pf_fit.png` · ~10 s. Posterior means typically recover
truth to within ~1%; smoother coverage is tighter than filter coverage.

## 03 — SMC² (cuthbert EKF inner) vs Liu-West PF on Model A

Both methods fit the same synthetic series; we plot their marginal posteriors
over `θ` on the same axes. SMC² uses ~30× fewer parameter particles (128 vs
3000) because the latent state is integrated out by the cuthbert EKF — yet
the two agree closely on this short-T case.

```bash
uv run python examples/03_smc2_vs_pf.py
```

Output: `figures/03_smc2_vs_pf.png` · ~30 s.

## 04 — Rolling-origin forecast with Liu-West PF

The operational story: fit on the first 80 days, forecast 28 days ahead,
*sequentially* update the particle cloud each week with new data and
re-forecast. No refit. Visualises (a) forecast fans at three rolling origins
with truth overlaid, (b) CRPS vs horizon, (c) interval coverage vs horizon.

```bash
uv run python examples/04_rolling_origin_forecast.py
```

Output: `figures/04_rolling_origin_forecast.png` · ~30 s.

## 05 — SMC² + EKF diagnostics (companion to 02)

Same reference data as example 02; swaps the Liu-West PF for SMC²-with-
Gaussian-inner. Plots the EKF filtered band for `log Rt(t)` and `σ_R(t)` at
the SMC² posterior-mean θ, and θ posteriors with the prior overlaid — making
the **prior-collapse on `log_τ_R` / `log_τ_F`** visible. The headline:
`log_φ` posterior is identified; the τ posteriors are visually
indistinguishable from the prior. This is the "Gaussian filter blind spot"
documented in `PYRENEW_FRICTION.md`.

```bash
uv run python examples/05_smc2_ekf_fit.py
```

Output: `figures/05_smc2_ekf_fit.png` · ~15 s.

## 06 — Rolling-origin forecast with SMC² + EKF (companion to 04)

Same reference data and horizons as example 04, but the inference engine is
SMC² + EKF instead of the Liu-West PF. Sample one latent state from each
θ-particle's terminal EKF Gaussian, then forward-simulate.

```bash
uv run python examples/06_smc2_ekf_rolling_origin.py
```

Output: `figures/06_smc2_ekf_rolling_origin.png` · ~30 s.

Contrast with example 04: EKF doesn't have particle tail trajectories that
blow up exponentially, so its CRPS at long horizons is orders of magnitude
smaller than the PF's. Neither method anticipates the seasonal turning point
— that's a structural limit of the driftless-RW Rt model, not a difference
between the inference algorithms.

## 07 — Model A on multi-season data: doubly-stochastic, separated timescales

Both `log Rt(t)` and `log F(t)` follow the same nested-RW family but on
**deliberately separated timescales**: log Rt swings fast (seasonal), log F
drifts slow (strain-turnover scale). The two are partially confounded; the
timescale gap is what lets the data separate them most of the time. SMC's
joint particle cloud gracefully represents the slight-unidentifiability
correlation a point-estimator would obliterate.

Truth construction: log Rt(t) is forced to a seasonal trajectory (4 annual
seasons, T=1440 = ~4 years); log_F(t) is drawn from the model's own nested-RW
with truth `log_τ_F = -7`. `F = exp(log_F) > 0` by construction, so the
feedback always damps — physically correct for susceptibility-depletion /
strain-evolution interpretations.

Uses **Steyn-style fixed-lag resampling** (`fixed_lag_L=21`): observations
at time t are informative about R from a few days ago, so resampling further
back induces path degeneracy without adding signal. Visibly improves log_Rt
smoother coverage from ~0.77 to ~0.91 vs nominal 0.9.

```bash
uv run python examples/07_multi_season.py
```

Output: `figures/07_multi_season.png` + `figures/07_multi_season_forecasts.png` · ~60 s.

The forecast figure (mid-season ascending-phase origins, log y-axis) shows
the model's structural limits honestly: Rt forecasts have proper exponential
growth/decay around the origin's inferred `log Rt`, but no notion of the
seasonal turning point.

## 08 — Liu-West placement: log_τ (Model A) vs log_σ (Model B)

Same synthetic data fit with two different placements of the Liu-West
approximation:

- **Model A** (`pf/runner.py::run_liu_west`): doubly-stochastic; Liu-West
  applied to the static `(log τ_R, log τ_F, log φ)` triple.
- **Model B** (`pf/runner_sigma.py::run_liu_west_sigma`): one fewer level
  of hierarchy. `log σ_R, log σ_F` are themselves Liu-West parameters
  (slowly drifting via shrink-jitter). No `τ`; the Liu-West shrinkage `h`
  plays the role `τ` played in Model A.

Plots `log Rt` filter bands side-by-side, then `log σ_R` trajectories
(Model A: latent state; Model B: Liu-West param cloud), then final-step
posterior histograms on the comparable parameters.

```bash
uv run python examples/08_lw_on_sigma_vs_tau.py
```

Output: `figures/08_lw_on_sigma_vs_tau.png` · ~60 s.

On short-T (T=360) the two give nearly identical performance (filter
coverage ~0.89 both, log marginal-lik within 0.1 nat). On longer-T
multi-season data (see example 09) Model B's identifiability is
substantially better.

## 09 — Model B on multi-season data (companion to 07)

Identical synthetic data as example 07, but fit with Model B (Liu-West on
`log σ` directly). Same 5-panel diagnostic + 4-origin forecast figures so
they can be compared file-for-file with example 07's outputs.

```bash
uv run python examples/09_model_b_multi_season.py
```

Output: `figures/09_model_b_multi_season.png` + `figures/09_model_b_forecasts.png` · ~75 s.

Headline result: Model B identifies `log σ_R ≈ -3.45` (truth-implied -3.7)
and `log φ ≈ +2.52` (truth +2.50) sharply, where Model A had τ posteriors
prior-stuck. **Forecast 95% upper at day 60+60: Model A ~300k vs Model B
~10k** — Model B's forecasts are far tighter and median trajectories track
the actual case rises better. The σ-placement is the recommended design for
this project's use case.

## 10 — Model C on multi-season data: integrated Brownian motion on log Rt / log F

Same synthetic data as examples 07 and 09, fit with **Model C**: the latent
`log Rt` and `log F` are once-integrated (their first derivatives — velocity
states `v_R`, `v_F` — do the random walking, levels move by the integrated
velocity). Discrete form:

```
v_R[t]    = v_R[t-1]    + σ_vR · η[t]
log Rt[t] = log Rt[t-1] + v_R[t-1]
```

Liu-West applied to `(log σ_vR, log σ_vF, log φ)`. The new state coordinates
`v_R`, `v_F` mean the median forecast actually *extrapolates the current
trend* (rising, falling, or accelerating), not just freezing at the current
level as in Models A / B.

```bash
uv run python examples/10_model_c_multi_season.py
```

Output: `figures/10_model_c_multi_season.png` + `figures/10_model_c_forecasts.png` · ~75 s.

The 5-panel figure now includes a `v_R(t)` panel showing the inferred slope
of `log Rt` over time. The forecasts panel shows the qualitative payoff:
medians rise steeply from ascending-phase origins, tracking the actual wave
much more *directionally* than Models A / B's flat-by-RW medians.

Caveat: the log_Rt **smoother** coverage drops to ~0.51 (vs ~0.9 nominal) —
when the model fits both level and velocity, the genealogy-tracing smoother
over-tightens. FFBS instead of genealogy tracing would fix this.

## 11 — Model D (discrete Poisson renewal + F-feedback + immigration) on multi-season data

Sibling of example 10 (Model C). Same integrated-BM dynamics on both
`log Rt` and `log F`, same F-feedback `exp(-F · conv(I, g))`, but the
renewal core is **stochastic**: infections at each step are Poisson-sampled
from `λ_t = μ + R_t · exp(-F · conv) · conv(I, g)`. A small immigration
rate `μ` (inferred as a Liu-West parameter, `log μ`) prevents extinction.
F-feedback keeps the discrete renewal bounded — without it, Rt > 1 sustained
for ~30 days produces 10⁶ case peaks.

So **Model D = Model C + Poisson I + immigration μ**. Liu-West cloud
is 4-D: `(log σ_vR, log σ_vF, log μ, log φ)`.

```bash
uv run python examples/11_model_d_discrete.py
```

Output: `figures/11_model_d_discrete.png` + `figures/11_model_d_forecasts.png` · ~75 s.

Demo's methodological point: bootstrap PFs handle a **discrete-count
latent process** just as comfortably as the deterministic-renewal variants.
The 5-panel figure adds a distinctive `I(t)` panel (filter + smoother bands
vs the Poisson-drawn truth) because in Model D the latent infections have
their own posterior, where in A/B/C they were deterministic given upstream
state. Headline: log_Rt filter cov ~0.94, log_μ posterior visibly shifted
from prior toward truth, log_φ sharply identified.

## 12 — Model E: outbreak analysis with GDM observation delay + guided PF proposal

Short single-outbreak (T=70 days) demonstrating that **high-ascertainment
regimes break the conditional-independence assumption** of examples 01–11's
observation model.  When ascertainment is close to 1, observations across
days from a single cohort are coupled through cohort-budget constraints:
if `N` people are infected, at most `N` of them can ever be observed, and
once individual reports on day `t` they can't be re-reported on day
`t' > t`.  Examples 01–11's simple `μ_y(t) = Σ d_s · I[t−s]` delay
convolution + NegBin observation can't represent this coupling.

Model E replaces that with a **Generalised Dirichlet Multinomial** (Stoner
et al) cohort partition: each cohort's eventually-observable cases get
stick-broken across reporting stages via independent Beta-Binomials, with
per-stage Beta means parameterised on the probit scale,
`Φ⁻¹(p_s) = b_0 + b_1 · s`.

Because the observation is now coupled with the latent partition, a
bootstrap PF fails (almost no particle samples a partition that hits the
observed `y_t` exactly).  We use a **guided proposal** for the cohort
partition: a multivariate **Wallenius noncentral hypergeometric** draw
(sequential weighted-without-replacement, weights = per-stage Beta means).
This (a) automatically hits `Σ O_s = y_t`, (b) respects per-cohort budgets
`O_s ≤ U[s]`, and (c) matches the target marginal means so IS weight
variance stays manageable.

```bash
uv run python examples/12_model_d_gdm.py
```

Output: `figures/12_model_d_gdm.png` + `figures/12_model_d_gdm_forecasts.png` · ~2 min.

The 6-panel diagnostic figure adds two Model-E-specific panels: a stacked
bar chart decomposing `y_t` into per-stage cohort contributions (the truth's
partition `O`), and a 6-axis posterior strip for the Liu-West cloud (now
6-D: `(log σ_vR, log σ_vF, log μ, b_0, b_1, log_M)`).

## What's where: A / B / C / D / E cheat-sheet

| Model | Latent dynamics | Liu-West parameters | Observation | Proposal |
|---|---|---|---|---|
| **A** (`pf/model.py`)          | log Rt RW with σ_R state walked by τ        | `(log τ_R, log τ_F, log φ)`    | delay-conv + NegBin | bootstrap |
| **B** (`pf/model_sigma.py`)    | log Rt RW with σ_R as LW param              | `(log σ_R, log σ_F, log φ)`    | delay-conv + NegBin | bootstrap |
| **C** (`pf/model_trend.py`)    | integrated BM (log Rt + v_R)                | `(log σ_vR, log σ_vF, log φ)`  | delay-conv + NegBin | bootstrap |
| **D** (`pf/model_discrete.py`) | C + **Poisson I** + immigration             | `(log σ_vR, log σ_vF, log μ, log φ)` | delay-conv + NegBin | bootstrap |
| **E** (`pf/model_gdm.py`)      | D + ascertainment thinning + cohort U-buffer | `(log σ_vR, log σ_vF, log μ, b_0, b_1, log_M)` | **GDM cohort partition** | **guided (Wallenius)** |
