# RenewalBirch

The idea of this example is to try out the particle filter approach given by the [Birch](https://birch-lang.org) probabilistic programming language, especially the rolling `Tape` concept in Birch's built-in Sequential Monte Carlo.
The example here is a simplified [EpiNow2](https://epiforecasts.io/EpiNow2/)-style renewal model
where I've attempted to redirect the `EpiNow2` data processing front and plotting back around a call to a `birch` program which does the SMC inference; essentially, this aims to slot `birch` into the slot where 
`stan` lives.
The point is to exercise Birch's SMC machinery: the alive particle filter and
Murray's lazy object-copy substrate on a realistic epi model.
I've added an `epinow_birch()` front-end that drops Birch-SMC into the compute slot but otherwise aims to reuse the other functions that `epinow()` entrypoint function uses under the hood.

`demo.R` shows fitting to EpiNow2's packaged `example_confirmed` COVID case series (130 days).

## EpiNow2 front-end: `epinow_birch()`

Reusing EpiNow2's *own* machinery on both ends:
- `discretise()`/`get_pmf()` to turn the distribution specs into the model's PMFs
- `calc_summary_measures()` and `plot_estimates()` to render the results.

So an epi user gets the familiar EpiNow2 Rt / infections / reported-cases plots, with a particle filter doing the inference instead of Stan.

```r
source("R/epinow_birch.R")
fit <- epinow_birch(
  EpiNow2::example_confirmed,                          # data (positional)
  generation_time = EpiNow2::example_generation_time,  # dist specs, as in epinow()
  delays          = EpiNow2::example_incubation_period +
                    EpiNow2::example_reporting_delay,
  method          = "rw"    # "rw" = self-organizing-RW + alive PF; "smc2" = nested SMC²
)
plot(fit)                   # EpiNow2-style Rt / infections / reported-cases panel
```

Or just run the demo (after `birch build`) — one call, ~1 minute:

```sh
Rscript demo.R            # -> figures/epinow_birch_demo.png
```

![EpiNow2 front-end, Birch compute](figures/epinow_birch_demo.png)

The signature mirrors the core of `epinow()` — `data` positional; `generation_time`,
`delays`, `obs`, and `rt_prior = c(mean, sd)` (the initial R₀ prior, as in
`rt_opts()`) kwargs. The static-parameter priors are configurable too
(`sigma_rw_prior`, `phi_prior`, `dow_prior_sd`) — one set of values that serves as
the *initial spread* for RW and the *outer prior* for SMC². Plus a Birch `method`
(`"rw"`, with `nparticles`/`nsamples`/`rw_sd`; or `"smc2"`, with `ntheta`/`nx`).
Other `epinow()` options are out of
scope for this toy. `method = "smc2"` currently returns the static-parameter
posterior (θ); routing its state trajectories into the same plots is the
`(θ_i, x_T,i)` extraction noted under Possible directions.

![Posterior Rt](figures/rt_rw.png)

## Installation

**1. Install the Birch toolchain.** ([full instructions](https://birch-lang.org/getting-started/))

```sh
# macOS (Homebrew):
brew install lawmurray/all/birch          # pulls in numbirch, membirch, birch-standard

# Linux: install from the software repository, or build from source
#   (see birch-lang.org/getting-started). A from-source build is also what enables
#   OpenMP threading for `parallel for` — see Notes on Birch capabilities.
```

**2. Install the R dependencies.**

```r
install.packages(c("EpiNow2", "jsonlite", "ggplot2", "patchwork", "data.table"))
```

**3. Build the model.** From this `RenewalBirch/` directory:

```sh
birch build          # transpile the .birch sources to C++ and compile the package
```

`birch build` compiles the package in place, which is all the demo and the `birch
sample` / `birch smc2` programs need. (`birch install` would additionally install
it system-wide as a library other Birch packages can `require` — not needed here.)

## Running it

The one-call front-end (after `birch build`, a single R call):

```sh
Rscript demo.R                                 # epinow_birch() end-to-end -> figures/
```

Or the underlying steps directly (run configs are generated, not committed —
`epinow_birch()`/`demo.R` write theirs automatically; here we write one inline):

```sh
Rscript R/prepare_data.R                          # EpiNow2 example data + PMFs -> input/

cat > config/renewal.json <<'JSON'
{ "model":  {"class": "RenewalModel"},
  "filter": {"class": "AliveParticleFilter", "nparticles": 512},
  "sampler":{"nsamples": 200},
  "input":  "input/epinow2_example.json",
  "output": "output/renewal.json" }
JSON
birch sample --config config/renewal.json         # RW fit -> output/renewal.json
Rscript R/plot_results.R output/renewal.json rw   # -> figures/{rt,infections,pp}_rw.png

# SMC² — proper static-parameter posterior (custom program, no config):
birch smc2 --ntheta 100 --nx 100 --nmoves 5 --output output/smc2.json   # nmoves = PMMH chain length
Rscript R/plot_smc2.R output/smc2.json output/renewal.json   # -> figures/smc2_posterior.png

# optional: EpiNow2 (Stan) reference Rt to overlay (slow, minutes):
Rscript R/epinow2_reference.R                     # -> output/reference_rt.csv
```


### The static-parameter problem, and two engines

Birch 2.1.7 ships only `ParticleFilter` / `AliveParticleFilter` and a single `ParticleSampler` (no PMMH, no working move kernel — see Notes below).
In this situation, the static parameters (`σ_rw`, `φ`, day-of-week) are only importance sampled, in the sense that the initial draw of static parameters across the particles from the prior contribute to the log-weight of each particle, and therefore, in an infinite partcle limit, are jointly inferred with the dynamic state of the epidemic.
However, realistically, its well known that this approach is degenerate on a long series of data, that is that particle resampling collapes the static parameter spread so that usually only one value is left across the filter.

So the example carries two inference engines:

**1. `RenewalModel` — self-organizing RW (fast; the default).**
The static parameters are promoted to *slowly-drifting states* via a small fixed-variance random walk on their unconstrained domains (Kitagawa 1998; Liu-West 2001 without shrinkage), so the alive particle filter rejuvenates them at every resample.
`ls_rw = log σ_rw`, `lphi = log φ`, and 7 day-of-week logits each drift with a fixed jitter `τ ≈ 0.03`.
Each particle carries its own tiny evolving static-state, copied on write by Birch's lazy object-copy. Infections stay deterministic and are advanced incrementally.
The `birch` computation model means that `simulate(t)` computes only `I[t]` as one `O(uot)` convolution.
It never re-solves from `t=1`, so this is a fully on-line inference approach (and restartable in principle, though Birch has no built-in restart yet — see the anchor-init note under Notes).

**2. `smc2` — SMC² (proper static-parameter inference).**
`src/smc2.birch` is a custom Birch program implementing SMC² (Chopin, Jacob &
Papaspiliopoulos 2013).
In SMC² the idea is that the static parameters _can_ make "moves" that leave their distribution stationary, and therefore avoid degeneracy, for example making a Metropolis-Hastings proposal-accept/reject sampling move (MH); this is called the outer particle filter over θ = (log σ_rw, log φ, 7 day-of-week logits).
There are two advantages of doing MH moves over a particle filter compared to sequential MH (i.e. MCMC):

1. We can propose a new static parameter set θ′ from an adaptive random walk scaled by the empirical variance over the filters θ-population (2.38²/d, Roberts–Rosenthal)
2. As more data is ingested we only resample when the θ-ESS falls below `ess_frac·ntheta`, so while the diversity of weight of outer filter remains high this method is on-line.

Each *inner* particle filter is associated with a θ-particle (`RenewalModelFixed`) over the states. 
At each observation every inner filter steps once, which increments their log-weight and therefore the log-weight of their θ-particle.
When resampling-move in the outer filter occurs the θ-particles are resampled (lazy-copying their inner filters) and **PMMH-rejuvenated** with the MH move which runs a fresh inner filter over the data
so far.
The fresh run of the inner filters is what makes SMC² slower than the fully on-line self-organising RW approach, however, it avoids degeneracy better than the self-organising RW approach because the likelihood of each new hyperparameter proposal is unbiasedly estimated by rerunning the full inner filter (the pseudo-marginal property that keeps the MH move exact).

By default the MH proposal uses only the per-component variances (`full_cov = false`); dropping the noisy off-diagonal correlations stabilises the move at modest particle counts, while
`full_cov = true` uses the full Cholesky.
Mixing hinges on `nx` (inner particles): the inner log-likelihood variance grows with `t`, so too-small `nx` gives noisy ML estimates → low PMMH acceptance → particle collapse; `nx` should scale with the series length (see growing-`Nx` under Possible directions).

**Rejuvenation needs a sequence of MH moves, not just a single move (`nmoves`).**
With only one PMMH step per resample-move the θ-particles can impoverish badly, the reason is that a resample makes many copies of whichever particle drew a high likelihood, perhaps by chance, every copy inherits the *same frozen* stored estimate `ll_c`, and a single fresh proposal rarely beats a lucky-high value, therefore all copies reject and stay collapsed.
Empirically this has the same degeneracy problems as not rejuvenating the static parameters with few **unique θs in the filter**.
Bumping `nx` alone does not neccesarily fix this because it shrinks the estimate noise but does nothing about a *stale* lucky-high weight.
The fix is a fixed **`nmoves`-step MH chain** per rejuvenation (default 5): each step is an ordinary accept/reject, but on the *first* acceptance the particle's `filters[j]`/`ll_c` is refreshed to a non-lucky estimate, so the rest of the chain compares against *that* and the stuck copies escape.

## The model

Continuous renewal; stochasticity enters at observation (EpiNow2's default).

| Component | Specification |
|---|---|
| **Seeding** | Each initial latent infection over the generation-interval window (`uot = 14` days) is its own sampled parameter, `logI0[j] ~ Normal(log(mean first-week cases), 1)` — replaces EpiNow2's exponential-growth root-find. |
| **Generation interval** | EpiNow2's default Gamma (mean 3.64 d, sd 3.08 d, max 14; Ganyani et al.), discretised via EpiNow2's own `discretise()`. |
| **Reporting delay** | EpiNow2's default incubation LogNormal (meanlog 1.62, sdlog 0.42) convolved with the reporting LogNormal (meanlog 0.58, sdlog 0.47), discretised and truncated. |
| **Rt** | Differenced random walk on `log Rt` — a random walk on the *first differences* (`Δ logR[t] = Δ logR[t-1] + ε`; equivalently a second-order / integrated RW): a level anchor `logR[uot] ~ Normal(logR0_mean, logR0_var)` (= EpiNow2 `LogNormal(mean 1, sd 1)` on R₀ by default), a gentle initial slope `logR[uot+1] ~ Normal(logR[uot], σ_rw²)`, then `logR[t] ~ Normal(2·logR[t-1] − logR[t-2], σ_rw²)` — a smooth, locally-linear trend. `σ_rw` (the innovation sd) drifts (self-organizing). (EpiNow2 default is a GP; this is a cheaper smoothing prior.) |
| **Renewal** | `I_t = R_t · Σ_{s=1..14} g_s · I_{t-s}` (no susceptible depletion — EpiNow2 default). |
| **Day of week** | Length-7 simplex (softmax of 7 logits) × 7 so it averages to 1 — EpiNow2's `week_effect`. |
| **Observation** | Delay-convolved, day-of-week-adjusted expected cases; Negative-Binomial-2 likelihood, `Var = μ + μ²/φ`, with `φ = exp(lphi)` and `lphi ~ Normal(log 10, 0.7²)` — a drifting (self-organizing) overdispersion for RW, the outer prior for SMC², configurable via `phi_prior`. Implemented with a `factor` matching Stan's `neg_binomial_2`. |


### Reading the output

Each `ParticleSampler` run's `draw()` already samples one ancestral trajectory
in proportion to the particle weights, so the `nsamples` runs are
**equally-weighted posterior draws** — the credible bands come from their spread.
The per-run `lweight` is the marginal-likelihood estimate (for evidence / an
outer PMMH), **not** a cross-run importance weight; do not reweight the
trajectories by it. `R/plot_results.R` pools them with equal weight.

### Notes on Birch capabilities (from this exercise)

- **Restart from output?** No built-in resume/checkpoint (`birch sample` has no
  such flag; the library's `Checkpoint` is delayed-expression autodiff
  memoisation). You *can* warm-start manually: `read(buffer)`/`read(t)` fill any
  model variable, so a saved state window from a prior output can be injected as
  the initial condition (an anchor-init hook — not yet added here).
- **Move kernel / fixed-lag.** The `Kernel` base carries `nlags` (fixed-lag
  window), `nmoves`, and a PID-tuned `scale`; `LangevinKernel` does a gradient
  (MALA) move. But this move machinery is **not functional for a particle filter
  in 2.1.7** (segfaults / hangs, and is untested in the repo test suite, whereas
  AD itself is tested) — which is why static-parameter inference here goes via
  SMC² rather than an in-filter move. See Possible directions for the MALA route
  that *would* work (Fisher's identity, outer θ-move).
- **Parallelism — works on Linux, not on the macOS Homebrew build.** `smc2.birch`
  uses `parallel for` over the θ-particles (the loop is embarrassingly parallel),
  and it *does* transpile to real `#pragma omp parallel for`. It is correct
  (posterior unchanged, no races) but runs **single-threaded on the Homebrew
  macOS 2.1.7 binary** (`user ≈ real` on 10 cores): Apple clang doesn't enable
  OpenMP by default, so the pragmas are compiled out (`-Wno-unknown-pragmas`, no
  `-fopenmp`), and the shipped `numbirch`/`membirch`/`birch-standard` `.dylib`s
  were themselves built single-threaded — forcing `-fopenmp` on just the package
  segfaults against those non-thread-safe libraries. `membirch`'s header *is*
  written for OpenMP (`#ifdef _OPENMP … omp_get_thread_num()`), so the substrate
  is designed to be thread-safe; it simply has to be **compiled** that way. On
  **Linux with GCC** (native `-fopenmp`, thread-safe throughout), a from-source
  build of the whole Birch stack threads correctly and the `parallel for` gives a
  near-linear speedup — that's the recommended route for real workloads. The
  macOS/Apple-clang/libomp friction is exactly why the Homebrew bottle is serial;
  it's fine to leave macOS single-threaded. (NumBirch also has a CUDA/GPU backend,
  NVIDIA-only, again a from-source build.)
- **For readers from Gen.jl:** Birch has no static-DSL argdiff incremental
  `update`; it offers checkpointed stepwise SMC, lazy object-copy (copy-on-write
  particle state), and delayed sampling / Rao-Blackwellisation instead.

## Results

- **RW engine** (`nparticles = 512`, `nsamples = 150`, 144 steps): ~200 s
  single-threaded on an M-series Mac. Smooth posterior Rt (≈2 in early March,
  crossing 1 in late March, ≈0.85 through spring, rising in June) with 50 %/90 %
  bands, `σ_rw ≈ 0.04`, `φ ≈ 6`.
- **SMC²** (`ntheta = 100`, `nx = 100`, `nmoves = 5`): ~20 min; clean unimodal θ
  posterior centred on the RW-engine values (`σ_rw ≈ 0.037` [0.030, 0.045],
  `φ ≈ 10` [8.2, 11.8]), tighter than the RW engine's, with θ-ESS ≈ 68 and ~29
  unique θ. With `nmoves = 1` the θ-particles impoverish (~8 unique, 58–77 % on one
  value, needle-spike KDE) and the estimate wobbles run-to-run — see the
  rejuvenation-chain note above. Bumping `nx` alone (100 → 200, ~12.5 min) does
  *not* fix it; the fixed MH chain does.

See `figures/`.

## Possible directions (not implemented)

- **Offline→online handoff: SMC² warm-start for a cheap RW filter.** Run the slow SMC²
  to burn in the hyperparameters, then hand off to the fast self-organizing-RW
  bootstrap filter for online continuation — seeded from the *proper* joint
  posterior instead of the prior. At time `T`, SMC² already holds a weighted set
  `{θ_i}` each with an inner filter approximating `p(x_{1:T} | y, θ_i)`; draw
  `(θ_i, x_T,i)` by sampling one inner particle per θ-particle (∝ inner weights)
  and reading its `logR_T` + recent infection window, then seed a new RW-θ filter
  with those `N` distinct particles and carry on for `t > T`. This is exactly the
  **anchor-init hook** (initialise from a supplied state window, not the `t=1`
  seeds). The one Birch unknown is whether a filter's particle array (`filter.x`)
  can be populated from outside — if not, wrap a custom filter loop as
  `smc2.birch` already does. Hand off right after a PMMH move, when θ diversity is
  freshest. Static→drifting θ is a deliberate approximation, now well-initialised.

- **Growing `Nx` for SMC².** The current driver fixes the inner particle count,
  which is why the late-`t` PMMH replays get both slow and less effective (the
  log-likelihood-estimator variance grows ~linearly in `t`). The classic fix is to
  grow `Nx` over the sweep — monitor the move acceptance rate and double `Nx` via
  the importance-sampling *exchange* step (Chopin–Jacob–Papaspiliopoulos 2013),
  keeping the estimator variance ≈ 1. `Nx` also need not be uniform across
  θ-particles.

- **MALA for the outer θ-move.** A gradient θ-move needs the marginal-likelihood
  score `∇_θ log p(y|θ)`, which a particle filter doesn't give differentiably
  (resampling is non-differentiable — the same wall the in-filter move kernel
  hits). The legitimate route is Fisher's identity: average each inner particle's
  `∇_θ` of its *complete-data* log-density — exactly what Birch's *working*
  per-model AD (`test_grad`) computes — but it needs per-particle score
  accumulation + resampling-ancestry bookkeeping, and pays off only if θ becomes
  high-dimensional (here it is 9-D and RW-MH mixes fine).

- **De-duplicate the two model files.** `RenewalModel` (drifting statics) and
  `RenewalModelFixed` (fixed statics, the SMC² inner model) share ~120 lines and
  differ in only one thing: whether the statics are sampled + drifted or set
  externally by the driver. They could collapse into one class with a
  `fixed_theta` flag — two `if !fixed_theta` guards around the sample/drift steps
  (the numerical guards are harmless always-on, so no branching there), with
  `smc2.birch` setting `m.fixed_theta <- true`. Left as two files here for
  one-file readability.


## Layout

```
RenewalBirch/
├── birch.yml                      # package manifest
├── demo.R                         # one-call epinow_birch() demo (start here)
├── src/
│   ├── RenewalModel.birch         # engine 1: self-organizing RW
│   ├── RenewalModelFixed.birch    # fixed-θ inner model for SMC²
│   └── smc2.birch                 # engine 2: SMC² driver program
├── config/                        # run configs written here (generated, not committed)
├── input/                         # generated by prepare_data.R / epinow_birch()
├── output/                        # written by `birch sample` / `birch smc2`
├── R/
│   ├── epinow_birch.R             # EpiNow2-style front-end (specs in, EpiNow2 plots out)
│   ├── prepare_data.R             # build the Birch input from EpiNow2 defaults
│   ├── plot_results.R             # RW trajectories -> Rt / infections / reported-cases
│   ├── plot_smc2.R                # SMC² θ posterior
│   └── epinow2_reference.R        # optional EpiNow2/Stan reference Rt
└── figures/                       # plots
```
