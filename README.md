# RenewalExamples

A monorepo of renewal-equation epidemic (Rt estimation) modelling examples, exploring
different inference approaches and languages: Julia/Turing, the Birch probabilistic
programming language, and JAX. Each subproject below is self-contained with its own
dependencies and README.

## Contents

### [`RenewalLayers/`](RenewalLayers/)

A Julia package of `Lux.jl` layers/cells for renewal-equation epidemic models:

- `UnknownGI` / `FixedGI` — generation-interval distribution types ([`src/gis.jl`](RenewalLayers/src/gis.jl))
- `RenewalCell` — a recurrent renewal-equation cell ([`src/steps.jl`](RenewalLayers/src/steps.jl))
- `delay_conv` — a causal delay-convolution layer ([`src/conv.jl`](RenewalLayers/src/conv.jl))

[`scripts/`](RenewalLayers/scripts/) has three runnable Literate.jl-style examples: generation-interval
inference via Turing/NUTS with an AD-backend comparison, a basic `RenewalCell` simulation,
and a renewal model with observation delay.

### [`RenewalBirch/`](RenewalBirch/)

An EpiNow2-style Rt renewal model implemented in the [Birch](https://birch-lang.org)
probabilistic programming language, with an `epinow_birch()` R front-end that slots
Birch's SMC into EpiNow2's own API and plotting. Two inference engines: a self-organizing
random-walk + particle filter, and a proper SMC² (with PMMH rejuvenation) for
static-parameter inference.

### [`smc-jax/`](smc-jax/)

A JAX-based exploration of SMC/particle-filter alternatives to NUTS refits of
[pyrenew](https://github.com/CDCgov/PyRenew)-style renewal models: a bootstrap/Liu-West
particle filter and an SMC² (with UKF/EKF inner filters), across five model variants
(A–E) of increasing structural realism, plus rolling-origin forecasting.

## Packages

### Julia (`RenewalLayers/Project.toml`)

| Package | Purpose |
|---|---|
| [CensoredDistributions](https://github.com/epinowcast/CensoredDistributions.jl) | interval-censored/truncated distribution utilities for generation-interval construction |
| [Distributions](https://github.com/JuliaStats/Distributions.jl) | probability distribution types |
| [Enzyme](https://github.com/EnzymeAD/Enzyme.jl) | automatic differentiation backend |
| [Lux](https://github.com/LuxDL/Lux.jl) | the neural-network-style layer framework `RenewalCell`/`delay_conv` are built on |
| [LuxCore](https://github.com/LuxDL/LuxCore.jl) | core layer abstractions used by `Lux` |
| [Mooncake](https://github.com/compintell/Mooncake.jl) | automatic differentiation backend |
| [NNlib](https://github.com/FluxML/NNlib.jl) | neural-network primitives (convolutions) used by `delay_conv` |
| Random | standard library RNG |
| [ReverseDiff](https://github.com/JuliaDiff/ReverseDiff.jl) | automatic differentiation backend |
| [StatsPlots](https://github.com/JuliaPlots/StatsPlots.jl) | plotting distributions |
| [Turing](https://github.com/TuringLang/Turing.jl) | probabilistic programming / MCMC (NUTS) for parameter inference |

### Python (`smc-jax/pyproject.toml`)

| Package | Purpose |
|---|---|
| [blackjax](https://github.com/blackjax-devs/blackjax) | sampling library (SMC, MCMC) in JAX |
| [cuthbert](https://github.com/state-space-models/cuthbert) | Gaussian/Kalman-filtering state-space library, backs the library EKF (`ekf_cuthbert`) |
| [jax](https://github.com/jax-ml/jax) | array computing and automatic differentiation |
| [matplotlib](https://github.com/matplotlib/matplotlib) | plotting |
| [numpyro](https://github.com/pyro-ppl/numpyro) | probabilistic programming in JAX |
| [pfjax](https://github.com/mlysy/pfjax) | particle filtering in JAX |
| [properscoring](https://github.com/properscoring/properscoring) | proper scoring rules (e.g. CRPS) for forecast evaluation |
| [pyrenew](https://github.com/CDCgov/PyRenew) | reference renewal-equation epi model (CDC), used for infection/feedback dynamics |

### R / Birch (`RenewalBirch/`)

| Package | Purpose |
|---|---|
| [Birch](https://birch-lang.org) | probabilistic programming language/compiler providing the SMC/SMC² inference engines (install via `brew install lawmurray/all/birch`) |
| [EpiNow2](https://github.com/epiforecasts/EpiNow2) | reference Rt estimation package; reused for example data, PMF discretisation, and plotting |
| jsonlite | JSON I/O for passing data between R and Birch |
| ggplot2 | plotting |
| patchwork | combining `ggplot2` panels |
| data.table | fast data wrangling |

## License

[MIT](LICENSE)
