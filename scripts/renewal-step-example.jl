# src using Pkg
# src Pkg.activate(joinpath(@__DIR__, ".."))

using Distributions, LinearAlgebra, Markdown
using Random: default_rng, seed!
using StatsPlots
using Turing
using Lux
using RenewalExamples: RenewalCell
using ReverseDiff
using Enzyme

seed!(1234)

md"""
# Renewal Model via Recurrent Cell

This notebook demonstrates how to simulate and fit a renewal model using `RenewalCell`,
a Lux.jl recurrent cell that implements the renewal equation:

    I(t) = R(t) * sum_{s} g(s) * I(t - s)

The cell is wrapped with `Lux.Recurrence` to scan over a time series of reproduction
numbers, giving us efficient forward passes with AD-compatible gradients.
"""

md"""
## 1. Define a known generation interval distribution

We discretise a Gamma(3, 2) distribution into a PMF vector.
"""

gi_dist = Gamma(3.0, 2.0)
gi_length = 15
gi_pmf = [cdf(gi_dist, s) - cdf(gi_dist, s - 1) for s in 1:gi_length]
gi_pmf = gi_pmf ./ sum(gi_pmf)  # normalise

bar(gi_pmf;
    xlabel = "Days", ylabel = "Probability",
    title = "Generation Interval Distribution",
    label = "GI PMF", legend = :topright
)

md"""
## 2. Set up the Lux recurrence model

We create a `RenewalCell` and wrap it with `Lux.Recurrence` to scan over the
time series. The cell parameters (`gi` and `init_infections`) will be overridden
with known values for simulation and with sampled values during inference.
"""

cell = RenewalCell(gi_length)
renewal_model = Lux.Recurrence(cell; return_sequence = true)
rng = default_rng()
ps_default, st = Lux.setup(rng, renewal_model)

md"""
## 3. Simulate ground-truth infections

We use a step-function R(t): R = 1.5 for the first 40 days, R = 0.8 for the
next 40 days, then R = 1.3 for the final 40 days. Seed infections are set to
10.0 per day for the GI length.
"""

T = 120
true_Rt = vcat(fill(1.5, 40), fill(0.8, 40), fill(1.3, 40))

true_init_infections = fill(10.0, gi_length)

true_ps = (; gi = gi_pmf, init_infections = true_init_infections)

# Recurrence expects input shape (features, time, batch)
Rt_input = reshape(true_Rt, 1, T, 1)

output, _ = renewal_model(Rt_input, true_ps, st)

# output is a vector of (1, 1) matrices — extract the scalar time series
latent_infections = [output[t][1] for t in 1:T]

observed_cases = rand.(Poisson.(latent_infections))

md"""
## 4. Plot simulated data
"""

p1 = plot(true_Rt;
    xlabel = "Day", ylabel = "R(t)",
    title = "True Reproduction Number",
    label = "R(t)", lw = 2, legend = :topright
)

p2 = plot(latent_infections;
    xlabel = "Day", ylabel = "Infections",
    title = "Infections",
    label = "Latent (true)", lw = 2, legend = :topright
)
scatter!(p2, observed_cases; label = "Observed (Poisson)", alpha = 0.6, ms = 3)

plot(p1, p2; layout = (2, 1), size = (800, 500))

md"""
## 5. Define a Turing model

We estimate R(t) as independent LogNormal draws per timestep and seed infections
as a single shared value. The GI distribution is fixed to its known value.
"""

@model function fit_renewal(
        observed, gi_pmf, renewal_model, st
)
    T = length(observed)
    gi_length = length(gi_pmf)

    # Priors
    log_Rt ~ filldist(Normal(0.0, 0.5), T)
    Rt = exp.(log_Rt)
    init_val ~ LogNormal(log(10.0), 0.5)

    # Build Lux parameter NamedTuple
    ps = (; gi = gi_pmf, init_infections = fill(init_val, gi_length))

    # Forward pass through the renewal model
    Rt_input = reshape(Rt, 1, T, 1)
    output, _ = renewal_model(Rt_input, ps, st)

    # Likelihood
    for t in 1:T
        expected = output[t][1]
        observed[t] ~ Poisson(max(expected, 1e-8))
    end

    return (; Rt, latent = [output[t][1] for t in 1:T])
end

md"""
## 6. Fit with NUTS
"""

fit_mdl = fit_renewal(observed_cases, gi_pmf, renewal_model, st)
chain_rvdiff = sample(fit_mdl,
    NUTS(; adtype = AutoReverseDiff(; compile = Val(true))), 
    MCMCThreads(),
    4,
    1000)

describe(chain_rvdiff)

md"""
## 7. Plot posterior R(t) estimates against truth
"""

Rt_params = [Symbol("log_Rt[$i]") for i in 1:T]
Rt_posterior = exp.(Array(chain_rvdiff[Rt_params]))  # (n_samples, T)
Rt_mean = vec(mean(Rt_posterior; dims = 1))
Rt_lower = vec(mapslices(x -> quantile(x, 0.025), Rt_posterior; dims = 1))
Rt_upper = vec(mapslices(x -> quantile(x, 0.975), Rt_posterior; dims = 1))

plot(Rt_mean;
    ribbon = (Rt_mean .- Rt_lower, Rt_upper .- Rt_mean),
    xlabel = "Day", ylabel = "R(t)",
    title = "Posterior R(t) Estimates",
    label = "Posterior mean (95% CI)",
    lw = 2, fillalpha = 0.2, legend = :topright
)
plot!(true_Rt; label = "True R(t)", lw = 2, ls = :dash, color = :red)
