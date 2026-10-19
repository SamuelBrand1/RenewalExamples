#src using Pkg
#src Pkg.activate(joinpath(@__DIR__, ".."))

using Distributions, LinearAlgebra, Markdown
using Random: default_rng, seed!
using StatsPlots
using Turing
using Lux
using RenewalExamples: RenewalCell, delay_conv
using ReverseDiff

seed!(1234)

md"""
# Renewal Model with Observation Delay

This notebook extends the basic renewal model by adding a delay convolution
between latent infections and observed cases. The full generative model is:

    I(t) = R(t) * sum_{s} g(s) * I(t - s)          (renewal equation)
    E_obs(t) = sum_{s} d(s) * I(t - s)              (delay convolution)
    C(t) ~ Poisson(E_obs(t))                         (observation noise)

Both steps are Lux layers composed in a `Chain`:
- `Recurrence(RenewalCell(...))` for the renewal dynamics
- `delay_conv(...)` for the infection-to-observation delay (1D causal Conv)
"""

md"""
## 1. Define generation interval and delay distributions

We discretise both distributions into PMF vectors.
"""

gi_dist = Gamma(3.0, 2.0)
gi_length = 15
gi_pmf = [cdf(gi_dist, s) - cdf(gi_dist, s - 1) for s in 1:gi_length]
gi_pmf = gi_pmf ./ sum(gi_pmf)

delay_dist = LogNormal(log(5.0), 0.5)
delay_length = 20
delay_pmf = [cdf(delay_dist, s) - cdf(delay_dist, s - 1) for s in 1:delay_length]
delay_pmf = delay_pmf ./ sum(delay_pmf)

p1 = bar(gi_pmf;
    xlabel = "Days", ylabel = "Probability",
    title = "Generation Interval", label = "GI PMF"
)
p2 = bar(delay_pmf;
    xlabel = "Days", ylabel = "Probability",
    title = "Observation Delay", label = "Delay PMF"
)
plot(p1, p2; layout = (1, 2), size = (800, 300))

md"""
## 2. Build the full Lux model

The model chains `Recurrence` (renewal dynamics) with `delay_conv` (observation
delay). The delay convolution uses NNlib's 1D `Conv` with causal padding.
"""

rng = default_rng()

full_model = Lux.Chain(
    Lux.Recurrence(RenewalCell(gi_length); return_sequence = true),
    delay_conv(delay_length),  # nested Chain: WrappedFunction → WrappedFunction → Conv
)
ps_default, st = Lux.setup(rng, full_model)

md"""
## 3. Simulate ground-truth data

R(t) follows a step function: 1.5 → 0.8 → 1.3 over 120 days.
We override the Lux parameters with known values and run the full pipeline.
"""

T = 120
true_Rt = vcat(fill(1.5, 40), fill(0.8, 40), fill(1.3, 40))
true_init_infections = fill(10.0, gi_length)

# Override parameters: layer_1 is Recurrence, layer_2 is the delay_conv Chain
true_ps = (;
    layer_1 = (; gi = gi_pmf, init_infections = true_init_infections),
    layer_2 = (;
        layer_1 = NamedTuple(),  # WrappedFunction (stack)
        layer_2 = NamedTuple(),  # WrappedFunction (reshape)
        layer_3 = (; weight = reshape(delay_pmf, delay_length, 1, 1)),  # Conv kernel
    ),
)

Rt_input = reshape(true_Rt, 1, T, 1)
output, _ = full_model(Rt_input, true_ps, st)

# output is (T, 1, 1) — extract expected observations
expected_obs = vec(output)

# Also extract latent infections for comparison
recurrence = Lux.Recurrence(RenewalCell(gi_length); return_sequence = true)
_, st_rec = Lux.setup(rng, recurrence)
inf_output, _ = recurrence(Rt_input, true_ps.layer_1, st_rec)
latent_infections = [inf_output[t][1] for t in 1:T]

observed_cases = rand.(Poisson.(max.(expected_obs, 1e-8)))

md"""
## 4. Plot simulated data
"""

p1 = plot(true_Rt;
    xlabel = "Day", ylabel = "R(t)",
    title = "True Reproduction Number",
    label = "R(t)", lw = 2, legend = :topright
)

p2 = plot(latent_infections;
    xlabel = "Day", ylabel = "Count",
    title = "Infections and Observed Cases",
    label = "Latent infections", lw = 2, legend = :topright
)
plot!(p2, expected_obs; label = "Expected obs (after delay)", lw = 2, ls = :dash)
scatter!(p2, observed_cases; label = "Observed (Poisson)", alpha = 0.5, ms = 3)

plot(p1, p2; layout = (2, 1), size = (800, 500))

md"""
## 5. Define a Turing model

We estimate R(t) and seed infections. Both the GI distribution and the delay
distribution are fixed to their known values.
"""

@model function fit_renewal_delay(
        observed, gi_pmf, delay_pmf, full_model, st
)
    T = length(observed)
    gi_length = length(gi_pmf)
    delay_length = length(delay_pmf)

    # Priors
    log_Rt ~ filldist(Normal(0.0, 0.5), T)
    Rt = exp.(log_Rt)
    init_val ~ LogNormal(log(10.0), 0.5)

    # Build Lux parameter NamedTuple matching the Chain structure
    ps = (;
        layer_1 = (; gi = gi_pmf, init_infections = fill(init_val, gi_length)),
        layer_2 = (;
            layer_1 = NamedTuple(),
            layer_2 = NamedTuple(),
            layer_3 = (; weight = reshape(delay_pmf, delay_length, 1, 1)),
        ),
    )

    # Forward pass: renewal → delay conv → expected observations
    Rt_input = reshape(Rt, 1, T, 1)
    output, _ = full_model(Rt_input, ps, st)
    expected = vec(output)

    # Likelihood
    for t in 1:T
        observed[t] ~ Poisson(max(expected[t], 1e-8))
    end

    return (; Rt, expected)
end

md"""
## 6. Fit with NUTS
"""

fit_mdl = fit_renewal_delay(observed_cases, gi_pmf, delay_pmf, full_model, st)
chain = sample(fit_mdl,
    NUTS(; adtype = AutoReverseDiff(; compile = Val(true))),
    MCMCThreads(),
    1000,
    4)

describe(chain)

md"""
## 7. Plot posterior R(t) estimates against truth
"""

Rt_params = [Symbol("log_Rt[$i]") for i in 1:T]
Rt_posterior = exp.(Array(chain[Rt_params]))
Rt_mean = vec(mean(Rt_posterior; dims = 1))
Rt_lower = vec(mapslices(x -> quantile(x, 0.025), Rt_posterior; dims = 1))
Rt_upper = vec(mapslices(x -> quantile(x, 0.975), Rt_posterior; dims = 1))

plot(Rt_mean;
    ribbon = (Rt_mean .- Rt_lower, Rt_upper .- Rt_mean),
    xlabel = "Day", ylabel = "R(t)",
    title = "Posterior R(t) Estimates (with observation delay)",
    label = "Posterior mean (95% CI)",
    lw = 2, fillalpha = 0.2, legend = :topright
)
plot!(true_Rt; label = "True R(t)", lw = 2, ls = :dash, color = :red)
