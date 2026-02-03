#src using Pkg
#src Pkg.activate(joinpath(@__DIR__, ".."))

using CensoredDistributions
using Distributions, LinearAlgebra, Markdown
using JET: @report_opt
using Random: seed!
using StatsPlots
using Turing
using ReverseDiff, Mooncake
using RenewalExamples: UnknownGI
using Enzyme: set_runtime_activity, Reverse, Forward

seed!(1234)

md"""
# Renewal Process Examples

This notebook demonstrates how to simulate renewal models in various ways.

## Defining an unknown generation interval distribution

We can define an unknown generation interval (GI) distribution using the `UnknownGI` struct.
`UnknownGI` allows us to specify a distribution type along with prior distributions for its parameters, as well as censoring parameters in anticipation of double interval censoring.
"""

ugi = UnknownGI(
    dist_type = Gamma,
    param_priors = (a = LogNormal(log(3.0), 1.0), b = truncated(Normal(4.0, 0.5), 0, Inf)),
    interval = 1.0,
    upper = 40.0
);

md"""
A sample from `ugi` generates a random generation interval distribution based on the specified priors.
"""

p = plot(;
    xlabel = "Generation Interval", ylabel = "Density",
    title = "Sampled Generation Interval Distributions"
)

for i in 1:5
    gi_dist = rand(ugi)
    plot!(p, gi_dist, label = "Sample $i", lw = 2, alpha = 0.7)
end
p

md"""
The sampling is type-stable, see [Modern Julia Workflows](https://modernjuliaworkflows.org/optimizing/#type_stability) for more information.
"""
@report_opt rand(ugi)

md"""
Any particular parameter values can be used to create a double interval censored distribution using the `double_interval_censored` method for `UnknownGI`.
"""
true_shape_param = 3.0
true_scale_param = 4.0
censored_gi = double_interval_censored(ugi, true_shape_param, true_scale_param) # shape=3.0, scale=4.0 

md"""
### Inference on the generation interval distribution

Often at the start of an outbreak, the generation interval distribution is not known and we have a joint estimation problem:

- Estimate the parameters of the generation interval distribution.
- Estimate the time-varying reproduction number.

We'll start by looking at a simpler example where we have observed some (censored) generation intervals from early traced infector-infectee pairs.
We set up a `Turing.jl` model to estimate the parameters of the generation interval distribution given these observed (censored) intervals.
"""

simulated_data_seqn = rand(censored_gi, 100)
vals = unique(simulated_data_seqn)
simulated_data_binned = [count(==(val), simulated_data_seqn) for val in vals]

@model function fit_gi(
        vals, binned_data, ugi::UnknownGI{DT, P, T}
    ) where {DT, P <: NamedTuple, T <: Real}
    
    params ~ arraydist(collect(values(ugi.param_priors))) # Sample parameters from priors
    censored_gi = double_interval_censored(ugi, params...) # Create censored GI distribution

    vals ~ weight(censored_gi, binned_data)     # Vectorized weighted likelihood c.f `CensoredDistributions.jl` 
    return
end

md"""
Now we can fit the generation interval distribution to the simulated (censored) data.
"""
fit_mdl = fit_gi(vals, simulated_data_binned, ugi);
chain = sample(fit_mdl, NUTS(), 1000);
describe(chain)

md"""
#### Other Automatic Differentiation backends
We can also use other AD backends supported by Turing.jl.
Types we try here include:

    - `ReverseDiff` with forward pass compilation
    - `Mooncake` (a source-to-source AD backend)
    - `Enzyme` in forward mode with runtime activity allowed
"""

adtypes = (
    forward_diff = AutoForwardDiff(),
    compile_revdiff = AutoReverseDiff(; compile = Val(true)),
    runtime_enzyme_forward = AutoEnzyme(; mode = set_runtime_activity(Forward)),
)

function fit_combo(adtypes, model; n_samples = 10, n_warmup = 5)
    names = collect(keys(adtypes))
    n_backends = length(names)
    results = Vector{Union{Nothing, @NamedTuple{chain::Any, wall_time::Float64}}}(nothing, n_backends)
    
    @info "Warming up AD backends (n=$n_warmup samples each)..."
    for (name, adtype) in pairs(adtypes)
        try
            sampler = NUTS(; adtype = adtype)
            sample(model, sampler, n_warmup)
            @info "  Warmed up $name"
        catch e
            @warn "  Warmup failed for $name" exception = e
        end
    end
    
    @info "Running timed fits (n=$n_samples samples each)..."
    Threads.@threads for i in eachindex(names)
        name = names[i]
        adtype = adtypes[name]
        @info "Starting fit with AD backend: $name on thread $(Threads.threadid())"
        start_time = time()
        try
            sampler = NUTS(; adtype = adtype)
            chain = sample(model, sampler, n_samples)
            wall_time = time() - start_time
            results[i] = (; chain, wall_time)
            @info "Completed $name in $(round(wall_time; digits=2))s"
        catch e
            @error "Error fitting with AD backend $name" exception = (e, catch_backtrace())
        end
    end
    return Dict(names[i] => results[i] for i in eachindex(names) if !isnothing(results[i]))
end

results = fit_combo(adtypes, fit_mdl; n_samples = 2000, n_warmup = 10);

md"""
The results dictionary contains the fitted chains and wall times for each AD backend used.
"""

wall_times = Dict(name => r.wall_time for (name, r) in results)
@info "Wall times (seconds):" wall_times

md"""
Unfortunately, at the time of writing, these adtypes failed and I couldn't debug quickly:

- `AutoEnzyme(; mode = set_runtime_activity(Reverse))` 
- `AutoMooncake()`
"""