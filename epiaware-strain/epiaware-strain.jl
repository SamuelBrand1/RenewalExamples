## Activate the environment and instantiate the dependencies

cd(@__DIR__)
using Pkg
Pkg.activate(".")
Pkg.instantiate()

## Deps
using ComposableTuringIDModels, Distributions
using BSplineKit
using Turing
using ReverseDiff                       # loads Turing's AutoReverseDiff backend
import DynamicPPL
using DynamicPPL: @model, to_submodel, @addlogprob!
using CSV, DataFrames, Dates
using CairoMakie
using LinearAlgebra, Random, Statistics
import ComposableTuringIDModels: PriorLike

#=
# Matching EpiStrainDynamics by composition

`EpiStrainDynamics` ships six Stan programs — `{rw, ps} × {single, multiple,
subtyped}` — that are almost entirely duplicated source. Reading them side by
side, they vary along exactly **two orthogonal axes** over one shared latent
process and one shared observation model:

  1. **Smoothing.** `rw_*` puts a second-order random walk directly on the data
     grid (`matrix[num_path, num_data] a`); `ps_*` puts the *same* RW2 on
     B-spline coefficients (`matrix[num_path, num_basis] a`) and maps it up with
     `a_new = a * B`. Nothing else differs between `rw_multiple.stan` and
     `ps_multiple.stan`.
  2. **Pathogen structure.** How the strain intensities are split into
     multinomial observations: not at all (`single`), one multinomial over all
     strains (`multiple`), or two nested multinomials (`subtyped`).

Everything else — the RW2 prior, `exp()`, the summed total, the negative
binomial on total cases, the day-of-week simplex — is common. So instead of six
programs this is one composed model with two swappable components, and the six
Stan variants are six argument combinations.
=#

## ---------------------------------------------------------------------------
## Latent process: a multivariate random walk
## ---------------------------------------------------------------------------

#=
`ComposableTuringIDModels.RandomWalk` is scalar-path-only: `only(rw_init)` forces
a scalar initial value, and `accumulate_scan`'s default `get_state` assembles
with `vcat(initial_state, last.(state))`, which collapses vector states to their
last component. The package has no multivariate latent process at all — the
whole latent contract is "length-`n` vector of reals", and `AbstractVector` is
its type-level stand-in for "a path over time", so a K-vector is ambiguous with
a path everywhere it matters (`_path_prior`, `_at`, `_prior_order`).

So this is the one genuinely new component. It returns a `K × n` matrix. That
still satisfies `implements_latent_interface`, which only requires
`as_turing_model(model, n)` to return a `DynamicPPL.Model` — it places no
constraint on the shape of the returned value. Nothing existing is broken.
=#

struct MVRandomWalk{D <: PriorLike, E <: PriorLike} <: AbstractLatentModel
    "Prior for the initial values. Must draw `K × d`, e.g. `IID(MvNormal(K, σ))` (a bare `MvNormal` suffices when `d == 1`)."
    init::D
    "Error model for the `d`-th differences. Must draw a `K × (n-d)` path, e.g. `IID(MvNormal(K, σ))`."
    ϵ_t::E
    "Differencing order. `1` is a plain random walk, `2` is a second-order walk."
    d::Int
end

# No `_path_prior` on `ϵ_t`. The package convention wraps a bare `Distribution` in
# an `Intercept` — one draw broadcast to every step — which for an `MvNormal`
# means a constant increment, never innovations. Since a K-vector and a length-K
# scalar path are indistinguishable to that machinery, the innovations are asked
# for explicitly (`IID`) rather than adapted.
function MVRandomWalk(; init, ϵ_t, d::Int = 1)
    @assert d>0 "d must be greater than 0"
    return MVRandomWalk(init, ϵ_t, d)
end

@model function ComposableTuringIDModels.as_turing_model(model::MVRandomWalk, n)
    d = model.d
    @assert n>d "n must be greater than d"
    rw_init ~ as_turing_submodel(model.init, d; prefix = true)   # K × d
    ϵ_t ~ as_turing_submodel(model.ϵ_t, n - d)                   # K × (n-d)
    I0 = rw_init isa AbstractVector ? reshape(rw_init, :, 1) : rw_init
    # Order-`d` random walk down the time axis. One `cumsum` per differencing
    # order, each prepending its own init column, so the `d`-th differences are
    # exactly `ϵ_t`: `d = 1` is Z[:, t] = Z[:, t-1] + ϵ, and `d = 2` is
    # ps_multiple.stan's `a[,i] ~ N(2a[,(i-1)] - a[,(i-2)], ·)`.
    #
    # Prepending one column per level (rather than all `d` at once, then
    # `cumsum`-ing `d` times) keeps the init columns *interpretable*: column 1 is
    # the starting level Z[:, 1], column 2 the starting slope Z[:, 2] - Z[:, 1],
    # and so on. Folding them together instead makes each init column act on the
    # walk with weight growing in `n`, so any prior wide enough to be vague on
    # the level implies an astronomically wide prior at the end of the series.
    Z = ϵ_t
    for j in d:-1:1
        Z = cumsum(hcat(I0[:, j], Z); dims = 2)
    end
    return Z                                                     # K × n
end

#=
The multivariate analogue of the package's `HierarchicalNormal`: a non-centred
innovation process with an *inferred* scale, which is what the Stan programs do
(`tau` is a parameter, not data). Covers two of Stan's three `cov_structure`
settings:

  - `std` a scalar `Distribution`  → `cov_structure = 0` (one shared τ)
  - `std` a length-K vector        → `cov_structure = 1` (per-strain τ)

`cov_structure = 2` (a full `cov_matrix[num_path] Sigma`, correlated innovations
across strains) would swap the standard-normal draw for an LKJ-parameterised
Cholesky factor; the shape contract here already accommodates it.
=#

struct MVHierarchicalNormal{S <: PriorLike} <: AbstractLatentModel
    "Number of strains K."
    K::Int
    "Prior for the innovation standard deviation τ."
    std::S
end

function MVHierarchicalNormal(K::Int; std = truncated(Normal(0, 0.1), 0, Inf))
    return MVHierarchicalNormal(K, std)
end

@model function ComposableTuringIDModels.as_turing_model(
        model::MVHierarchicalNormal, n)
    std ~ as_turing_submodel(model.std, model.K; prefix = true)
    ϵ_t ~ as_turing_submodel(IID(MvNormal(model.K, 1.0)), n)      # K × n, standard
    # Broadcasts either way: a scalar τ scales everything, a K-vector τ scales
    # down the rows (one scale per strain).
    return std .* ϵ_t
end

## ---------------------------------------------------------------------------
## Axis 1: smoothing — where the random walk lives
## ---------------------------------------------------------------------------

"""
    bspline_basis(X, knots; degree = 3)

Build the `num_basis × length(X)` B-spline design matrix `B`, the analogue of
`ps_*.stan`'s `transformed data` block. Rows are basis functions, columns are
evaluation points, so a `K × num_basis` coefficient matrix `a` gives the smoothed
per-strain log-intensities as `a * B`.

`num_basis == length(knots) + degree - 1`, matching Stan. `BSplineBasis` clamps
the boundary knots itself, so Stan's manual `ext_knots` padding is not needed.

Unlike the Stan implementation this is correct at the right-hand boundary:
Stan's order-1 base case uses a half-open interval `[k_i, k_{i+1})`, so every
basis function is zero at `X[end] == knots[end]` and that column sums to 0
instead of 1 — their final observation gets `a_new[:, end] == 0` and hence a
uniform multinomial regardless of the data. Here the final column is a valid
partition of unity.
"""
function bspline_basis(X, knots; degree::Int = 3)
    # `BSplineBasis` takes ownership of the knot vector and augments it in place,
    # so hand it a copy.
    basis = BSplineBasis(BSplineOrder(degree + 1), collect(float.(knots)))
    B = [basis[i](x) for i in eachindex(basis), x in X]
    @assert size(B, 1)==length(knots) + degree - 1 "num_basis must equal length(knots) + degree - 1"
    return B
end

abstract type Smoothing end

"Second-order random walk directly on the data grid — the `rw_*.stan` programs."
struct RandomWalkSmoothing <: Smoothing end

"""
Second-order random walk on B-spline coefficients, mapped up by `B` — the
`ps_*.stan` programs. `B` is `num_basis × num_data`, from [`bspline_basis`](@ref).
"""
struct SplineSmoothing{M <: AbstractMatrix} <: Smoothing
    B::M
end

# How many columns the walk itself has: the data grid, or the coefficient grid.
n_latent(::RandomWalkSmoothing, n_data) = n_data
n_latent(s::SplineSmoothing, n_data) = size(s.B, 1)

# `TransformLatentModel` is shape-agnostic (`return model.transform(untransformed)`
# — no vector assumption anywhere), so the package's own modifier carries the
# K × num_basis matrix straight through. This is the whole `rw` → `ps` difference.
apply_smoothing(::RandomWalkSmoothing, walk) = walk
apply_smoothing(s::SplineSmoothing, walk) = TransformLatentModel(walk, a -> a * s.B)

## ---------------------------------------------------------------------------
## Axis 2: pathogen structure — how strains are split into multinomials
## ---------------------------------------------------------------------------

abstract type PathogenStructure end

"One pathogen, no strain decomposition — the `*_single.stan` programs."
struct SinglePathogen <: PathogenStructure end

"`K` strains observed through one multinomial — the `*_multiple.stan` programs."
struct MultiplePathogens <: PathogenStructure
    K::Int
end

"""
`K` strains where the first `n_subtypes` rows are subtypes of a single reported
group — the `*_subtyped.stan` programs. Two nested multinomials: `P1` over
(aggregated group, remaining strains) and `P2` over the subtypes within the
group. For the influenza data the subtypes are H3N2 and H1N1 within influenza A.
"""
struct SubtypedPathogens <: PathogenStructure
    K::Int
    n_subtypes::Int
end

n_strains(::SinglePathogen) = 1
n_strains(s::MultiplePathogens) = s.K
n_strains(s::SubtypedPathogens) = s.K

"""
    strain_split(structure, λ) -> (total, probs)

Given the `K × T` intensity matrix `λ = exp.(a)`, return the summed total series
and a `NamedTuple` of `strains × T` multinomial probability matrices — one entry
per multinomial in the corresponding Stan program.

Normalising a column of `λ` by its own sum is exactly a softmax down the strain
axis, which is what the Stan code writes as `exp(a[,i])/sum(exp(a[,i]))`. There
is no simplex *parameter* to sample, so the package having no `Multinomial`
observation model or `Dirichlet` prior costs nothing here.
"""
function strain_split(::SinglePathogen, λ)
    return vec(sum(λ; dims = 1)), NamedTuple()
end

function strain_split(::MultiplePathogens, λ)
    total = vec(sum(λ; dims = 1))
    return total, (; P = λ ./ total')
end

function strain_split(s::SubtypedPathogens, λ)
    total = vec(sum(λ; dims = 1))
    ns = s.n_subtypes
    total_group = vec(sum(λ[1:ns, :]; dims = 1))
    # Stan's `theta`: the aggregated subtype group first, then the other strains.
    θ = vcat(total_group', λ[(ns + 1):end, :])
    return total, (; P1 = θ ./ total', P2 = λ[1:ns, :] ./ total_group')
end

## ---------------------------------------------------------------------------
## The unified model
## ---------------------------------------------------------------------------

"""
    EpiStrainModel(structure, smoothing; kwargs...)

One composed model covering all six `EpiStrainDynamics` Stan programs. Pick a
[`PathogenStructure`](@ref) and a [`Smoothing`](@ref); everything else is shared.

  - `SinglePathogen()`   + `RandomWalkSmoothing()` → `rw_single.stan`
  - `SinglePathogen()`   + `SplineSmoothing(B)`    → `ps_single.stan`
  - `MultiplePathogens(K)` + `RandomWalkSmoothing()` → `rw_multiple.stan`
  - `MultiplePathogens(K)` + `SplineSmoothing(B)`    → `ps_multiple.stan`
  - `SubtypedPathogens(K, n)` + either              → `rw_subtyped.stan` / `ps_subtyped.stan`

`week_effect` matches Stan: `1` disables the day-of-week effect, otherwise a flat
`Dirichlet` simplex of that length scaled by `week_effect` (so a uniform simplex
is a no-op).
"""
struct EpiStrainModel{S <: PathogenStructure, M <: Smoothing,
    L <: PriorLike, E <: PriorLike, F <: PriorLike}
    structure::S
    smoothing::M
    "Prior for the two RW2 initial columns."
    init::L
    "Innovation process for the RW2 second differences."
    innovations::E
    "Prior for the negative-binomial dispersion φ."
    φ::F
    "Day-of-week period: 1 for none, else the simplex length (7 for daily data)."
    week_effect::Int
end

function EpiStrainModel(structure, smoothing;
        log_level = log(500.0), init = nothing, innovations = nothing,
        φ = truncated(Normal(0, 10), 0, Inf), week_effect::Int = 1)
    K = n_strains(structure)
    # Column 1 is the starting log-intensity per strain, column 2 the starting
    # slope — see the note in `MVRandomWalk`. They need very different scales, so
    # this is a per-column prior rather than one `IID`.
    #
    # `product_distribution` rather than a `Vector{<:Distribution}`: the package's
    # vector-prior seam tests homogeneity with `all(first(v) .== v)`, and
    # broadcasting `==` over an `MvNormal` tries to iterate it and throws. Handing
    # it a single (matrix-variate) `Distribution` goes through the scalar seam
    # instead and draws the `K × d` block in one go.
    init = isnothing(init) ?
           product_distribution([MvNormal(fill(log_level, K), 1.0),
        MvNormal(K, 0.05)]) : init
    # A second-order walk accumulates variance like τ²n³/3, so the innovation
    # scale needs a much tighter prior than a first-order one would.
    innovations = isnothing(innovations) ?
                  MVHierarchicalNormal(K; std = truncated(Normal(0, 0.05), 0, Inf)) :
                  innovations
    return EpiStrainModel(structure, smoothing, init, innovations, φ, week_effect)
end

"""
    as_turing_model(model::EpiStrainModel, data)

`data` is a `NamedTuple` with `Y` (total counts), `P` (a `NamedTuple` of
`strains × T` count matrices matching `strain_split`'s keys), `n_data`, and `DOW`
(1-based day/week index, only used when `week_effect > 1`).

`Y` is a **model argument** defaulting to `data.Y`, not a field read inside the
body: DynamicPPL decides observed-vs-sampled from the tilde's left-hand name
matching an argument, so `Y ~ ...` reading `data.Y` from the body would silently
sample a fresh `Y` and never condition on the data at all. Pass `missing`
explicitly for a prior predictive draw.
"""
@model function ComposableTuringIDModels.as_turing_model(
        model::EpiStrainModel, data, Y = data.Y)
    # --- latent: RW2, on the data grid or the spline coefficient grid ---------
    walk = MVRandomWalk(; init = model.init, ϵ_t = model.innovations, d = 2)
    latent = apply_smoothing(model.smoothing, walk)
    a ~ as_turing_submodel(latent, n_latent(model.smoothing, data.n_data))
    λ = exp.(a)                                   # K × n_data
    total, probs = strain_split(model.structure, λ)

    # --- day-of-week: Stan's `simplex[week_effect] day_of_week_simplex` -------
    if model.week_effect == 1
        μ = total
    else
        dow ~ Dirichlet(ones(model.week_effect))
        μ = total .* (model.week_effect .* dow[data.DOW])
    end

    # --- total counts: Stan's neg_binomial(μ*φ, φ) ---------------------------
    # Stan's `neg_binomial(α, β)` has mean α/β, so α = μφ, β = φ gives mean μ and
    # variance μ(1 + 1/φ). In Distributions.jl that is NegativeBinomial(r, p)
    # with r = μφ and p = φ/(1+φ).
    φ ~ model.φ
    p = φ / (1 + φ)
    Y ~ product_distribution([NegativeBinomial(μ[t] * φ, p) for t in eachindex(μ)])

    # --- strain composition: one multinomial per entry in `probs` ------------
    # The counts are always data (never sampled), so these enter as explicit
    # likelihood terms rather than tildes — that keeps the trial counts read from
    # the data and sidesteps sampling a slice-indexed variable.
    for k in keys(probs)
        Pk, pk = data.P[k], probs[k]
        for t in axes(pk, 2)
            n_t = sum(view(Pk, :, t))
            n_t > 0 && @addlogprob! logpdf(Multinomial(n_t, pk[:, t]), Pk[:, t])
        end
    end

    return (; a, λ, total, probs, μ)
end

## ---------------------------------------------------------------------------
## Data: WHO Global Influenza Programme, Australia (the package's `influenza`)
## ---------------------------------------------------------------------------

const DATA_URL = "https://raw.githubusercontent.com/acefa-hubs/EpiStrainDynamics/main/data-raw/aus_influenza_data.csv"
const DATA_CSV = joinpath(@__DIR__, "aus_influenza_data.csv")

"""
    influenza_data(; from, to)

Reproduce `EpiStrainDynamics`' `influenza` dataset from `data-raw/influenza.R`:
weekly Australian ILI counts with influenza specimens split into A (subtyped
into H3N2 / H1N1), B, and other.

Strain rows are ordered `[H3N2, H1N1, B, other]` so that the first two are the
subtypes of influenza A, matching `*_subtyped.stan`'s requirement that "the first
two rows/pathogens will be influenza A H3N2 and influenza A H1N1".
"""
function influenza_data(; from = Date(2012, 1, 1), to = Date(2020, 3, 1))
    isfile(DATA_CSV) || download(DATA_URL, DATA_CSV)
    df = CSV.read(DATA_CSV, DataFrame)
    df.week = Date.(df.week, dateformat"d/m/y")
    df.other = df.num_spec .- df.inf_all
    df = sort(df[(df.week .>= from) .& (df.week .< to), :], :week)

    Y = Vector{Int}(df.ili)
    # P2: the two influenza A subtypes. P1: influenza A as a whole, then B, then
    # other — Stan's `theta`, hence `num_path - 1` rows.
    P2 = permutedims(hcat(df.inf_H3N2, df.inf_H1N1))
    P1 = permutedims(hcat(df.inf_A, df.inf_B, df.other))
    return (; week = df.week, Y, P1, P2,
        n_data = length(Y), DOW = ones(Int, length(Y)))
end

flu = influenza_data(; from = Date(2015, 1, 1), to = Date(2020, 3, 1))
K = 4                                    # H3N2, H1N1, B, other
println("influenza: $(flu.n_data) weeks, $(first(flu.week)) → $(last(flu.week))")

## ---------------------------------------------------------------------------
## The four model variants
## ---------------------------------------------------------------------------

# Spline basis: a knot every ~8 weeks, cubic — the `ps_*` coefficient grid.
knots = collect(range(1.0, float(flu.n_data); length = max(4, flu.n_data ÷ 8)))
B = bspline_basis(1.0:flu.n_data, knots)
println("spline basis: $(size(B, 1)) basis functions × $(size(B, 2)) weeks")

subtyped = SubtypedPathogens(K, 2)
multiple = MultiplePathogens(K)

models = (
    rw_single = EpiStrainModel(SinglePathogen(), RandomWalkSmoothing()),
    ps_single = EpiStrainModel(SinglePathogen(), SplineSmoothing(B)),
    rw_subtyped = EpiStrainModel(subtyped, RandomWalkSmoothing()),
    ps_subtyped = EpiStrainModel(subtyped, SplineSmoothing(B)),
    # the remaining two Stan programs fall out of the same two axes for free
    rw_multiple = EpiStrainModel(multiple, RandomWalkSmoothing()),
    ps_multiple = EpiStrainModel(multiple, SplineSmoothing(B)))

# Data shaped for each structure. `SinglePathogen` has no multinomial at all.
data_for(::SinglePathogen) = (; flu.n_data, flu.DOW, flu.Y, P = NamedTuple())
data_for(::MultiplePathogens) = (; flu.n_data, flu.DOW, flu.Y,
    P = (; P = vcat(flu.P2, flu.P1[2:end, :])))     # H3N2, H1N1, B, other
data_for(::SubtypedPathogens) = (; flu.n_data, flu.DOW, flu.Y,
    P = (; P1 = flu.P1, P2 = flu.P2))

## Every variant builds and returns the right shapes. The latent parameters are
## prior draws; `Y` stays observed so nothing tries to sample counts from a
## second-order walk's prior, whose cubic variance growth overflows `Int64`.
let rng = Xoshiro(1)
    for (name, m) in pairs(models)
        gq = ComposableTuringIDModels.as_turing_model(m, data_for(m.structure))(rng)
        println(rpad(name, 12), " a = ", size(gq.a), "   total = ", size(gq.total),
            "   probs = ", NamedTuple(k => size(v) for (k, v) in pairs(gq.probs)))
    end
end

## ---------------------------------------------------------------------------
## Fit
## ---------------------------------------------------------------------------

model = models.ps_subtyped                      # ps_subtyped.stan
data = data_for(model.structure)
mdl = ComposableTuringIDModels.as_turing_model(model, data)

# 142 parameters (4×2 init, 4×33 innovations, τ, φ), so reverse-mode AD. Bump the
# iterations for real use — this is sized to run in a few minutes.
chn = sample(Xoshiro(2), mdl,
    NUTS(200, 0.8; adtype = AutoReverseDiff(; compile = true)), 400; progress = false)

## ---------------------------------------------------------------------------
## Makie plots
## ---------------------------------------------------------------------------

"Posterior draws of a generated quantity, as a `draws × T` matrix."
function gq_matrix(mdl, chn, f)
    draws = DynamicPPL.returned(mdl, chn)
    return reduce(hcat, [f(g) for g in vec(draws)])'
end

"Median and a `q`-central credible band down the draw axis."
function band_stats(M; q = 0.9)
    lo, hi = (1 - q) / 2, 1 - (1 - q) / 2
    return (med = [median(view(M, :, t)) for t in axes(M, 2)],
        lower = [quantile(view(M, :, t), lo) for t in axes(M, 2)],
        upper = [quantile(view(M, :, t), hi) for t in axes(M, 2)])
end

strain_names = ["A(H3N2)", "A(H1N1)", "B", "other"]

# Makie's `band!` has no `Date` conversion (`scatter!`/`lines!` do), so plot
# against the week index and label the ticks with the calendar years.
x = 1:flu.n_data
date_ticks = let years = unique(year.(flu.week))
    pos = [findfirst(w -> year(w) == y, flu.week) for y in years]
    keep = .!isnothing.(pos)
    (Int.(pos[keep]), string.(years[keep]))
end

fig = Figure(; size = (1000, 780))

# --- total ILI -------------------------------------------------------------
tot = band_stats(gq_matrix(mdl, chn, g -> g.μ))
ax1 = Axis(fig[1, 1]; title = "Total ILI — ps_subtyped (EpiStrainDynamics influenza data)",
    ylabel = "weekly ILI cases", xticks = date_ticks)
band!(ax1, x, tot.lower, tot.upper; color = (:steelblue, 0.3), label = "90% CrI")
lines!(ax1, x, tot.med; color = :steelblue, linewidth = 2, label = "posterior median")
scatter!(ax1, x, flu.Y; color = (:black, 0.55), markersize = 4, label = "observed")
axislegend(ax1; position = :lt, framevisible = false)

# --- strain proportions ----------------------------------------------------
# P1 rows are (influenza A, B, other); expand A back into its two subtypes using
# P2 so the plot shows all four strains on one axis.
obs_prop = let
    p1 = flu.P1 ./ sum(flu.P1; dims = 1)
    p2 = flu.P2 ./ max.(sum(flu.P2; dims = 1), 1)
    vcat(p2 .* p1[1:1, :], p1[2:end, :])
end

ax2 = Axis(fig[2, 1]; title = "Strain composition",
    xlabel = "year", ylabel = "proportion of ILI specimens", xticks = date_ticks)
palette = Makie.wong_colors()
# The full-strain proportions come straight off the latent intensities, which is
# the same softmax the nested multinomials are built from.
prop_k(g, k) = vec(g.λ[k, :] ./ vec(sum(g.λ; dims = 1)))
for k in 1:K
    st = band_stats(gq_matrix(mdl, chn, g -> prop_k(g, k)))
    band!(ax2, x, st.lower, st.upper; color = (palette[k], 0.25))
    lines!(ax2, x, st.med; color = palette[k], linewidth = 2, label = strain_names[k])
    scatter!(ax2, x, obs_prop[k, :]; color = (palette[k], 0.5), markersize = 3)
end
axislegend(ax2; position = :lt, framevisible = false, nbanks = 4)

save(joinpath(@__DIR__, "epistrain-fit.png"), fig)
fig
