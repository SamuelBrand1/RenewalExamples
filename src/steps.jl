"""
    RenewalCell <: Lux.AbstractRecurrentCell

Recurrent cell for the renewal equation. At each timestep `t`:

    I(t) = R(t) * dot(g, state)

where `state` carries the most recent infections (shape `(gi_length, batch)`)
and `g` is the generation interval distribution vector (stored as a parameter).

The initial infections are stored as a trainable parameter `init_infections`
(shape `(gi_length,)`), which is broadcast to batch size on the first timestep.
This makes seed infections learnable / sampleable in e.g. Turing.

Designed to be wrapped with `Lux.Recurrence` for scanning over a sequence of
reproduction numbers.

# Example

```julia
using Lux, Random, RenewalExamples
rng = Random.default_rng()
cell = RenewalCell(10)
model = Lux.Recurrence(cell; return_sequence=true)
(ps, st) = Lux.setup(rng, model)
# Set GI to actual distribution and provide R(t) as (1, time, batch)
Rts = cumsum(randn(10)) |> x -> reshape(x, 1, 10, 1)
output, st = model(Rts, ps, st)
```
"""
struct RenewalCell <: Lux.AbstractRecurrentCell
    gi_length::Int
end

function LuxCore.initialparameters(rng::AbstractRNG, cell::RenewalCell)
    gi = ones(Float32, cell.gi_length) ./ cell.gi_length
    init_infections = ones(Float32, cell.gi_length)
    return (; gi, init_infections)
end

function LuxCore.initialstates(rng::AbstractRNG, ::RenewalCell)
    return NamedTuple()
end

# First timestep: no carry provided, initialise state from trainable parameter
function (cell::RenewalCell)(x::AbstractMatrix, ps, st::NamedTuple)
    # Broadcast (gi_length,) → (gi_length, batch)
    state = repeat(ps.init_infections, 1, size(x, 2))
    return cell((x, (state,)), ps, st)
end

# Core computation with carry
function (cell::RenewalCell)(
        (x, (state,))::Tuple{<:AbstractMatrix, Tuple{<:AbstractMatrix}},
        ps, st::NamedTuple
)
    # x: (1, batch) — R(t) at this timestep
    # state: (gi_length, batch) — recent infections, most-recent-first
    # ps.gi: (gi_length,) — GI distribution vector
    I_t = x .* (ps.gi' * state)                       # (1, batch)
    new_state = cat(I_t, state[1:(end - 1), :]; dims = 1)  # immutable shift
    return (I_t, (new_state,)), st
end
