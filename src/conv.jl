using NNlib: conv

"""
    delay_conv(delay_length) -> Lux.Chain

Build a causal delay convolution pipeline from Lux built-in layers.
Expects input from `Recurrence(cell; return_sequence=true)` — a `Vector`
of `(1, batch)` matrices — and returns a 3D array `(T, 1, batch)` of
expected observations.

The delay PMF lives in the `Conv` layer's weight parameter
(shape `(delay_length, 1, 1)`).

    E_obs[t] = sum_{s=1}^{D} delay[s] * I[t - s]

# Example

```julia
using Lux, Random, RenewalExamples
rng = Random.default_rng()
gi_length = 10; delay_length = 7
model = Lux.Chain(
    Lux.Recurrence(RenewalCell(gi_length); return_sequence=true),
    delay_conv(delay_length),
)
ps, st = Lux.setup(rng, model)
```
"""
function delay_conv(delay_length::Int)
    return Lux.Chain(
        Lux.WrappedFunction(x -> reduce(vcat, x)),
        Lux.WrappedFunction(x -> reshape(x, size(x, 1), 1, size(x, 2))),
        Lux.Conv(
            (delay_length,), 1 => 1;
            use_bias = false,
            pad = (delay_length - 1, 0)
        ),
    )
end
