"""
    UnknownGI{DT, P<:NamedTuple, T<:Real}

Represents an unknown generation interval distribution with parameters to be estimated.

# Fields
- `dist_type::Type{DT}`: The type of distribution for the generation interval.
- `param_priors::P`: Named tuple containing prior distributions for the parameters.
NOTE: The order of parameters in `param_priors` must match the order expected by `dist_type`.
- `interval::T`: The interval width for double interval censoring.
- `upper::T`: Upper bound for the generation interval support.
"""
@kwdef struct UnknownGI{DT, P <: NamedTuple, T <: Real}
    dist_type::Type{DT}
    param_priors::P
    interval::T
    upper::T
end

function Base.rand(rng::AbstractRNG, ugi::UnknownGI)
    params = rand.(values(ugi.param_priors))
    return ugi.dist_type(params...)
end

"""
    double_interval_censored(ugi::UnknownGI, params...)

Create a double interval censored distribution from an `UnknownGI` specification and parameter values.

# Arguments
- `ugi::UnknownGI`: An `UnknownGI` object specifying the distribution type and censoring parameters.
- `params...`: Parameter values for the distribution specified in `ugi.dist_type`.
  The order must match the parameter order expected by `ugi.dist_type`.

# Returns
A double interval censored distribution constructed from the given parameters with the
censoring interval and upper bound specified in `ugi`.
"""
function CD.double_interval_censored(ugi::UnknownGI, params...)
    return CD.double_interval_censored(
        ugi.dist_type(params...); interval = ugi.interval, upper = ugi.upper
    )
end
