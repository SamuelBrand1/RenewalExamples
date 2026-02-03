module RenewalExamples

import CensoredDistributions as CD
using Distributions
using Random: AbstractRNG

export UnknownGI, FixedGI

# Fixed/discretised and unknown generation interval distributions
include("gis.jl")

end
