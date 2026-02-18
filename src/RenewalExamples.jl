module RenewalExamples

import CensoredDistributions as CD
using Distributions
using Random: AbstractRNG
using Lux
import LuxCore

export UnknownGI, FixedGI
export RenewalCell
export delay_conv

# Fixed/discretised and unknown generation interval distributions
include("gis.jl")

# Recurrent renewal equation step (Lux cell)
include("steps.jl")

# Delay convolution layer
include("conv.jl")

end
