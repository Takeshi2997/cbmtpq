include("./setup.jl")
include("./initialize.jl")
include("./functions.jl")
include("./ml_core.jl")
using .Const, .Init, .Func, LinearAlgebra, Serialization,
Combinatorics, InteractiveUtils, .MLcore

weight     = Init.w()
biasB      = zeros(Complex{Float64}, Const.dimB)
biasS      = zeros(Complex{Float64}, Const.dimS)
 
network = (weight, biasB, biasS)

@code_warntype MLcore.diff_error(network, 0.0)
