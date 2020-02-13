include("./setup.jl")
include("./ann.jl")
using LinearAlgebra, Serialization,
Combinatorics, InteractiveUtils

B = -[1 0 0 0
     0 -1 2 0
     0 2 -1 0
     0 0 0 1] * im ./ 4.0

v = [1.0, 1.0, 1.0, 1.0] * im

