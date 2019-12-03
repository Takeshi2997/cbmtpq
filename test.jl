include("./setup.jl")
include("./functions.jl")
include("./ml_core.jl")
using .Const, .Func, LinearAlgebra, Serialization,
Combinatorics, InteractiveUtils, .MLcore

B = [0 0 0 0
     0 0 2 0
     0 2 0 0
     0 0 0 0] 

function control(nx, ny)

    out = 1.0
    if nx[1] == nx[2] || nx[1] .== ny[1] || nx[2] .== ny[2]
        out *= 0.0
    end
    return out
end


v = [1 2 3 4 5 6 7]

println(eigvals(B))
println(typeof(4/2))
