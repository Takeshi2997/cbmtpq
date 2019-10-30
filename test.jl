include("./setup.jl")
include("./ml_core.jl")
include("./initialize.jl")
include("./functions.jl")
using .Const, .MLcore, .Init, .Func, LinearAlgebra, Serialization

function f(s1, s2)
    σx = [1.0 1.0]
    return prod(σx .* (s1 .!= s2))
end

A = zeros(Float32, 4, 4)
s1array = [[1.0 1.0], [-1.0 1.0], [1.0 -1.0], [-1.0 -1.0]]
s2array = copy(s1array)
println([1.0 1.0] .* (s1array[2] .!= s2array[1]))
for ix in 1:4
    for iy in 1:4
        global A[ix, iy] = f(s1array[ix], s2array[iy])
    end
end

display(A)

println()

