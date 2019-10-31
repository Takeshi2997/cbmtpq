include("./setup.jl")
include("./ml_core.jl")
include("./initialize.jl")
include("./functions.jl")
using .Const, .MLcore, .Init, .Func, LinearAlgebra, Serialization

function fx(s1, s2)
    e = [1.0 1.0]
    return prod(e .* (s1 .!= s2))
end

function fy(s1, s2)
    
    return prod(1.0im * s2 .* (s1 .!= s2))
end

function fz(s1, s2)

    return prod(s1 .* (s1 .== s2))
end

A = zeros(Float32, 4, 4)
s1array = [[1.0 1.0], [-1.0 1.0], [1.0 -1.0], [-1.0 -1.0]]
s2array = copy(s1array)
println([1.0 1.0] .* (s1array[2] .!= s2array[1]))
for ix in 1:4
    for iy in 1:4
        global A[ix, iy] = Func.hamiltonianS(s1array[ix], s2array[iy])
    end
end

display(A)

println()

