include("./setup.jl")
using .Const, LinearAlgebra, Serialization,
Combinatorics, InteractiveUtils

mutable struct HogeType

    W::Array{Float32, 2}
    b::Array{Float32, 1}
end

var = Array{HogeType, 1}(undef, 2)
var[1] = HogeType(zeros(Float32, 2,2), zeros(Float32, 2))
var[2] = HogeType(ones(Float32, 2,2), ones(Float32, 2))

println(typeof(var))
println(var[2].b)



