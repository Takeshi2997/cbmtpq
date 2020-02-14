include("./setup.jl")
using .Const, LinearAlgebra, Serialization,
Combinatorics, InteractiveUtils

mutable struct HogeType

    W::Array{Float64, 2}
    b::Array{Float64, 1}
end

var = Array{HogeType, 1}(undef, 2)
var[1] = HogeType(zeros(Float64, 2,2), zeros(Float64, 2))
var[2] = HogeType(ones(Float64, 2,2), ones(Float64, 2))

println(typeof(var))
println(var[2].b)



