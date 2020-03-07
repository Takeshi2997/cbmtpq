include("./setup.jl")
using .Const, LinearAlgebra, Serialization,
Combinatorics, InteractiveUtils

B =-[1  0  0  0
     0 -1  2  0
     0  2 -1  0
     0  0  0  1] ./ 4.0

println(eigvals(B))

