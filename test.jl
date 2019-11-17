include("./setup.jl")
include("./functions.jl")
include("./ml_core.jl")
using .Const, .Func, LinearAlgebra, Serialization,
Combinatorics, InteractiveUtils, .MLcore

A = -[1 0 0 0
     0 -1 2 0
     0 2 -1 0
     0 0 0 1] ./ 4.0
B = -[0 0 0 0
     0 -2 2 0
     0 2 -2 0
     0 0 0 0] ./ 4.0
display(A)
println()
println(eigvals(A))
display(B)
println()
println(eigvals(B))
