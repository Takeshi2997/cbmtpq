include("./setup.jl")
using .Const, LinearAlgebra, Serialization,
Combinatorics, InteractiveUtils

B = -[1 0 0 0
     0 -1 2 0
     0 2 -1 0
     0 0 0 1] ./ 4.0
 
v = [1 2 3 4 5 6 7]
for iy in 1:100
    println(iy, "\t", 10 + iy%10 + 1)
end
