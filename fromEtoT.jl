include("./setup.jl")
include("./functions.jl")

using .Const, .Func, LinearAlgebra, Serialization

f = open("solution.txt", "w")
for iβ in 1:500
    β = iβ * 0.01
    write(f, string(β))
    write(f, "\t")
    write(f, string(-3.0 * Const.J / 8.0 * sinh(Const.J * β / 2.0) / 
                    (exp(Const.J * β / 2.0) + cosh(Const.J * β / 2.0))))
    write(f, "\n")
end
close(f)

