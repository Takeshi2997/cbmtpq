include("./setup.jl")
include("./functions.jl")

using .Const, .Func, LinearAlgebra, Serialization

f = open("energy-temperature.txt", "w")
for iβ in 1:490
    β = iβ * 0.01
    ϵ = Func.translate(β)
    write(f, string(β))
    write(f, "\t")
    write(f, string(ϵ))
    write(f, "\t")
    write(f, string(-3.0 * Const.J / 4.0 * sinh(Const.J * β / 2.0) / 
                    (exp(Const.J * β / 2.0) + cosh(Const.J * β / 2.0))))
    write(f, "\n")
end
close(f)

