include("./setup.jl")
include("./functions.jl")

using .Const, .Func, LinearAlgebra, Serialization

f = open("energy-temperature.txt", "w")
for iϵ in 1:490
    ϵ = iϵ * Const.ω * Const.dimB * 0.001
    write(f, string(ϵ))
    write(f, "\t")
    write(f, string(Func.retranslate(ϵ)))
    write(f, "\t")
    write(f, string(-Const.dimS / 2.0 * 
                    Const.J * tanh(Const.J / Func.retranslate(ϵ))))
    write(f, "\n")
end
close(f)

