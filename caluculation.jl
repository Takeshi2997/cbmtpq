include("./setup.jl")
include("./ml_core.jl")
include("./initialize.jl")
include("./functions.jl")
using .Const, .MLcore, .Init, .Func, LinearAlgebra, Serialization

function calculate()
    dirname = "./data"
    
    f = open("energy_data.txt", "w")
    for iϵ in 1:Const.iϵmax
    
        filename = dirname * "/param_at_" * lpad(iϵ, 3, "0") * ".dat"
        network = open(deserialize, filename)
    
        energy, energyS, energyB = MLcore.forward(network)
        β = Func.retranslate(real(energyB) / Const.dimB)
    
        # Write energy
        write(f, string(β))
        write(f, "\t")
        write(f, string(real(energyS) / Const.dimS))
        write(f, "\t")
        write(f, string(real(energyB) / Const.dimB))
        write(f, "\t")
        write(f, string(-3.0 * Const.J / 4.0 * sinh(Const.J * β / 2.0) / 
                        (exp(Const.J * β / 2.0) + cosh(Const.J * β / 2.0))))
        write(f, "\n")
    end
end

@time calculate()

