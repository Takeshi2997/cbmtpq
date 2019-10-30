include("./setup.jl")
include("./ml_core.jl")
include("./initialize.jl")
include("./functions.jl")
using .Const, .MLcore, .Init, .Func, LinearAlgebra, Serialization

dirname = "./data"

f = open("energy_data.txt", "w")
for iϵ in 1:990

    filename = dirname * "/param_at_" * lpad(iϵ, 3, "0") * ".dat"
    network = open(deserialize, filename)

    energyS, energyB = MLcore.forward(network)
    temperature = Func.retranslate(0.0005 * iϵ)

    # Write energy
    write(f, string(0.0005 * iϵ))
    write(f, "\t")
    write(f, string(energyS))
    write(f, "\t")
    write(f, string(energyB))
    write(f, "\t")
    write(f, string(-Const.J * tanh(Const.J/temperature)))
    write(f, "\n")
end

