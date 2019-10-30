include("./setup.jl")
include("./ml_core.jl")
include("./initialize.jl")
include("./functions.jl")
using .Const, .MLcore, .Init, .Func, LinearAlgebra, Serialization

# Make data file
dirname = "./data"
rm(dirname, force=true, recursive=true)
mkdir(dirname)

f = open("error.txt", "w")
for iϵ in 1:1

    ϵ = 0.04

    filename = dirname * "/param_at_" * lpad(iϵ, 3, "0") * ".dat"

    # Initialize weight, bias and η
    weight = Init.weight(Const.dimB, Const.dimS)
    wmoment = zeros(Float32, Const.dimB, Const.dimS)
    wvelocity = zeros(Float32, Const.dimB, Const.dimS)
    biasB = Init.bias(Const.dimB)
    bmomentB = zeros(Float32, Const.dimB)
    bvelocityB = zeros(Float32, Const.dimB)
    biasS = Init.bias(Const.dimS)
    bmomentS = zeros(Float32, Const.dimS)
    bvelocityS = zeros(Float32, Const.dimS)
    error = 0.0
    energyS = 0.0
    energyB = 0.0
    lr = Const.lr

    # Define network
    network = (weight, biasB, biasS)
    
    # Learning
    for it in 1:Const.it_num
        error, energy, energyS, energyB, dispersion, dweight, dbiasB, 
        dbiasS = MLcore.diff_error(network, ϵ)

        # Adam
        lr_t = lr * sqrt(1.0 - 0.999^it) / (1.0 - 0.9^it)
        wmoment += (1.0 - 0.9) * (dweight - wmoment)
        wvelocity += (1.0 - 0.999) * (dweight.^2 - wvelocity)
        weight -= lr_t * wmoment ./ (sqrt.(wvelocity) .+ 1.0 * 10^(-7))
        bmomentB += (1.0 - 0.9) * (dbiasB - bmomentB)
        bvelocityB += (1.0 - 0.999) * (dbiasB.^2 - bvelocityB)
        biasB -= lr_t * bmomentB ./ (sqrt.(bvelocityB) .+ 1.0 * 10^(-7))
        bmomentS += (1.0 - 0.9) * (dbiasS - bmomentS)
        bvelocityS += (1.0 - 0.999) * (dbiasS.^2 - bvelocityS)
        biasS -= lr_t * bmomentS ./ (sqrt.(bvelocityS) .+ 1.0 * 10^(-7))

        write(f, string(it))
        write(f, "\t")
        write(f, string(error))
        write(f, "\t")
        write(f, string(dispersion))
        write(f, "\t")
        write(f, string(energyS))
        write(f, "\t")
        write(f, string(energyB))
        write(f, "\t")
        write(f, string(energyB + energyS))
        write(f, "\n")

        network = (weight, biasB, biasS)
    end

    # Write error
#    write(f, string(iϵ))
#    write(f, "\t")
#    write(f, string(error))
#    write(f, "\t")
#    write(f, string(energyS))
#    write(f, "\t")
#    write(f, string(energyB))
#    write(f, "\n")
    
    open(io -> serialize(io, network), filename, "w")
end
close(f)
