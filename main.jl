include("./setup.jl")
include("./ml_core.jl")
include("./initialize.jl")
include("./functions.jl")
using .Const, .MLcore, .Init, .Func, LinearAlgebra, Serialization

# Make data file
dirname = "./data"
rm(dirname, force=true, recursive=true)
mkdir(dirname)

weight = Init.weight(Const.dimB, Const.dimS)
biasB = zeros(Complex{Float32}, Const.dimB)
biasS = zeros(Complex{Float32}, Const.dimS)
network = (weight, biasB, biasS) 
filename = dirname * "/param_at_" * lpad(0, 3, "0") * ".dat"
open(io -> serialize(io, network), filename, "w")

f = open("error.txt", "w")
for iϵ in 1:1

    ϵ = 0.4 * (iϵ / Const.iϵmax) * Const.ω * Const.dimB

    filenameprev = dirname * "/param_at_" * lpad(iϵ-1, 3, "0") * ".dat"
    filename = dirname * "/param_at_" * lpad(iϵ, 3, "0") * ".dat"

    # Initialize weight, bias and η
    wmoment = zeros(Complex{Float32}, Const.dimB, Const.dimS)
    wvelocity = zeros(Complex{Float32}, Const.dimB, Const.dimS)
    bmomentB = zeros(Complex{Float32}, Const.dimB)
    bvelocityB = zeros(Complex{Float32}, Const.dimB)
    bmomentS = zeros(Complex{Float32}, Const.dimS)
    bvelocityS = zeros(Complex{Float32}, Const.dimS)
    error = 0.0
    energyS = 0.0
    energyB = 0.0
    energy = 0.0
    lr = Const.lr

    # Define network
    network = open(deserialize, filenameprev)
    
    # Learning
    for it in 1:Const.it_num

        (weight, biasB, biasS) = network

        error, energy, energyS, energyB, dweight, dbiasB, 
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
        write(f, string(real(error)))
        write(f, "\t")
        write(f, string(real(energyS)))
        write(f, "\t")
        write(f, string(real(energyB)))
        write(f, "\t")
        write(f, string(real(energy)))
        write(f, "\n")

        network = (weight, biasB, biasS)
    end

    # Write error
#    write(f, string(iϵ))
#    write(f, "\t")
#    write(f, string(real(error)))
#    write(f, "\n")
    
    open(io -> serialize(io, network), filename, "w")
end
close(f)
