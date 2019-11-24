include("./setup.jl")
include("./ml_core.jl")
include("./functions.jl")
using .Const, .MLcore, .Func, LinearAlgebra, Serialization, InteractiveUtils

mutable struct Network

    weight::Array{Complex{Float64}, 2}
    biasB::Array{Complex{Float64}, 1}
    biasS::Array{Complex{Float64}, 1}
end

function mainrepeat()

    dirname = "./data"

    f = open("error.txt", "w")
    for iϵ in 1:Const.iϵmax
    
        ϵ = 0.2 * Float64(Const.iϵmax - iϵ + 1) / Const.iϵmax * Const.ω * Const.dimB

        filename = dirname * "/param_at_" * lpad(iϵ, 3, "0") * ".dat"

        # Initialize weight, bias
        wmoment    = zeros(Complex{Float64}, Const.dimB, Const.dimS)
        wvelocity  = zeros(Complex{Float64}, Const.dimB, Const.dimS)
        bmomentB   = zeros(Complex{Float64}, Const.dimB)
        bvelocityB = zeros(Complex{Float64}, Const.dimB)
        bmomentS   = zeros(Complex{Float64}, Const.dimS)
        bvelocityS = zeros(Complex{Float64}, Const.dimS)
        error   = 0.0
        energyS = 0.0
        energyB = 0.0
        energy  = 0.0
        lr = Const.lr
    
        # Define network
        params = open(deserialize, filename)
        network = Network(params...)

        # Learning
        for it in 1:Const.it_num
    
            error, energy, energyS, energyB, dweight, dbiasB, dbiasS = 
            MLcore.diff_error(network.weight, network.biasB, network.biasS, ϵ)

            # Adam
            lr_t = lr * sqrt(1.0 - 0.999^it) / (1.0 - 0.9^it)
            wmoment        += (1.0 - 0.9) * (dweight - wmoment)
            wvelocity      += (1.0 - 0.999) * (dweight.^2 - wvelocity)
            network.weight -= lr_t * wmoment ./ (sqrt.(wvelocity) .+ 1.0 * 10^(-7))
            bmomentB       += (1.0 - 0.9) * (dbiasB - bmomentB)
            bvelocityB     += (1.0 - 0.999) * (dbiasB.^2 - bvelocityB)
            network.biasB  -= lr_t * bmomentB ./ (sqrt.(bvelocityB) .+ 1.0 * 10^(-7))
            bmomentS       += (1.0 - 0.9) * (dbiasS - bmomentS)
            bvelocityS     += (1.0 - 0.999) * (dbiasS.^2 - bvelocityS)
            network.biasS  -= lr_t * bmomentS ./ (sqrt.(bvelocityS) .+ 1.0 * 10^(-7))

        end
   
        # Write error
        write(f, string(iϵ))
        write(f, "\t")
        write(f, string(real(error)))
        write(f, "\t")
        write(f, string(real(energyB / Const.dimB)))
        write(f, "\t")
        write(f, string(real(energyS / Const.dimS)))
        write(f, "\n")

        params = (network.weight, network.biasB, network.biasS)
        open(io -> serialize(io, params), filename, "w")
    end
    close(f)
end

for n in 1:20
    @time mainrepeat()
end
