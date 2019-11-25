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
        lr = Const.lr_repeat
    
        # Define network
        params = open(deserialize, filename)
        network = Network(params...)

        # Learning
        for it in 1:Const.it_num
    
            error, energy, energyS, energyB, dweight, dbiasB, dbiasS = 
            MLcore.diff_error(network.weight, network.biasB, network.biasS, ϵ)

            # Momentum
            wmoment = 0.9 * wmoment - lr * dweight
            network.weight += wmoment
            bmomentS = 0.9 * bmomentS - lr * dbiasS
            network.biasS += bmomentS
            bmomentB = 0.9 * bmomentB - lr * dbiasB
            network.biasB += bmomentB

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

for i in 1:5
    @time mainrepeat()
end
