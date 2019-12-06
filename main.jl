include("./setup.jl")
include("./ml_core.jl")
include("./initialize.jl")
include("./functions.jl")
using .Const, .MLcore, .Init, .Func, LinearAlgebra, Serialization, InteractiveUtils

mutable struct Network

    weight::Array{Complex{Float64}, 2}
    biasB::Array{Complex{Float64}, 1}
    biasS::Array{Complex{Float64}, 1}
end

function main()

    dirname = "./data"

    f = open("error.txt", "w")
    for iϵ in 1:1 #Const.iϵmax
    
        ϵ = -(0.15 - 0.5 * (iϵ - 1) / Const.iϵmax) * Const.t * Const.dimB

        filename = dirname * "/param_at_" * lpad(iϵ, 3, "0") * ".dat"
        filenameinit = dirname * "/param_at_" * lpad(iϵ-1, 3, "0") * ".dat"
#        filenameinit = dirname * "/param_at_000.dat"

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
        numberB = 0.0
        lr = Const.lr

        # Define network
        params = open(deserialize, filenameinit)
        network = Network(params...)

        # Learning
        for it in 1:Const.it_num
    
            error, energyS, energyB, dweight, dbiasB, dbiasS = 
            MLcore.diff_error(network.weight, network.biasB, network.biasS, ϵ)

            # Optimize
#            lr_t = lr * sqrt(1.0 - 0.999^it) / (1.0 - 0.9^it)
#            wmoment        += (1.0 - 0.9) * (dweight - wmoment)
#            wvelocity      += (1.0 - 0.999) * (dweight.^2 - wvelocity)
#            network.weight -= lr_t * wmoment ./ (sqrt.(wvelocity) .+ 1.0 * 10^(-7))
#            bmomentB       += (1.0 - 0.9) * (dbiasB - bmomentB)
#            bvelocityB     += (1.0 - 0.999) * (dbiasB.^2 - bvelocityB)
#            network.biasB  -= lr_t * bmomentB ./ (sqrt.(bvelocityB) .+ 1.0 * 10^(-7))
#            bmomentS       += (1.0 - 0.9) * (dbiasS - bmomentS)
#            bvelocityS     += (1.0 - 0.999) * (dbiasS.^2 - bvelocityS)
#            network.biasS  -= lr_t * bmomentS ./ (sqrt.(bvelocityS) .+ 1.0 * 10^(-7))

            wmoment         = 0.9 * wmoment - lr * dweight
            network.weight += wmoment
            bmomentS        = 0.9 * bmomentS - lr * dbiasS
            network.biasS  += bmomentS
            bmomentB        = 0.9 * bmomentB - lr * dbiasB
            network.biasB  += bmomentB

            write(f, string(it))
            write(f, "\t")
            write(f, string(real(error)))
            write(f, "\t")
            write(f, string(real(energyS / Const.dimS)))
            write(f, "\t")
            write(f, string(real(energyB / Const.dimB)))
            write(f, "\n")
        end
   
        # Write error
#        write(f, string(iϵ))
#        write(f, "\t")
#        write(f, string(real(error)))
#        write(f, "\t")
#        write(f, string(real(energyB / Const.dim)))
#        write(f, "\t")
#        write(f, string(real(energyS / Const.systemsize)))
#        write(f, "\t")
#        write(f, string(real(numberB / Const.dim)))
#        write(f, "\n")

        params = (network.weight, network.biasB, network.biasS)
        open(io -> serialize(io, params), filename, "w")
    end
    close(f)
end

Init.network()
@time main()

