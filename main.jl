include("./setup.jl")
include("./ml_core.jl")
include("./functions.jl")
using .Const, .MLcore, .Func, LinearAlgebra, Serialization, InteractiveUtils

function main()

    dirname = "./data"

    f = open("error.txt", "w")
    for iϵ in 1:Const.iϵmax
    
        ϵ = (0.9 - 0.5 * (iϵ - 1) / Const.iϵmax) * Const.t * Const.dimB

        filename = dirname * "/param_at_" * lpad(iϵ, 3, "0") * ".dat"
        filenameinit = dirname * "/param_at_000.dat"

        # Initialize weight, bias
        wmoment = zeros(Complex{Float32}, Const.dimB, Const.dimS)
        bmomentB = zeros(Complex{Float32}, Const.dimB)
        bmomentS = zeros(Complex{Float32}, Const.dimS)
        error   = 0.0
        energyS = 0.0
        energyB = 0.0
        numberB = 0.5 * Const.dimB
        μm = 0.0
        lr = Const.lr

        # Define network
        params  = open(deserialize, filenameinit)
        network = MLcore.Network(params...)

        # Learning
        for it in 1:Const.it_num

            numberBprev = numberB

            error, energyS, energyB, numberB, numverB, 
            dweight, dbiasB, dbiasS = 
            MLcore.diff_error(network, ϵ)

            # Optimize
            wmoment         = 0.9 * wmoment - lr * dweight
            network.weight += wmoment
            bmomentB        = 0.9 * bmomentB - lr * dbiasB
            network.biasB  += bmomentB
            bmomentS        = 0.9 * bmomentS - lr * dbiasS
            network.biasS  += bmomentS
            dμ = (numberB - numberBprev) / numverB
            μm = 0.9 * μm + dμ
            network.μ += μm

#            write(f, string(it))
#            write(f, "\t")
#            write(f, string(error))
#            write(f, "\t")
#            write(f, string(energyS / Const.dimS))
#            write(f, "\t")
#            write(f, string(energyB / Const.dimB))
#            write(f, "\t")
#            write(f, string(numberB / Const.dimB))
#            write(f, "\t")
#            write(f, string(network.μ))
#            write(f, "\n")
        end
   
        # Write error
        write(f, string(iϵ))
        write(f, "\t")
        write(f, string(error))
        write(f, "\t")
        write(f, string(energyB / Const.dimB))
        write(f, "\t")
        write(f, string(energyS / Const.dimS))
        write(f, "\t")
        write(f, string(numberB / Const.dimB))
        write(f, "\t")
        write(f, string(network.μ))
        write(f, "\n")

        params = (network.weight, network.biasB, network.biasS, network.μ)
        open(io -> serialize(io, params), filename, "w")
    end
    close(f)
end

@time main()

