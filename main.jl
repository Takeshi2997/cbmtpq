include("./setup.jl")
include("./ml_core.jl")
include("./initialize.jl")
include("./functions.jl")
using .Const, .MLcore, .Init, .Func, LinearAlgebra, Serialization, InteractiveUtils

function main()

    dirname = "./data"

    f = open("error.txt", "w")
    for iϵ in 2:2 #Const.iϵmax
    
        ϵ = 0.2 * iϵ / Const.iϵmax * Const.ω * Const.dimB

        filename = dirname * "/param_at_" * lpad(iϵ, 3, "0") * ".dat"
        filenameinit = dirname * "/param_at_000.dat"

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
        network = open(deserialize, filenameinit)

        # Learning
        for it in 1:Const.it_num
    
            (weight, biasB, biasS) = network
    
            error, energy, energyS, energyB, dweight, dbiasB, 
            dbiasS = MLcore.diff_error(network, ϵ)

            # Adam
            lr_t = lr * sqrt(1.0 - 0.999^(it + (iϵ - 1) * Const.it_num)) / 
            (1.0 - 0.9^(it + (iϵ - 1) * Const.it_num))
            wmoment    += (1.0 - 0.9) * (dweight - wmoment)
            wvelocity  += (1.0 - 0.999) * (dweight.^2 - wvelocity)
            weight     -= lr_t * wmoment ./ (sqrt.(wvelocity) .+ 1.0 * 10^(-7))
            bmomentB   += (1.0 - 0.9) * (dbiasB - bmomentB)
            bvelocityB += (1.0 - 0.999) * (dbiasB.^2 - bvelocityB)
            biasB      -= lr_t * bmomentB ./ (sqrt.(bvelocityB) .+ 1.0 * 10^(-7))
            bmomentS   += (1.0 - 0.9) * (dbiasS - bmomentS)
            bvelocityS += (1.0 - 0.999) * (dbiasS.^2 - bvelocityS)
            biasS      -= lr_t * bmomentS ./ (sqrt.(bvelocityS) .+ 1.0 * 10^(-7))

            write(f, string(it))
            write(f, "\t")
            write(f, string(real(error)))
            write(f, "\t")
            write(f, string(real(energyS / Const.dimS)))
            write(f, "\t")
            write(f, string(real(energyB / Const.dimB)))
            write(f, "\t")
            write(f, string(real(energy / (Const.dimS + Const.dimB))))
            write(f, "\n")

            network = (weight, biasB, biasS)
        end
   
        # Write error
#        write(f, string(iϵ))
#        write(f, "\t")
#        write(f, string(real(error)))
#        write(f, "\t")
#        write(f, string(real(energyB / Const.dimB)))
#        write(f, "\t")
#        write(f, string(real(energyS / Const.dimS)))
#        write(f, "\n")
       
        open(io -> serialize(io, network), filename, "w")
    end
    close(f)
end

Init.network()
@time main()


