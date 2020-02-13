include("./setup.jl")
include("./ml_core.jl")
include("./params.jl")
include("./ann.jl")
using LinearAlgebra, Serialization, InteractiveUtils

function main()

    dirname = "./data"

    f = open("error.txt", "w")
    for iϵ in 1:1 #Const.iϵmax
    
        ϵ = (0.0 - 0.5 * (iϵ - 1) / iϵmax) * t * dimB

        filename = dirname * "/param_at_" * lpad(iϵ, 3, "0") * ".dat"
        filenameinit = dirname * "/param_at_000.dat"

        # Initialize weight, bias
        error   = 0.0
        energy  = 0.0
        energyS = 0.0
        energyB = 0.0

        # Define network
        params  = open(deserialize, filenameinit)
        network = Network(params...)
        moment  = Moment(zeros(ComplexF64, dimB, dimS), 
                         zeros(ComplexF64, dimB))

        # Learning
        for it in 1:it_num

            #Calculate expected value
            error, energy, energyS, energyB, numberB =
            sampling(network, ϵ)

            #Update Parameter
            updateparams(network, moment, energy, ϵ, lr)

            write(f, string(it))
            write(f, "\t")
            write(f, string(error))
            write(f, "\t")
            write(f, string(energyS / dimS))
            write(f, "\t")
            write(f, string(energyB / dimB))
            write(f, "\t")
            write(f, string(numberB / dimB))
            write(f, "\n")
        end
   
        # Write error
#        write(f, string(iϵ))
#        write(f, "\t")
#        write(f, string(error))
#        write(f, "\t")
#        write(f, string(energyB / dimB))
#        write(f, "\t")
#        write(f, string(energyS / dimS))
#        write(f, "\t")
#        write(f, string(numberB / dimB))
#        write(f, "\n")

        params = (network.w, network.b)
        open(io -> serialize(io, params), filename, "w")
    end
    close(f)
end

@time main()

