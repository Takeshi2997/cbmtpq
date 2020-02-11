include("./setup.jl")
include("./ml_core.jl")
include("./params.jl")
include("./ann.jl")
using .Const, .MLcore, .Params, .ANN, LinearAlgebra, Serialization, InteractiveUtils

function main()

    dirname = "./data"

    f = open("error.txt", "w")
    for iϵ in 1:1 #Const.iϵmax
    
        ϵ = (0.0 - 0.5 * (iϵ - 1) / Const.iϵmax) * Const.t * Const.dimB

        filename = dirname * "/param_at_" * lpad(iϵ, 3, "0") * ".dat"
        filenameinit = dirname * "/param_at_000.dat"

        # Initialize weight, bias
<<<<<<< HEAD
>>>>>>> origin/ann
        error   = 0.0
        energy  = 0.0
        energyS = 0.0
        energyB = 0.0
        lr      = Const.lr

        # Define network
        params  = open(deserialize, filenameinit)
        network = Params.Network(params...)
        moment  = Params.Moment(zeros(ComplexF64, Const.dimB, Const.dimS), 
                                zeros(ComplexF64, Const.dimB))

        # Learning
        for it in 1:Const.it_num

<<<<<<< HEAD
            #Calculate expected value
            error, energy, energyS, energyB, numberB =
            MLcore.sampling(network, ϵ)

            #Update Parameter
            MLcore.updateparam(network, moment, energy, ϵ, lr)
>>>>>>> origin/ann

            write(f, string(it))
            write(f, "\t")
            write(f, string(error))
            write(f, "\t")
            write(f, string(energyS / Const.dimS))
            write(f, "\t")
            write(f, string(energyB / Const.dimB))
            write(f, "\t")
            write(f, string(numberB / Const.dimB))
            write(f, "\n")
        end
   
        # Write error
#        write(f, string(iϵ))
#        write(f, "\t")
#        write(f, string(error))
#        write(f, "\t")
#        write(f, string(energyB / Const.dimB))
#        write(f, "\t")
#        write(f, string(energyS / Const.dimS))
#        write(f, "\t")
#        write(f, string(numberB / Const.dimB))
#        write(f, "\n")

<<<<<<< HEAD
        params = (network.w, network.b)
>>>>>>> origin/ann
        open(io -> serialize(io, params), filename, "w")
    end
    close(f)
end

@time main()

