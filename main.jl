include("./setup.jl")
include("./ml_core.jl")
using .Const, .MLcore, InteractiveUtils
using Flux

function learning(io, ϵ)

    for it in 1:Const.it_num

        # Calculate expected value
        error, energy, energyS, energyB, numberB = MLcore.sampling(ϵ)

        write(io, string(it))
        write(io, "\t")
        write(io, string(error))
        write(io, "\t")
        write(io, string(energyS / Const.dimS))
        write(io, "\t")
        write(io, string(energyB / Const.dimB))
        write(io, "\t")
        write(io, string(numberB / Const.dimB))
        write(io, "\n")
    end
end

function main()

    dirname = "./data"
    rm(dirname, force=true, recursive=true)
    mkdir(dirname)

    io = open("error.txt", "w")
    for iϵ in 1:1 #Const.iϵmax
    
        ϵ = (0.4 + 0.5 * (iϵ - 1) / Const.iϵmax) * Const.t * Const.dimB

        filenamereal = dirname * "/realparam_at_" * lpad(iϵ, 3, "0") * ".bson"
        filenameimag = dirname * "/imagparam_at_" * lpad(iϵ, 3, "0") * ".bson"

        # Initialize
        error   = 0.0
        energy  = 0.0
        energyS = 0.0
        energyB = 0.0
        numberB = 0.0

        # Learning
        @time learning(io, ϵ) 

        # Write error
#        write(io, string(iϵ))
#        write(io, "\t")
#        write(io, string(error))
#        write(io, "\t")
#        write(io, string(energyB / Const.dimB))
#        write(io, "\t")
#        write(io, string(energyS / Const.dimS))
#        write(io, "\t")
#        write(io, string(numberB / Const.dimB))
#        write(io, "\n")

        MLcore.Func.ANN.save(filenamereal, filenameimag)
        MLcore.Func.ANN.load(filenamereal, filenameimag)
    end
    close(io)
end

main()

