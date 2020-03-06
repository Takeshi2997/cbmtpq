include("./setup.jl")
include("./ml_core.jl")
using .Const, .MLcore, InteractiveUtils
using Flux
using BSON

function learning(io, ϵ::Float64, lr::Float64)

    for it in 1:Const.it_num

        # Calculate expected value
        error, energy, energyS, energyB, numberB = MLcore.sampling(ϵ, lr)

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

    dirnameerror = "./error"
    rm(dirnameerror, force=true, recursive=true)
    mkdir(dirnameerror)

    g = open("error.txt", "w")
    for iϵ in 1:Const.iϵmax
    
        ϵ = (1.0 - 0.6 * (iϵ - 1) / Const.iϵmax) * Const.t * Const.dimB

        filenameparams = dirname * "/params_at_" * lpad(iϵ, 3, "0") * ".bson"

        # Initialize
        error   = 0.0
        energy  = 0.0
        energyS = 0.0
        energyB = 0.0
        numberB = 0.0
        lr = Const.lr

        # Learning
        filename = dirnameerror * "/error" * lpad(iϵ, 3, "0") * ".txt"
        f = open(filename, "w")
        @time learning(f, ϵ, lr) 
        close(f)

        # Write error
        write(g, string(iϵ))
        write(g, "\t")
        write(g, string(error))
        write(g, "\t")
        write(g, string(energyB / Const.dimB))
        write(g, "\t")
        write(g, string(energyS / Const.dimS))
        write(g, "\t")
        write(g, string(numberB / Const.dimB))
        write(g, "\n")

        MLcore.Func.ANN.save(filenameparams)
        network = BSON.load(filenameparams)
        setfield!(MLcore.Func.ANN.network, :f, network[:f])
        setfield!(MLcore.Func.ANN.network, :g, network[:g])
    end
    close(g)
end

main()

