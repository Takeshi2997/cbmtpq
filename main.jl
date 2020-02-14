include("./setup.jl")
include("./params.jl")
include("./ml_core.jl")
include("./ann.jl")
using .Const, InteractiveUtils

function main()

    dirname = "./data"

    f = open("error.txt", "w")
    for iϵ in 1:1 #Const.iϵmax
    
        ϵ = (0.0 - 0.5 * (iϵ - 1) / Const.iϵmax) * Const.t * Const.dimB

        filenamereal = dirname * "/realparam_at_" * lpad(iϵ, 3, "0") * ".bson"
        filenameimag = dirname * "/imagparam_at_" * lpad(iϵ, 3, "0") * ".bson"
        filenamerealinit = dirname * "/realparam_at_" * lpad(iϵ-1, 3, "0") * ".bson"
        filenameimaginit = dirname * "/imagparam_at_" * lpad(iϵ-1, 3, "0") * ".bson"

        load(filenamerealinit, filenameimaginit)

        # Initialize
        error   = 0.0
        energy  = 0.0
        energyS = 0.0
        energyB = 0.0

        # Learning
        for it in 1:Const.it_num

            # Initialize gradient params
            o, oer, oei = initO()

            # Calculate expected value
            error, energy, energyS, energyB, numberB = sampling(o, oer, oei, ϵ)

            # Update Parameter
            update(o, oer, oei, energy, ϵ)

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

        save(filenamereal, filenameimag)
    end
    close(f)
end

function initO()

    o   = Array{DiffReal, 1}(undef, Const.layers_num)
    oer = Array{DiffComplex, 1}(undef, Const.layers_num)
    oei = Array{DiffComplex, 1}(undef, Const.layers_num)

    for i in 1:Const.layers_num
        o[i]   = DiffReal(zeros(Float64,       Const.layer[i+1], Const.layer[i]), 
                          zeros(Float64,       Const.layer[i+1]))
        oer[i] = DiffComplex(zeros(ComplexF64, Const.layer[i+1], Const.layer[i]), 
                             zeros(ComplexF64, Const.layer[i+1]))
        oei[i] = DiffComplex(zeros(ComplexF64, Const.layer[i+1], Const.layer[i]), 
                             zeros(ComplexF64, Const.layer[i+1]))
    end

    return o, oer, oei
end

@time main()

