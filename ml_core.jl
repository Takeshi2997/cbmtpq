include("./functions.jl")
include("./ann.jl")
include("./params.jl")

function sampling(o::Array{DiffReal, 1}, oer::Array{DiffComplex, 1}, oei::Array{DiffComplex, 1},
                  ϵ::Float64)

    n = rand([1.0, 0.0],  Const.dimB)
    s = rand([1.0, -1.0], Const.dimS)
    energy  = 0.0
    energyS = 0.0
    energyB = 0.0
    numberB = 0.0

    for i in 1:Const.burnintime

        s = updateS(s, n)
        n = updateB(n, s)
    end

    for i in 1:Const.iters_num
        s     = updateS(s, n)
        nnext = updateB(n, s)

        eS = energyS_shift(s, n)
        eB = energyB_shift(n, s)
        e  = eS + eB
        energy    += e
        energyS   += eS
        energyB   += eB
        numberB   += sum(n)
        backward(o, oer, oei, n, s, e)
        n = nnext
    end
    energy   = real(energy)  / Const.iters_num
    energyS  = real(energyS) / Const.iters_num
    energyB  = real(energyB) / Const.iters_num
    numberB /= Const.iters_num
    error    = (energy - ϵ)^2

    return error, energy, energyS, energyB, numberB
end


