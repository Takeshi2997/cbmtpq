include("./functions.jl")
include("./ann.jl")
include("./params.jl")

function sampling(ϵ::Float64)

    n = rand([1.0, 0.0],  Const.dimB)
    s = rand([1.0, -1.0], Const.dimS)
    energy  = 0.0
    energyS = 0.0
    energyB = 0.0
    numberB = 0.0

    o, oer, oei = initO()

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

    for i in 1:Const.layers_num
        ΔWreal = 2.0 * (energy - ϵ) * 2.0 * (real.(oer[i].W) - energy * o[i].W) / Const.iters_num
        Δbreal = 2.0 * (energy - ϵ) * 2.0 * (real.(oer[i].b) - energy * o[i].b) / Const.iters_num
        ΔWimag = 2.0 * (energy - ϵ) * 2.0 * imag.(oei[i].W) / Const.iters_num
        Δbimag = 2.0 * (energy - ϵ) * 2.0 * imag.(oei[i].b) / Const.iters_num
        update(ΔWreal, Δbreal, ΔWimag, Δbimag, i)
    end

    return error, energy, energyS, energyB, numberB
end

function initO()

    o   = Array{DiffReal, 1}(undef, Const.layers_num)
    oer = Array{DiffComplex, 1}(undef, Const.layers_num)
    oei = Array{DiffComplex, 1}(undef, Const.layers_num)

    for i in 1:Const.layers_num
        o[i]   = DiffReal(zeros(Float64, Const.layer[i+1], Const.layer[i]), 
                          zeros(Float64, Const.layer[i+1]))
        oer[i] = DiffComplex(zeros(ComplexF64, Const.layer[i+1], Const.layer[i]), 
                             zeros(ComplexF64, Const.layer[i+1]))
        oei[i] = DiffComplex(zeros(ComplexF64, Const.layer[i+1], Const.layer[i]), 
                             zeros(ComplexF64, Const.layer[i+1]))
    end

    return o, oer, oei
end


