module MLcore
include("./setup.jl")
include("./functions.jl")
using .Const, .Func
using Flux: Zygote
using BSON: @load

function sampling(ϵ::Float32, lr::Float32)

    n = rand([1.0f0, 0.0f0],  Const.dimB)
    s = rand([1.0f0, -1.0f0], Const.dimS)
    x = vcat(s, n)
    energy  = 0.0f0
    energyS = 0.0f0
    energyB = 0.0f0
    numberB = 0.0f0

    o, oer, oei = initO()

    for i in 1:Const.burnintime

        x = Func.updateS(x)
        x = Func.updateB(x)
    end

    for i in 1:Const.iters_num
        x = Func.updateS(x)
        eS = Func.energyS_shift(x)

        x = Func.updateB(x)
        eB = Func.energyB_shift(x)

        e  = eS + eB
        energy    += e
        energyS   += eS
        energyB   += eB
        numberB   += sum(x[Const.dimS+1:end])
        realgs, imaggs = Func.ANN.setupbackward(x)
        for i in 1:Const.layers_num
            dwr, dbr, dwi, dbi = Func.ANN.backward(realgs, imaggs, i)
            o[i].W   += dwr
            o[i].b   += dbr
            oe[i].W += (dwr .+ im * dwi) * e
            oe[i].b += (dbr .+ im * dbi) * e
        end
    end
    energy   = real(energy)  / Const.iters_num
    energyS  = real(energyS) / Const.iters_num
    energyB  = real(energyB) / Const.iters_num
    numberB /= Const.iters_num
    error    = (energy - ϵ)^2

    for i in 1:Const.layers_num
        ΔWreal = 2.0f0 * (energy - ϵ) * 2.0f0 * (real.(oe[i].W) - energy * o[i].W) / 
        Const.iters_num
        Δbreal = 2.0f0 * (energy - ϵ) * 2.0f0 * (real.(oe[i].b) - energy * o[i].b) / 
        Const.iters_num
        Func.ANN.update(ΔWreal, Δbreal, i, lr)
    end

    return error, energy, energyS, energyB, numberB
end

mutable struct DiffReal

    W::Array{Float32, 2}
    b::Array{Float32, 1}
end

mutable struct DiffComplex

    W::Array{Complex{Float32}, 2}
    b::Array{Complex{Float32}, 1}
end

function initO()

    o  = Array{DiffReal, 1}(undef, Const.layers_num)
    oe = Array{DiffComplex, 1}(undef, Const.layers_num)

    for i in 1:Const.layers_num
        o[i]   = DiffReal(zeros(Float32, Const.layer[i+1], Const.layer[i]), 
                          zeros(Float32, Const.layer[i+1]))
        oe[i] = DiffComplex(zeros(Complex{Float32}, Const.layer[i+1], Const.layer[i]), 
                            zeros(Complex{Float32}, Const.layer[i+1]))
    end

    return o, oe
end

function calculation_energy()

    n = rand([1.0f0, 0.0f0],  Const.dimB)
    s = rand([1.0f0, -1.0f0], Const.dimS)
    x = vcat(s, n)
    energy  = 0.0f0
    energyS = 0.0f0
    energyB = 0.0f0
    numberB = 0.0f0

    for i in 1:Const.burnintime

        x = Func.updateS(x)
        x = Func.updateB(x)
    end

    for i in 1:Const.iters_num
        x = Func.updateS(x)
        eS = Func.energyS_shift(x)

        x = Func.updateB(x)
        eB = Func.energyB_shift(x)

        e  = eS + eB
        energy    += e
        energyS   += eS
        energyB   += eB
        numberB   += sum(x[Const.dimS+1:end])
    end
    energy   = real(energy)  / Const.iters_num
    energyS  = real(energyS) / Const.iters_num
    energyB  = real(energyB) / Const.iters_num
    numberB /= Const.iters_num

    return energyS, energyB, numberB
end

end
