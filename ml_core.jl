module MLcore
include("./setup.jl")
include("./functions.jl")
using .Const, .Func, CuArrays
using Flux: Zygote

function sampling() end

CuArrays.@cufunc function sampling(ϵ::Float32)

    n = rand([1.0f0, 0.0f0],  Const.dimB)
    s = rand([1.0f0, -1.0f0], Const.dimS)
    energy  = 0.0f0
    energyS = 0.0f0
    energyB = 0.0f0
    numberB = 0.0f0

    o, oer, oei = initO()

    for i in 1:Const.burnintime

        s = Func.updateS(s, n)
        n = Func.updateB(n, s)
    end

    for i in 1:Const.iters_num
        s     = Func.updateS(s, n)
        nnext = Func.updateB(n, s)

        eS = Func.energyS_shift(s, n)
        eB = Func.energyB_shift(n, s)
        e  = eS + eB
        energy    += e
        energyS   += eS
        energyB   += eB
        numberB   += sum(n)
        realgs, imaggs = Func.ANN.setupbackward(n, s)
        for i in 1:Const.layers_num
            dwr, dbr, dwi, dbi = Func.ANN.backward(realgs, imaggs, i)
            o[i].W   += dwr
            o[i].b   += dbr
            oer[i].W += dwr * e
            oer[i].b += dbr * e
            oei[i].W += dwi * e
            oei[i].b += dbi * e
        end

        n = nnext
    end
    energy   = real(energy)  / Const.iters_num
    energyS  = real(energyS) / Const.iters_num
    energyB  = real(energyB) / Const.iters_num
    numberB /= Const.iters_num
    error    = (energy - ϵ)^2

    for i in 1:Const.layers_num
        ΔWreal = 2.0f0 * (energy - ϵ) * 2.0f0 * (real.(oer[i].W) - energy * o[i].W) / Const.iters_num
        Δbreal = 2.0f0 * (energy - ϵ) * 2.0f0 * (real.(oer[i].b) - energy * o[i].b) / Const.iters_num
        ΔWimag = 2.0f0 * (energy - ϵ) * 2.0f0 * imag.(oei[i].W) / Const.iters_num
        Δbimag = 2.0f0 * (energy - ϵ) * 2.0f0 * imag.(oei[i].b) / Const.iters_num
        Func.ANN.update(ΔWreal, Δbreal, ΔWimag, Δbimag, i)
    end

    return error, energy, energyS, energyB, numberB
end

mutable struct DiffReal

    W::CuArray{Float32, 2}
    b::CuArray{Float32, 1}
end

mutable struct DiffComplex

    W::CuArray{ComplexF32, 2}
    b::CuArray{ComplexF32, 1}
end

function initO() end

CuArrays.@cufunc function initO()

    o   = Array{DiffReal, 1}(undef, Const.layers_num)
    oer = Array{DiffComplex, 1}(undef, Const.layers_num)
    oei = Array{DiffComplex, 1}(undef, Const.layers_num)

    for i in 1:Const.layers_num
        o[i]   = DiffReal(zeros(Float32, Const.layer[i+1], Const.layer[i]), 
                          zeros(Float32, Const.layer[i+1]))
        oer[i] = DiffComplex(zeros(ComplexF32, Const.layer[i+1], Const.layer[i]), 
                             zeros(ComplexF32, Const.layer[i+1]))
        oei[i] = DiffComplex(zeros(ComplexF32, Const.layer[i+1], Const.layer[i]), 
                             zeros(ComplexF32, Const.layer[i+1]))
    end

    return o, oer, oei
end

function calculation_energy() end

CuArrays.@cufunc function calculation_energy(filename1::String, filename2::String)

    Func.ANN.load(filename1, filename2)
 
    n = rand([1.0f0, 0.0f0],  Const.dimB)
    s = rand([1.0f0, -1.0f0], Const.dimS)
    energy  = 0.0f0
    energyS = 0.0f0
    energyB = 0.0f0
    numberB = 0.0f0

    for i in 1:Const.burnintime

        s = Func.updateS(s, n)
        n = Func.updateB(n, s)
    end

    for i in 1:Const.iters_num
        s     = Func.updateS(s, n)
        nnext = Func.updateB(n, s)

        eS = Func.energyS_shift(s, n)
        eB = Func.energyB_shift(n, s)
        e  = eS + eB
        energy    += e
        energyS   += eS
        energyB   += eB
        numberB   += sum(n)

        n = nnext
    end
    energy   = real(energy)  / Const.iters_num
    energyS  = real(energyS) / Const.iters_num
    energyB  = real(energyB) / Const.iters_num
    numberB /= Const.iters_num

    return error, energy, energyS, energyB, numberB
end

end
