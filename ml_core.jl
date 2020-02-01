module MLcore
    include("./setup.jl")
    include("./functions.jl")
    using .Const, .Func, LinearAlgebra

    mutable struct Network
    
        weight::Array{Complex{Float64}, 2}
        biasH::Array{Complex{Float64}, 1}
        biasV::Array{Complex{Float64}, 1}
    end

    function diff_error(network, ϵ)

        weight = network.weight
        biasH  = network.biasH
        biasV  = network.biasV

        n = rand([1.0, 0.0], Const.dimB)
        s = rand([1.0, -1.0], Const.dimS)
        v = vcat(n, s)
        energy  = 0.0
        energyS = 0.0
        energyB = 0.0
        numberB = 0.0
        dweight_h = zeros(Complex{Float64}, Const.dimH, Const.dimV)
        dweight   = zeros(Complex{Float64}, Const.dimH, Const.dimV)
        dbiasH_h  = zeros(Complex{Float64}, Const.dimH)
        dbiasH    = zeros(Complex{Float64}, Const.dimH)
        dbiasV_h  = zeros(Complex{Float64}, Const.dimV)
        dbiasV    = zeros(Complex{Float64}, Const.dimV)

        for i in 1:Const.burnintime
            v = Func.updateV(v, weight, biasH, biasV)
        end

        for i in 1:Const.iters_num
            v = Func.updateV(v, weight, biasH, biasV)
            activationV = weight * v .+ biasH
 
            n = v[1:Const.dimB]
            s = v[Const.dimB+1:end]

            eS, eB = Func.energy_shift(v, weight, biasH, biasV)

            e  = eS + eB
            energy    += e
            energyS   += eS
            energyB   += eB
            numberB   += sum(n)
            dweight_h += transpose(v) .* tanh.(activationV) * e
            dweight   += transpose(v) .* tanh.(activationV)
            dbiasH_h  += tanh.(activationV) * e
            dbiasH    += tanh.(activationV)
            dbiasV_h  += v * e
            dbiasV    += v
        end
        energy     = real(energy) / Const.iters_num
        energyS    = real(energyS) / Const.iters_num
        energyB    = real(energyB) / Const.iters_num
        numberB   /= Const.iters_num
        dweight_h /= Const.iters_num
        dweight   /= Const.iters_num
        dbiasH_h  /= Const.iters_num
        dbiasH    /= Const.iters_num
        dbiasV_h  /= Const.iters_num
        dbiasV    /= Const.iters_num
        error   = (energy - ϵ)^2

        diff_weight = 2.0 * (energy - ϵ) * (dweight_h - energy * dweight)
        diff_biasH  = 2.0 * (energy - ϵ) * (dbiasH_h - energy * dbiasH)
        diff_biasV  = 2.0 * (energy - ϵ) * (dbiasV_h - energy * dbiasV)

        return error, energyS, energyB, numberB,
        diff_weight, diff_biasH, diff_biasV
    end

    function forward(weight, biasB, biasS)

        n = rand([1.0, 0.0], Const.dimB)
        s = rand([1.0, -1.0], Const.dimS)
        v = vcat(n, s)
        energy  = 0.0
        energyS = 0.0
        energyB = 0.0
        numberB = 0.0

        for i in 1:Const.burnintime
            v = Func.updateV(v, weight, biasH, biasV)
        end

        for i in 1:Const.num
            v = Func.updateV(v, weight, biasH, biasV)
 
            n = v[1:Const.dimB]
            s = v[Const.dimB+1:end]

            eS, eB = Func.energy_shift(v, weight, biasH, biasV)
            e  = eS + eB
            energy    += e
            energyS   += eS
            energyB   += eB
            numberB   += sum(n)

            n = nnext
        end
        energy   = real(energy) / Const.num
        energyS  = real(energyS) / Const.num
        energyB  = real(energyB) / Const.num
        numberB /= Const.num

        return energyS, energyB, numberB
   end
end
