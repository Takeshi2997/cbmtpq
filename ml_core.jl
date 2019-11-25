module MLcore
    include("./setup.jl")
    include("./functions.jl")
    using .Const, .Func, LinearAlgebra

    function diff_error(weight, biasB, biasS, ϵ)

        n = zeros(Float64, Const.dimB)
        s = zeros(Float64, Const.dimS)
        energy  = 0.0
        energyS = 0.0
        energyB = 0.0
        dweight_h = zeros(Complex{Float64}, Const.dimB, Const.dimS)
        dweight   = zeros(Complex{Float64}, Const.dimB, Const.dimS)
        dbiasB_h  = zeros(Complex{Float64}, Const.dimB)
        dbiasB    = zeros(Complex{Float64}, Const.dimB)
        dbiasS_h  = zeros(Complex{Float64}, Const.dimS)
        dbiasS    = zeros(Complex{Float64}, Const.dimS)

        for i in 1:Const.burnintime
            activationB = transpose(weight) * n .+ biasS
            realactivationB = 2.0 * real(activationB)
            s = Func.updateS(realactivationB)

            activationS = weight * s .+ biasB
            realactivationS = 2.0 * real(activationS)
            n = Func.updateB(realactivationS)
        end

        for i in 1:Const.iters_num
            activationB = transpose(weight) * n .+ biasS
            realactivationB = 2.0 * real(activationB)
            s = Func.updateS(realactivationB)

            activationS = weight * s .+ biasB
            realactivationS = 2.0 * real(activationS)
            n = Func.updateB(realactivationS)

            eS = Func.energyS(s, activationB)
            eB = Func.energyB(n, activationS)
            eI = Func.energyI(n, s, weight, biasB, biasS)
            e  = eS + eB + eI
            energy    += e
            energyS   += eS
            energyB   += eB
            dweight_h += transpose(s) .* n .* e
            dweight   += transpose(s) .* n
            dbiasB_h  += n * e
            dbiasB    += n
            dbiasS_h  += s * e
            dbiasS    += s
        end
        energy    /= Const.iters_num
        energyS   /= Const.iters_num
        energyB   /= Const.iters_num
        dweight_h /= Const.iters_num
        dweight   /= Const.iters_num
        dbiasB_h  /= Const.iters_num
        dbiasB    /= Const.iters_num
        dbiasS_h  /= Const.iters_num
        dbiasS    /= Const.iters_num
        error = abs2(energy - ϵ)

        diff_weight = 2.0 * (energy - ϵ) * (dweight_h - energy * dweight)
        diff_biasB  = 2.0 * (energy - ϵ) * (dbiasB_h - energy * dbiasB)
        diff_biasS  = 2.0 * (energy - ϵ) * (dbiasS_h - energy * dbiasS)

        return error, energy, energyS, energyB,
        diff_weight, diff_biasB, diff_biasS
    end

    function forward(weight, biasB, biasS)

        n = zeros(Const.dimB)
        energy  = 0.0
        energyS = 0.0
        energyB = 0.0

        for i in 1:Const.burnintime
            activationB = transpose(weight) * n .+ biasS
            realactivationB = 2.0 * real(activationB)
            s = Func.updateS(realactivationB)

            activationS = weight * s .+ biasB
            realactivationS = 2.0 * real(activationS)
            n = Func.updateB(realactivationS)
        end

        for i in 1:Const.num
            activationB = transpose(weight) * n .+ biasS
            realactivationB = 2.0 * real(activationB)
            s = Func.updateS(realactivationB)

            activationS = weight * s .+ biasB
            realactivationS = 2.0 * real(activationS)
            n = Func.updateB(realactivationS)

            eS = Func.energyS(s, activationB)
            eB = Func.energyB(n, activationS)
            eI = Func.energyI(n, s, weight, biasB, biasS)
            e  = eS + eB + eI
            energy  += e
            energyS += eS
            energyB += eB
        end

        energy  /= Const.num
        energyS /= Const.num
        energyB /= Const.num

        return energy, energyS, energyB
    end
end
