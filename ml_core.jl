module MLcore
    include("./setup.jl")
    include("./functions.jl")
    using .Const, .Func, LinearAlgebra

    function diff_error(weight, biasB, biasS, ϵ)

        sB = -ones(Float64, Const.dim)
        energy  = 0.0
        energyS = 0.0
        energyB = 0.0
        dweight_h = zeros(Complex{Float64}, Const.dim, Const.dim)
        dweight   = zeros(Complex{Float64}, Const.dim, Const.dim)
        dbiasB_h  = zeros(Complex{Float64}, Const.dim)
        dbiasB    = zeros(Complex{Float64}, Const.dim)
        dbiasS_h  = zeros(Complex{Float64}, Const.dim)
        dbiasS    = zeros(Complex{Float64}, Const.dim)

        for i in 1:Const.burnintime
            activationB = transpose(weight) * sB .+ biasS
            realactivationB = 2.0 * real.(activationB)
            sS = Func.update(realactivationB)

            activationS = weight * sS .+ biasB
            realactivationS = 2.0 * real.(activationS)
            sB = Func.update(realactivationS)
        end

        for i in 1:Const.iters_num
            activationB = transpose(weight) * sB .+ biasS
            realactivationB = 2.0 * real.(activationB)
            sS = Func.update(realactivationB)

            activationS = weight * sS .+ biasB
            realactivationS = 2.0 * real.(activationS)
            sB = Func.update(realactivationS)

            eS = Func.energyS(sS, activationB)
            eB = Func.energyB(sB, activationS)
            eI = Func.energyI(sB, sS, weight, biasB, biasS)
            e  = eS + eB + eI
            energy    += e
            energyS   += eS
            energyB   += eB
            dweight_h += transpose(sS) .* sB .* e
            dweight   += transpose(sS) .* sB
            dbiasB_h  += sB * e
            dbiasB    += sB
            dbiasS_h  += sS * e
            dbiasS    += sS
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
        error = (real(energy - ϵ))^2

        diff_weight = 2.0 * (energy - ϵ) * (dweight_h - energy * dweight)
        diff_biasB  = 2.0 * (energy - ϵ) * (dbiasB_h - energy * dbiasB)
        diff_biasS  = 2.0 * (energy - ϵ) * (dbiasS_h - energy * dbiasS)

        return error, energyS, energyB, 
        diff_weight, diff_biasB, diff_biasS
    end

    function forward()

        sB = -ones(Float64, Const.dim)
        energy  = 0.0
        energyS = 0.0
        energyB = 0.0

        for i in 1:Const.burnintime
            activationB = transpose(weight) * sB .+ biasS
            realactivationB = 2.0 * real.(activationB)
            sS = Func.update(realactivationB)

            activationS = weight * sS .+ biasB
            realactivationS = 2.0 * real.(activationS)
            sB = Func.update(realactivationS)
        end

        for i in 1:Const.num
            activationB = transpose(weight) * sB .+ biasS
            realactivationB = 2.0 * real.(activationB)
            sS = Func.update(realactivationB)

            activationS = weight * sS .+ biasB
            realactivationS = 2.0 * real.(activationS)
            sB = Func.update(realactivationS)

            eS = Func.energyS(sS, activationB)
            eB = Func.energyB(sB, activationS)
            eI = Func.energyI(sB, sS, weight, biasB, biasS)
            e  = eS + eB + eI
            energy    += e
            energyS   += eS
            energyB   += eB
        end
        energy    /= Const.num
        energyS   /= Const.num
        energyB   /= Const.num

        return energy, energyS, energyB
    end
end
