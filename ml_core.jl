module MLcore
    include("./setup.jl")
    include("./functions.jl")
    using .Const, .Func, LinearAlgebra

    function diff_error(network, ϵ)

        (weight, biasB, biasS) = network
        n = zeros(Const.dimB)
        energy = 0.0
        energyS = 0.0
        energyB = 0.0
        dweight_h = zeros(Complex{Float32}, Const.dimB, Const.dimS)
        dweight = zeros(Complex{Float32}, Const.dimB, Const.dimS)
        dbiasB_h = zeros(Complex{Float32}, Const.dimB)
        dbiasB = zeros(Complex{Float32}, Const.dimB)
        dbiasS_h = zeros(Complex{Float32}, Const.dimS)
        dbiasS = zeros(Complex{Float32}, Const.dimS)

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
            s1 = s

            activationS = weight * s .+ biasB
            realactivationS = 2.0 * real(activationS)
            n = Func.updateB(realactivationS)
            n1 = n

            activationB = transpose(weight) * n .+ biasS
            realactivationB = 2.0 * real(activationB)
            s = Func.updateS(realactivationB)
            s2 = s

            activationS = weight * s .+ biasB
            realactivationS = 2.0 * real(activationS)
            n = Func.updateB(realactivationS)
            n2 = n

            eS = Func.hamiltonianS(s2, s1) * 
            Func.wavefunctionfactorS(s2, n1, s1, weight, biasS)
            eB = Func.hamiltonianB(n2, n1) * 
            Func.wavefunctionfactorB(n2, s2, n1, weight, biasB)
            eI = Func.hamiltonianI(n2, s2, n1, s1) * 
            Func.wavefunctionfactorI(n2, s2, n1, s1, weight, biasS, biasB)
            e = eS + eB + eI
            energy += e
            energyS += eS
            energyB += eB
            dweight_h +=  (transpose(s1) .* n1 + transpose(s2) .* n2) .* e
            dweight +=  transpose(s1) .* n1 + transpose(s2) .* n2
            dbiasB_h += n * e
            dbiasB += n
            dbiasS_h += s * e
            dbiasS += s
        end
        energy /= Const.iters_num
        energyS /= Const.iters_num
        energyB /= Const.iters_num
        dweight_h /= Const.iters_num
        dweight /= Const.iters_num
        dbiasB_h /= Const.iters_num
        dbiasB /= Const.iters_num
        dbiasS_h /= Const.iters_num
        dbiasS /= Const.iters_num
        error = abs2(energy - ϵ)

        diff_weight = 2.0 * (energy - ϵ) * (dweight_h - energy * dweight)
        diff_biasB = 2.0 * (energy - ϵ) * (dbiasB_h - energy * dbiasB)
        diff_biasS = 2.0 * (energy - ϵ) * (dbiasS_h - energy * dbiasS)

        return error, energy, energyS, energyB,
        diff_weight, diff_biasB, diff_biasS
    end

    function forward(network)

        (weight, biasB, biasS, η) = network
        s = [1.0, 1.0]
        energyS = 0.0
        energyB = 0.0
        num = 10000

        for i in 1:num+Const.burnintime
            activationS = weight * s .+ biasB
            n = Func.updateB(activationS)
            activationB = transpose(n) * weight .+ biasS
            s = Func.updateS(activationB)
            if i > Const.burnintime
                energyS += Func.energyS(s)
                energyB += Func.energyB(n)
            end
        end
        energyS /= num
        energyB /= num

        return energyS, energyB
    end
end
