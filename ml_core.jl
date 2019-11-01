module MLcore
    include("./setup.jl")
    include("./functions.jl")
    using .Const, .Func, LinearAlgebra

    function diff_error(network, ϵ)

        (weight, biasB, biasS) = network
        n = zeros(Const.dimB)
        s = zeros(Const.dimS)
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

        s1 = s
        n1 = n

        for i in 1:Const.iters_num
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
            dbiasB_h += (n1 + n2)* e
            dbiasB += n1 + n2
            dbiasS_h += (s1 + s2) * e
            dbiasS += s1 + s2

            s1 = s2
            n1 = n2
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

        (weight, biasB, biasS) = network
        n = zeros(Const.dimB)
        energy = 0.0
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
        end
        energy /= Const.num
        energyS /= Const.num
        energyB /= Const.num

        return energy, energyS, energyB
    end

 end
