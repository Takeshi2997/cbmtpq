module MLcore
    include("./setup.jl")
    include("./functions.jl")
    using .Const, .Func, LinearAlgebra

    function diff_error(network, ϵ)

        (weight, biasB, biasS) = network
        n = zeros(Const.dimB)
        normalizefactor = 0.0
        energy = 0.0
        energyS = 0.0
        energyB = 0.0
        squareenergy = 0.0
        dweight_h = zeros(Float32, Const.dimB, Const.dimS)
        dweight = zeros(Float32, Const.dimB, Const.dimS)
        dbiasB_h = zeros(Float32, Const.dimB)
        dbiasB = zeros(Float32, Const.dimB)
        dbiasS_h = zeros(Float32, Const.dimS)
        dbiasS = zeros(Float32, Const.dimS)

        for i in 1:Const.burnintime
            activationB = transpose(n) * weight .+ biasS
            realactivationB = 2.0 * real(activationB)
            s = Func.updateS(realactivationB)

            activationS = weight * s .+ biasB
            realactivationS = 2.0 * real(activationS)
            n = Func.updateB(realactivationS)
        end

        for i in 1:Const.iters_num
            activationB = transpose(n) * weight .+ biasS
            realactivationB = 2.0 * real(activationB)
            s = Func.updateS(realactivationB)
            s1 = s
            ϕ1 = Func.wavefunctionS(s1, activationB)

            activationS = weight * s .+ biasB
            realactivationS = 2.0 * real(activationS)
            n = Func.updateB(realactivationS)
            n1 = n
            ψ1 = Func.wavefunctionB(n1, activationS)

            activationB = transpose(n) * weight .+ biasS
            realactivationB = 2.0 * real(activationB)
            s = Func.updateS(realactivationB)
            s2 = s
            ϕ2 = Func.wavefunctionS(s2, activationB)

            activationS = weight * s .+ biasB
            realactivationS = 2.0 * real(activationS)
            n = Func.updateB(realactivationS)
            n2 = n
            ψ2 = Func.wavefunctionB(n2, activationB)

            eS = Func.hamiltonianS(s1, s2) * ϕ1 * ϕ2
            eB = Func.hamiltonianB(n1, n2) * ψ1 * ψ2
            eint = Func.hamiltonianint(n1, n2, s1, s2) * ϕ1 * ψ1 * ϕ2 * ψ2
            e = eS + eB + eint
            e2 = Func.squarehamiltonian(n1, n2, s1, s2) * ϕ1 * ψ1 * ϕ2 * ψ2
            energy += e
            energyS += eS
            energyB += eB
            squareenergy += e2
            dweight_h +=  (transpose(s1) .* n1 + transpose(s2) .* n2) .* e
            dweight +=  transpose(s1) .* n1 + transpose(s2) .* n2
            dbiasB_h += n * e
            dbiasB += n
            dbiasS_h += s * e
            dbiasS += s
            normalizefactor += abs2(ϕ1) + abs2(ϕ2)
        end
        energy /= Const.iters_num * normalizefactor^2
        energyS /= Const.iters_num * normalizefactor
        energyB /= Const.iters_num * normalizefactor
        squareenergy /= Const.iters_num * normalizefactor^2
        dweight_h /= Const.iters_num * normalizefactor^2
        dweight /= Const.iters_num
        dbiasB_h /= Const.iters_num * normalizefactor^2
        dbiasB /= Const.iters_num
        dbiasS_h /= Const.iters_num * normalizefactor^2
        dbiasS /= Const.iters_num
        dispersion = squareenergy - energy^2
        error = abs2(energy - ϵ)

        diff_weight = 2.0 * (energy - ϵ) * (dweight_h - energy * dweight)
        diff_biasB = 2.0 * (energy - ϵ) * (dbiasB_h - energy * dbiasB)
        diff_biasS = 2.0 * (energy - ϵ) * (dbiasS_h - energy * dbiasS)

        return error, energy, energyS, energyB, dispersion, 
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
