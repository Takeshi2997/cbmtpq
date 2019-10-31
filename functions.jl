module Func
    include("./setup.jl")
    using .Const, LinearAlgebra

    function sigmoid(x)

        return 1 ./ (exp.(-x) .+ 1.0)
    end

    function diff_sigmoid(x)

        return sigmoid(x) .* (1.0 - sigmoid(x))
    end

    function translate(t)

        return Const.dimB * Const.ω / (exp(Const.ω / t) + 1.0)
    end

    function retranslate(ϵ)

        return Const.ω / log(Const.dimB * Const.ω / ϵ - 1.0)
    end

    function updateB(z)

        n = ones(Float32, Const.dimB)
        prob = 1.0 ./ (1.0 .+ exp.(z))
        pzero = rand(Float32, Const.dimB)
        for ix in 1:Const.dimB
            if pzero[ix] < prob[ix]
                n[ix] = 0.0
            end
        end
        return n
    end

    function updateS(z)

        s = -ones(Float32, Const.dimS)
        prob = 1.0 ./ (1.0 .+ exp.(-2.0 * z))
        pup = rand(Float32, Const.dimS)
        for ix in 1:Const.dimS
            if pup[ix] < prob[ix]
                s[ix] = 1.0
            end
        end
        return s
    end

    function wavefunctionfactorS(s2, n1, s1, weight, biasS)

        z = transpose(weight) * n1 .+ biasS
        zstar = conj(z)
        realz = 2.0 * real(z)
        action1 = transpose(conj(z)) * s1
        action2 = transpose(z) * s2
        marginalfactor = 
        exp(sum(log.(exp.(realz) .+ exp.(-realz))) / Const.dimS)
        return marginalfactor * exp(-(action1 + action2))
    end

    function wavefunctionfactorB(n2, s2, n1, weight, biasB)

        z = weight * s2 .+ biasB
        zstar = conj(z)
        realz = 2.0 * real(z)
        action1 = transpose(conj(z)) * n1
        action2 = transpose(z) * n2
        marginalfactor = exp(sum(log.(1.0 .+ exp.(realz))) / Const.dimB)
        return marginalfactor * exp(-(action1 + action2))
    end

    function wavefunctionfactorI(n2, s2, n1, s1, weight, biasS, biasB)

        zS = weight * s2 .+ biasB
        zB = transpose(weight) * n1 .+biasS
        zSstar = conj(zS)
        zBstar = conj(zB)
        realzS = zS .+ zSstar
        realzB = zB .+ zBstar
        action1 = transpose(zS) * n2
        action2 = transpose(realzS) * n1
        action3 = transpose(conj(zB)) * s1
        marginalfactorS = exp(sum(log.(1.0 .+ exp.(realzS))) / Const.dimB)
        marginalfactorB = 
        exp(sum(log.(exp.(realzB) .+ exp.(-realzB))) / Const.dimS)
        return marginalfactorS * marginalfactorB * 
        exp(-(action1 + action2 + action3))
    end

    function hamiltonianS(s2, s1)

        sum = 0.0
        e = [1.0 1.0]
        for ix in 1:2:Const.dimS-1
            sum += (prod(e .* (s1[ix:ix+1] .!= s2[ix:ix+1])) + 
            prod(1.0im * s1[ix:ix+1] .* (s1[ix:ix+1] .!= s2[ix:ix+1])) + 
            prod(s1[ix:ix+1] .* (s1[ix:ix+1] .== s2[ix:ix+1]))) / 4.0
        end
        return -Const.J * sum
    end

    function hamiltonianB(n2, n1)

        return Const.ω * sum(n1[n1 .== n2])
    end

    function hamiltonianI(n2, s2, n1, s1)

        w = ones(Const.dimB, Const.dimS)
        n = n1 .* (n1 .!= n2)
        s = s1 .* (s1 == s2) / 2.0
        return - Const.δ * transpose(n) * w * s
    end

    function hamiltonian(n2, s2, n1, s1)

        out = hamiltonianS(s2, s1) + hamiltonianB(n2, n1) + 
        hamiltonianI(n2, s2, n1, s1)
        return out
    end
end
