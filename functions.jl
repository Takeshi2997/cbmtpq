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
        realz = z .+ zstar
        factor = (2.0 .* cosh.(realz)) ./ 
        exp.(s2 .* z .+ s1 .* zstar)
        return exp(sum(log.(factor)) / Const.dimS)
    end

    function wavefunctionfactorB(n2, s2, n1, weight, biasB)

        z = weight * s2 .+ biasB
        zstar = conj(z)
        realz = z .+ zstar
        factor = (1.0 .+ exp.(realz)) ./
        exp.(n2 .* z .+ n1 .* zstar)
        return exp(sum(log.(factor)) / Const.dimB)
    end

    function wavefunctionfactorI(n2, s2, n1, s1, weight, biasS, biasB)

        zS = weight * s2 .+ biasB
        zB = transpose(weight) * n2 .+biasS
        zSstar = conj(weight * s1 .+ biasB)
        zBstar = conj(transpose(weight) * n1 .+biasS)
        realzS = zS .+ conj(zS)
        realzB = conj(zB) .+ zBstar
        factor1 =  (2.0 .* cosh.(realzB))./ 
        exp.(s2 .* zB .+ s2 .* zBstar)
        factor2 = (1.0 .+ exp.(realzS)) ./
        exp.(n1 .* zS .+ n1 .* zSstar)
        return exp(sum(log.(factor1)) / Const.dimS) * 
        exp(sum(log.(factor2)) / Const.dimB)
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
        e = ones(Const.dimB)
        n = e .* (n1 .!= n2)
        s = s1 .* (s1 == s2) / 2.0
        return - Const.δ * transpose(n) * w * s
    end

    function hamiltonian(n2, s2, n1, s1)

        out = hamiltonianS(s2, s1) + hamiltonianB(n2, n1) + 
        hamiltonianI(n2, s2, n1, s1)
        return out
    end
end
