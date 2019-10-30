module Func
    include("./setup.jl")
    using .Const

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

    function wavefunctionS(s, z)

        realz = 2.0 * real(z)
        action = z * s
        partitionfunction = prob(exp.(realz) .+ exp.(-realz))
        return exp(action) / partitionfunction
    end

    function wavefunctionS(n, z)

        realz = 2.0 * real(z)
        action = transpose(n) * z
        partitionfunction = prob(1.0 .+ exp.(-realz))
        return exp(action) / partitionfunction
    end

    function hamiltonianS(s)

        sum = 0.0
        for ix in 1:2:Const.dimS-1
            sum += s[ix] * s[ix + 1]
        end
        return -Const.J * sum
    end

    function hamiltonianB(n1, n2)

        return Const.ω * sum(n1[n1 .== n2])
    end

    function hamiltonianint(n1, n2, s1, s2)

        w = ones(Const.dimB, Const.dimS)
        bassmatrix = 
        return - Const.δ * transpose(n) * w * s
    end

    function hamiltonian(n, s)


        return energyB(n) + energyS 
    end

    function squarehamiltonian(n, s)

        h = hamiltonian(n, s)
        return h^2
    end
end
