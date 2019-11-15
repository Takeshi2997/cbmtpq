module Func
    include("./setup.jl")
    using .Const, LinearAlgebra

    function sigmoid(x)

        return 1 ./ (exp.(-x) .+ 1.0)
    end

    function diff_sigmoid(x)

        return sigmoid(x) .* (1.0 - sigmoid(x))
    end

    function translate(β)

        return Const.ω / (exp(Const.ω * β) + 1.0)
    end

    function retranslate(ϵ)

        return log(Const.ω / ϵ - 1.0) / Const.ω
    end

    function updateB(z)

        n = ones(Float64, Const.dimB)
        prob = 1.0 ./ (1.0 .+ exp.(z))
        pzero = rand(Float64, Const.dimB)
        for ix in 1:Const.dimB
            if pzero[ix] < prob[ix]
                n[ix] = 0.0
            end
        end
        return n
    end

    function updateS(z)

        s = -ones(Float64, Const.dimS)
        prob = 1.0 ./ (1.0 .+ exp.(-2.0 * z))
        pup = rand(Float64, Const.dimS)
        for ix in 1:Const.dimS
            if pup[ix] < prob[ix]
                s[ix] = 1.0
            end
        end
        return s
    end

    function energyS(inputs, z)

        sum = 0.0 + 0.0im
        e = [1.0 + 0.0im, 1.0 + 0.0im]
        for s in Const.sset
            localsum = 0.0 + 0.0im
            for ix in 1:2:Const.dimS
                localsum += prod(e .* (s[ix:ix+1] .!= inputs[ix:ix+1])) + 
                            prod(1.0im .* s[ix:ix+1] .* (s[ix:ix+1] .!= inputs[ix:ix+1])) +
                            prod(s[ix:ix+1] .* (s[ix:ix+1] .== inputs[ix:ix+1]))
            end
            localsum = localsum / 4.0 * exp(transpose(z) * s)
            sum += localsum
        end

        return -Const.J * sum / exp(transpose(z) * inputs)
    end

    function energyB(inputn, z)

        wf = exp.(z)
        nf = exp(transpose(inputn) * z)
        return Const.ω * sum(wf .* (inputn .== 1.0)) / nf
    end

    function energyI(inputn, inputs, weight, biasB, biasS)

        ematrix = ones(Const.dimB, Const.dimS)
        e = ones(Const.dimB)
        nf = exp(transpose(inputn) * weight * inputs + 
                 transpose(inputn) * biasB + transpose(biasS) * inputs)
        n = -inputn .+ 1.0
        s = inputs

        sum = transpose(n) * ematrix * s
        factor = exp(transpose(n) * weight * s + 
                     transpose(n) * biasB + transpose(biasS) * s)
        return Const.δ * sum * factor / nf
    end

    function hamiltonian(n2, s2, n1, s1)

        out = hamiltonianS(s2, s1) + hamiltonianB(n2, n1) + 
        hamiltonianI(n2, s2, n1, s1)
        return out
    end
end
