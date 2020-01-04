module Func
    include("./setup.jl")
    using .Const, LinearAlgebra

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

    function updateB(z)

        n = ones(Float64, Const.dimB)
        prob = 1.0 ./ (1.0 .+ exp.(z))
        pup = rand(Float64, Const.dimB)
        for ix in 1:Const.dimB
            if pup[ix] < prob[ix]
                n[ix] = 0.0
            end
        end
        return n
    end

    function hamiltonianS_shift(s, z)

        out = 1.0
        if s[1] != s[2]
            out = -1.0 + 2.0 * exp(-2.0 * transpose(z) * s)
        end

        return -Const.J * out / 4.0 + 1.0 / 4.0
    end

    function energyS_shift(inputs, z)

        sum = 0.0 + 0.0im
        for ix in 1:2:Const.dimS-1
            sum += hamiltonianS_shift(inputs[ix:ix+1], z[ix:ix+1])
        end
 
        return sum
    end

    function hamiltonianB_shift(n, z, μ)

        out = 0.0im
        s = (1.0 / 2.0 .- n) * 2.0
        if n[1] != n[2]
            out += -exp(transpose(s) * z)
        end

        return Const.t * out - μ / 2.0 * (n[1] + n[2]) + 1.0
    end

    function energyB_shift(inputn, z, μ)

        sum = 0.0im
        for ix in 1:Const.dimB-1
            sum += hamiltonianB_shift(inputn[ix:ix+1], z[ix:ix+1], μ)
        end
        sum += 
        hamiltonianB_shift(inputn[end:-Const.dimB+1:1], 
                           z[end:-Const.dimB+1:1], μ)
        return sum
    end
end
