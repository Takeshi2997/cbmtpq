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

        s = -ones(Float64, Const.dimB)
        prob = 1.0 ./ (1.0 .+ exp.(-2.0 * z))
        pup = rand(Float64, Const.dimB)
        for ix in 1:Const.dimB
            if pup[ix] < prob[ix]
                s[ix] = 1.0
            end
        end
        return s
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
            sum += hamiltonianB_shift(inputs[ix:ix+1], z[ix:ix+1])
        end

        return sum
    end

    function hamiltonianB_shift(s, z)

        out = 0.0
        if s[1] != s[2]
            out = -exp(-2.0 * transpose(s) * z)
        end

        return Const.t * out / 2.0 + 1.0 / 2.0
    end

    function energyB_shift(inputs, z)

        sum = 0.0 + 0.0im
        for ix in 1:Const.dimB-1
            sum += hamiltonianB_shift(inputs[ix:ix+1], z[ix:ix+1])
        end
        sum += hamiltonianB_shift(inputs[end:-Const.dimB+1:1], z[end:-Const.dimB+1:1])
        return sum
    end
end
