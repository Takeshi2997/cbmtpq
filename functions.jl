module Func
    include("./setup.jl")
    using .Const, LinearAlgebra

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

    function hamiltonianB_shift(n, z)

        out = 0.0im
        s = (1.0 / 2.0 .- n) * 2.0
        if n[1] != n[2]
            out += -exp(transpose(s) * z)
        end

        return Const.t * out + 1.0
    end

    function energyB_shift(inputn, z)

        sum = 0.0im
        for ix in 1:Const.dimB-1
            sum += hamiltonianB_shift(inputn[ix:ix+1], z[ix:ix+1])
        end
        sum += 
        hamiltonianB_shift(inputn[end:-Const.dimB+1:1], 
                           z[end:-Const.dimB+1:1])
        return sum
    end
end
