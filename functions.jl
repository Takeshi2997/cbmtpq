module Func
    include("./setup.jl")
    using .Const, LinearAlgebra, Random

    function updateS(s, n, network)

        z = 2.0 * real.(transpose(network.w) * n)
        rate = exp.(-2.0 * s .* z)
        for ix in 1:Const.dimS
            if 1.0 > rate[ix]
                prob = rand(Float64)
                if prob < rate[ix]
                    s[ix] *= -1.0
                end
            else
                s[ix] *= -1.0
            end
        end

        return s
    end

    function updateB(n, s, network)
        
        z = 2.0 * real.(network.w * s .+ network.b)
        rate = exp.((1.0 .- 2.0 * n) .* z)
        for iy in 1:Const.dimB
            if 1.0 > rate[iy]
                prob = rand(Float64)
                if prob < rate[iy]
                    n[iy] = 1.0 - n[iy]
                end
            else
                n[iy] = 1.0 - n[iy]
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

    function energyS_shift(inputs, n, network)

        z = transpose(network.w) * n
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

    function energyB_shift(inputn, s, network)

        z = network.w * s .+ network.b
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
