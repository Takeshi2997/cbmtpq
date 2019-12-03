module Func
    include("./setup.jl")
    using .Const, LinearAlgebra

    function update(z)

        s = -ones(Float64, Const.dim)
        prob = 1.0 ./ (1.0 .+ exp.(-2.0 * z))
        pup = rand(Float64, Const.dim)
        for ix in 1:Const.dim
            if pup[ix] < prob[ix]
                s[ix] = 1.0
            end
        end
        return s
    end

    function hS(s, z)

        out = 0.0
        if s[1] != s[2]
            out += (-1.0 + exp(-2.0 * transpose(z) * s)) / 2.0
        end

        return out
    end

    function energyS(inputs, z)

        sum = 0.0 + 0.0im
        for ix in 1:2:Const.dim-1
            sum += hS(inputs[ix:ix+1], z[ix:ix+1])
        end

        return -Const.J * sum / Const.copysize
    end

    function hB(s, z)

        out = 1.0 / 2.0
        if s[1] != s[2]
            out *= 1.0 + exp(-2.0 * transpose(s) * z)
        end

        return out
    end

    function energyB(inputs, z)

        sum = 0.0 + 0.0im
        for ix in 1:Const.dim-1
            sum += hB(inputs[ix:ix+1], z[ix:ix+1])
        end
        sum += hB(inputs[end:-Const.dim+1:1], z[end:-Const.dim+1:1])

        return 2.0 * Const.t * sum
    end

    function energyI(inputsB, inputsS, weight, biasB, biasS)

        num = 0.0
        for ix in 1:Const.dim
            for iy in 1:Const.dim
                num += exp(-inputsS[ix] * (weight[ix,iy] + biasB[iy] + biasS[ix]))
            end
        end

        return Const.δ * num
    end
end
