module Func
    include("./setup.jl")
    using .Const, LinearAlgebra

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
        for ix in 1:Const.systemsize:Const.dimS
            for s in Const.sset
                sum += (prod(e .* (s .!= inputs[ix:ix+1])) + 
                        prod(1.0im .* s .* (s .!= inputs[ix:ix+1])) +
                        prod(s .* (s .== inputs[ix:ix+1]))) / 
                4.0 * exp(transpose(z[ix:ix+1]) * (s - inputs[ix:ix+1]))
            end
        end

        return -Const.J * sum / Const.copysize
    end

    function control(n, z)
   
        s = -(n .- 1.0 / 2.0) * 2.0
        out = 0.0
        if n[1] != n[2]
            out += exp(transpose(s) * z)
        end

        return out
    end
    
    function energyB(inputn, z)

        moment = 0.0 + 0.0im
        for iy in 1:Const.dimB-1
            moment += control(inputn[iy:iy+1], z[iy:iy+1])
        end
        moment += control(inputn[end:-Const.dimB+1:1], z[end:-Const.dimB+1:1])

        return -Const.t * moment
    end

    function numberB(inputn)

        return sum(inputn)
    end

    function energyI(inputn, inputs, weight, biasB, biasS)

        num = 0.0
        for ix in 1:Const.dimS
            for iy in 1:Const.dimB
                num += exp(-inputs[ix] * (weight[ix,iy] + biasB[iy] + biasS[ix]))
            end
        end

        return Const.δ * num
    end
end
