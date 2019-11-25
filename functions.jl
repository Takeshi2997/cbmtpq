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

    function energyS_shift(inputs, z)

        sum = 0.0 + 0.0im
        e = [1.0 + 0.0im, 1.0 + 0.0im]
        for ix in 1:2:Const.dimS
            for s in Const.sset
                sum += (prod(e .* (s .!= inputs[ix:ix+1])) + 
                        prod(1.0im .* s .* (s .!= inputs[ix:ix+1])) +
                        prod(s .* (s .== inputs[ix:ix+1])) - 
                        prod(e .* (s .== inputs[ix:ix+1]))) / 
                4.0 * exp(transpose(z[ix:ix+1]) * (s - inputs[ix:ix+1]))
            end
        end

        return -Const.J * sum
    end

    function energyS(inputs, z)

        sum = 0.0 + 0.0im
        e = [1.0 + 0.0im, 1.0 + 0.0im]
        for ix in 1:2:Const.dimS
            for s in Const.sset
                sum += (prod(e .* (s .!= inputs[ix:ix+1])) + 
                        prod(1.0im .* s .* (s .!= inputs[ix:ix+1])) +
                        prod(s .* (s .== inputs[ix:ix+1]))) / 
                4.0 * exp(transpose(z[ix:ix+1]) * (s - inputs[ix:ix+1]))
            end
        end

       return -Const.J * sum
    end

   function energyB(inputn, z)

        return Const.ω * sum(inputn)
    end

    function energyI(inputn, inputs, weight, biasB, biasS)

        ematrix = ones(Const.dimB, Const.dimS)
        e = ones(Const.dimB)
        nf = exp(transpose(inputn) * weight * inputs + 
                 transpose(inputn) * biasB + transpose(biasS) * inputs)
        reversen = -inputn .+ 1.0
        s = inputs

        sum = transpose(reversen) * ematrix * s
        factor = exp(transpose(reversen) * weight * s + 
                     transpose(reversen) * biasB + transpose(biasS) * s)
        return Const.δ * sum * factor / nf
    end
end
