module Func
    include("./setup.jl")
    using .Const, LinearAlgebra, Random

    function updateV(v, rate)

        rng = MersenneTwister(1234)

        s = -ones(Float64, Const.dimS)
        n = ones(Float64, Const.dimB)
        v = vcat(n, s)
        probB = 1.0 ./ (1.0 .+ exp.(z[1:Const.dimB]))
        probS = 1.0 ./ (1.0 .+ exp.(-2.0 * z[Const.dimB+1:end]))
        prob  = vcat(probB, probS)
        pflip = rand(rng, Const.dimV)

        for iy in 1:Const.dimB
            if pflip[iy] < prob[iy]
                v[iy] = 0.0
            end
        end
        for ix in Const.dimB+1:Const.dimB+Const.dimS
            if pflip[ix] < prob[ix]
                v[ix] = 1.0
            end
        end

        return v
    end

    function updateH(h, z)

        rng = MersenneTwister(1234)

        h = -ones(Float64, Const.dimH)
        prob = 1.0 ./ (1.0 .+ exp.(-2.0 * z))
        pflip = rand(rng, Const.dimH)
        for ix in 1:Const.dimH
            if pflip[ix] < prob[ix]
                h[ix] = 1.0
            end
        end

        return h
    end

    function hamiltonianS_shift(v, weight, biasH, biasV, ix)

        out = 1.0 + 0.0im
        ixnext = Const.dimB + ix%Const.dimS + 1
        if v[ix] != v[ixnext]
            vflip = copy(v)
            vflip[ix]     = -v[ix]
            vflip[ixnext] = -v[ixnext]
            out += -1.0 + 2.0 * prod(ψ(vflip, weight, biasH, biasV) ./ ψ(v, weight, biasH, biasV))
        end

        return -Const.J * out / 4.0 + 1.0 / 4.0
    end

    function hamiltonianB_shift(v, weight, biasH, biasV, iy)

        out = 0.0im
        iynext = iy%Const.dimB + 1
        if v[iy] != v[iynext]
            vflip = copy(v)
            vflip[iy]     = 1.0 - v[iy]
            vflip[iynext] = 1.0 - v[iynext]
            out += -prod(ψ(vflip, weight, biasH, biasV) ./ ψ(v, weight, biasH, biasV))
        end

        return Const.t * out + 1.0
    end

    function ψ(v, weight, biasH, biasV)

        return exp(transpose(biasV) * v) * cosh.(weight * v .+ biasH)
    end

    function energy_shift(v, weight, biasH, biasV)

        sumS = 0.0im
        sumB = 0.0im

        for ix in 1:2:Const.dimS-1
            sumS += hamiltonianS_shift(v, weight, biasH, biasV, ix)
        end
 
        for iy in 1:Const.dimB-1
            sumB += hamiltonianB_shift(v, weight, biasH, biasV, iy)
        end

        return sumS, sumB
    end
end
