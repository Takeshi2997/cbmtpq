module Func
    include("./setup.jl")
    using .Const, LinearAlgebra, Random

    function updateV(v, weight, biasH, biasV)

        rng = MersenneTwister(1234)

        for iy in 1:Const.dimB
            vflip     = copy(v)
            vflip[iy] = 1.0 - v[iy]
            rate = prod(abs2.(ψ(vflip, weight, biasH, biasV) ./ ψ(v, weight, biasH, biasV)))
            if 1.0 > rate
                prob = rand(rng)
                if prob < rate
                    v[iy] = vflip[iy]
                end
            else
                v[iy] = vflip[iy]
            end
        end

        for ix in Const.dimB+1:Const.dimV
            vflip     = copy(v)
            vflip[ix] = -v[ix]
            rate = prod(abs2.(ψ(vflip, weight, biasH, biasV) ./ ψ(v, weight, biasH, biasV)))
            if 1.0 > rate
                prob = rand(rng)
                if prob < rate
                    v[ix] = vflip[ix]
                end
            else
                v[ix] = vflip[ix]
            end
        end

        return v
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

        out = exp(transpose(biasV) * v) * cosh.(weight * v .+ biasH)
        return out
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
