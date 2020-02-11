module Func
    include("./setup.jl")
    include("./ann.jl")
    using .Const, .ANN, LinearAlgebra, Random

<<<<<<< HEAD
    function updateS(s, n, network)

        z = 2.0 * real.(transpose(network.w) * n)
        rate = exp.(-2.0 * s .* z)
>>>>>>> origin/ann
        for ix in 1:Const.dimS
            sflip      = copy(s)
            sflip[ix] *= -1
            xflip = weight * sflip .+ bias
            x     = weight * s .+ bias
            zflip = ANN.forward(xflip)
            z     = ANN.forward(x)
            rate = abs2(ψ(n, zflip) / ψ(n, z))
            if 1.0 > rate
                prob = rand(Float64)
                if prob < rate
                    s[ix] *= -1.0
                end
            else
                s[ix] *= -1.0
            end
        end

        return s
    end

<<<<<<< HEAD
    function updateB(n, s, network)
        
        z = 2.0 * real.(network.w * s .+ network.b)
        rate = exp.((1.0 .- 2.0 * n) .* z)
=======
>>>>>>> origin/ann
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

    function ψ(n::Array, z::Array{ComplexF64, 1})

        return exp(transpose(n) * z)
    end

    function hamiltonianS_shift(s::Array, weight::Array{ComplexF64, 2}, 
                                bias::Array{ComplexF64, 1}, n::Array, ix)

        out = 1.0
        ixnext = ix%Const.dimS + 1
        if s[ix] != s[ixnext]
            sflip = copy(s)
            sflip[ix]     *= -1
            sflip[ixnext] *= -1
            xflip = weight * sflip .+ bias
            x     = weight * s .+ bias
            zflip = ANN.forward(xflip)
            z     = ANN.forward(x)
            out = -1.0 + 2.0 * ψ(n, zflip) / ψ(n, z)
        end

        return -Const.J * out / 4.0 + 1.0 / 4.0
    end

<<<<<<< HEAD
    function energyS_shift(inputs, n, network)
>>>>>>> origin/ann

        z = transpose(network.w) * n
        sum = 0.0 + 0.0im
        for ix in 1:2:Const.dimS-1
            sum += hamiltonianS_shift(inputs, weight, bias, n, ix)
        end
 
        return sum
    end

    function hamiltonianB_shift(n::Array, z::Array{ComplexF64, 1})

        out = 0.0im
        s = (1.0 / 2.0 .- n) * 2.0
        if n[1] != n[2]
            out += -exp(transpose(s) * z)
        end

        return Const.t * out + 1.0
    end

<<<<<<< HEAD
    function energyB_shift(inputn, s, network)
>>>>>>> origin/ann

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
