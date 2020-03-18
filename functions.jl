module Func
include("./setup.jl")
include("./ann.jl")
using .Const, .ANN, LinearAlgebra

function updateS(s::Array{Float32, 1}, n::Array{Float32, 1})

    z = 2.0f0 * real.(ANN.forward(n))
    rate = exp.(-2.0f0 * s .* z)
    for ix in 1:Const.dimS
        if 1.0f0 > rate[ix]
            prob = rand(Float32)
            if prob < rate[ix]
                s[ix] *= -1.0f0
            end
        else
            s[ix] *= -1.0f0
        end
    end

    return s
end

function flip(n::Array{Float32, 1}, iy::Integer)

    nflip = copy(n)
    nflip[iy] = 1.0f0 - nflip[iy]
    return nflip
end

function flip2(n::Array{Float32, 1}, iy::Integer, ix::Integer)

    nflip = copy(n)
    nflip[iy] = 1.0f0 - nflip[iy]
    nflip[ix] = 1.0f0 - nflip[ix]
    return nflip
end

function updateB(n::Array{Float32, 1}, s::Array{Float32, 1})
    
    for iy in 1:Const.dimB
        z = ANN.forward(n)
        nflip = flip(n, iy)
        zflip = ANN.forward(nflip)
        rate = exp(2.0f0 * real(dot(s, zflip .- z)))
        if 1.0f0 > rate
            prob = rand(Float32)
            if prob < rate
                n[iy] = 1.0f0 - n[iy]
            end
        else
            n[iy] = 1.0f0 - n[iy]
        end
    end

    return n
end

function hamiltonianS_shift(s::Array{Float32, 1}, z::Array{Complex{Float32}, 1})

    out = 0.0f0im
    if s[1] != s[2]
        out += 1.0f0 - exp(-2.0f0 * transpose(z) * s)
    end

    return Const.J * out / 4.0f0
end

function energyS_shift(inputs::Array{Float32, 1}, n::Array{Float32, 1})

    z = ANN.forward(n)
    sum = 0.0f0 + 0.0f0im
    for ix in 1:Const.dimS-1
        sum += hamiltonianS_shift(inputs[ix:ix+1], z[ix:ix+1])
    end

    return sum
end

function hamiltonianB_shift(n::Array{Float32, 1}, s::Array{Float32, 1}, 
                            z::Array{Complex{Float32}, 1}, iy::Integer)

    out = 0.0f0im
    iynext = iy%Const.dimB + 1
    if n[iy] != n[iynext]
        nflip = flip2(n, iy, iynext)
        zflip = ANN.forward(nflip)
        rate  = exp(dot(s, zflip .- z))
        out  += -rate
    end

    return Const.t * out + 1.0f0
end

function energyB_shift(inputn::Array{Float32, 1}, s::Array{Float32, 1})

    z = ANN.forward(inputn)
    sum = 0.0f0im
    for iy in 1:Const.dimB
        sum += hamiltonianB_shift(inputn, s, z, iy)
    end

    return sum
end

end
