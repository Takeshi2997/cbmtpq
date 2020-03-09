module Func
include("./setup.jl")
include("./ann.jl")
using .Const, .ANN, LinearAlgebra

function updateS(s::Array{Float64, 1}, n::Array{Float64, 1})

    z = ANN.forward(s)
    for ix in 1:Const.dimS
        sflip = flip(s, ix)
        zflip = ANN.forward(sflip)
        rate = exp(2.0 * real(dot(n, zflip .- z)))
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

function flip(s::Array{Float64, 1}, ix::Integer)

    sflip = copy(s)
    sflip[ix] *= -1.0
    return sflip
end

function flip2(s::Array{Float64, 1}, iy::Integer, ix::Integer)

    sflip = copy(s)
    sflip[iy] *= -1.0
    sflip[ix] *= -1.0
    return sflip
end

function updateB(n::Array{Float64, 1}, s::Array{Float64, 1})
    
    z = 2.0 * real.(ANN.forward(s))
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

function hamiltonianS_shift(s::Array{Float64, 1}, n::Array{Float64, 1}, 
                            z::Array{ComplexF64, 1}, ix::Integer)

    out = 0.0im
    ixnext = ix%Const.dimS + 1
    if s[ix] != s[ixnext]
        sflip = flip2(s, ix, ixnext)
        zflip = ANN.forward(sflip)
        rate  = exp(dot(n, zflip .- z))
        out   = 1.0 - rate
    end
 
    return Const.J * out / 4.0
end

function energyS_shift(inputs::Array{Float64, 1}, n::Array{Float64, 1})

    z = ANN.forward(inputs)
    sum = 0.0im
    for ix in 1:2:Const.dimS-1
        sum += hamiltonianS_shift(inputs, n, z, ix)
    end

    return sum
end

function hamiltonianB_shift(n::Array{Float64, 1}, z::Array{ComplexF64, 1})

    out = 0.0im
    if n[1] != n[2]
        s = (1.0 / 2.0 .- n) * 2.0
        out += -exp(transpose(s) * z)
    end

    return Const.t * out + 1.0
end

function energyB_shift(inputn::Array{Float64, 1}, s::Array{Float64, 1})

    z = ANN.forward(s)
    sum = 0.0im
    for iy in 1:Const.dimB-1
        sum += hamiltonianB_shift(inputn[iy:iy+1], z[iy:iy+1])
    end
    sum += 
    hamiltonianB_shift(inputn[end:-Const.dimB+1:1], 
                       z[end:-Const.dimB+1:1])

    return sum
end

end
