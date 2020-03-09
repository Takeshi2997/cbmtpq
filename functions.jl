module Func
include("./setup.jl")
include("./ann.jl")
using .Const, .ANN, LinearAlgebra

function updateS(x::Array{Float64, 1})

    for ix in 1:Const.dimS
        z = ANN.forward(x)
        xflip = flip(x, ix)
        zflip = ANN.forward(xflip)
        rate = exp(2.0 * real(zflip .- z))
        if 1.0 > rate
            prob = rand(Float64)
            if prob < rate
                x[ix] *= -1.0
            end
        else
            x[ix] *= -1.0
        end
    end

    return x
end

function flip(x::Array{Float64, 1}, ix::Integer)

    xout = copy(x)
    if ix <= Const.dimS
        xout[ix] *= -1.0
    else
        xout[ix] = 1.0 - xout[ix]
    end

    return xout
end

function flip(x::Array{Float64, 1}, iy::Integer, ix::Integer)

    xout = copy(x)
    if ix <= Const.dimS
        xout[ix] *= -1.0
        xout[iy] *= -1.0
    else
        xout[ix] = 1.0 - xout[ix]
        xout[iy] = 1.0 - xout[iy]
    end

    return xout
end

function updateB(x::Array{Float64, 1})
    
    for iy in Const.dimS+1:Const.dimS+Const.dimB
        z = ANN.forward(x)
        xflip = flip(x, iy)
        zflip = ANN.forward(xflip)
        rate = exp(2.0 * real(zflip .- z))
        if 1.0 > rate
            prob = rand(Float64)
            if prob < rate
                x[iy] = 1.0 - x[iy]
            end
        else
            x[iy] = 1.0 - x[iy]
        end
    end

    return x
end

function hamiltonianS_shift(x::Array{Float64, 1}, 
                            z::ComplexF64, ix::Integer)

    out = 0.0im
    ixnext = ix%Const.dimS + 1
    if x[ix] != x[ixnext]
        xflip = flip(x, ix, ixnext)
        zflip = ANN.forward(xflip)[1]
        rate  = exp(zflip .- z)
        out   = 1.0 - rate
    end
 
    return Const.J * out / 4.0
end

function energyS_shift(x::Array{Float64, 1})

    z = ANN.forward(x)
    sum = 0.0im
    for ix in 1:2:Const.dimS-1
        sum += hamiltonianS_shift(x, z, ix)
    end

    return sum
end

function hamiltonianB_shift(x::Array{Float64, 1}, 
                            z::ComplexF64, iy::Integer)

    out = 0.0im
    iynext = Const.dimS + iy%Const.dimB + 1
    if x[iy] != x[iynext]
        xflip = flip(x, iy, iynext)
        zflip = ANN.forward(xflip)
        rate  = exp(zflip .- z)
        out  += -rate
    end

    return Const.t * out + 1.0
end

function energyB_shift(x::Array{Float64, 1})

    z = ANN.forward(x)
    sum = 0.0im
    for iy in 1:Const.dimB
        sum += hamiltonianB_shift(x, z, iy)
    end

    return sum
end

end
