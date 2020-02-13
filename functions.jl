include("./setup.jl")
include("./params.jl")
include("./ann.jl")
using LinearAlgebra

function updateS(s::Array{Float64, 1}, n::Array{Float64, 1}, network::Network)

    z = 2.0 * real.(forward(n, network))
    rate = exp.(-2.0 * s .* z)
    for ix in 1:dimS
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

function updateB(n::Array{Float64, 1}, s::Array{Float64, 1}, network::Network)
    
    z = 2.0 * real.(network.w * s)
    rate = exp.((1.0 .- 2.0 * n) .* z)
    for iy in 1:dimB
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

function hamiltonianS_shift(s::Array{Float64, 1}, z::Array{ComplexF64, 1})

    out = 1.0
    if s[1] != s[2]
        out = -1.0 + 2.0 * exp(-2.0 * transpose(z) * s)
    end

    return -J * out / 4.0 + 1.0 / 4.0
end

function energyS_shift(inputs::Array{Float64, 1}, n::Array{Float64, 1}, network::Network)

    z = forward(n, network)
    sum = 0.0 + 0.0im
    for ix in 1:2:dimS-1
        sum += hamiltonianS_shift(inputs[ix:ix+1], z[ix:ix+1])
    end

    return sum
end

function hamiltonianB_shift(n::Array{Float64, 1}, z::Array{ComplexF64, 1})

    out = 0.0im
    s = (1.0 / 2.0 .- n) * 2.0
    if n[1] != n[2]
        out += -exp(transpose(s) * z)
    end

    return t * out + 1.0
end

function energyB_shift(inputn::Array{Float64, 1}, s::Array{Float64, 1}, network::Network)

    z = network.w * s
    sum = 0.0im
    for ix in 1:dimB-1
        sum += hamiltonianB_shift(inputn[ix:ix+1], z[ix:ix+1])
    end
    sum += 
    hamiltonianB_shift(inputn[end:-dimB+1:1], 
                       z[end:-dimB+1:1])
    return sum
end
