include("./setup.jl")
include("./params.jl")
using LinearAlgebra

σ(x::Float64)     = 1.0 / (1.0 + exp(-x))
dσ(x::Float64)    = σ(x) * (1.0 - σ(x))
resσ(x::Float64)  = x + σ(x)
dresσ(x::Float64) = 1.0 + dσ(x)

function forward(n::Array{Float64, 1}, network::Network)

    zr = transpose(network.wr) * n .+ network.br
    zi = transpose(network.wi) * n .+ network.bi
    return resσ.(zr) .+ im * resσ.(zi)
end

function backward(o::O, oe::OE, network::Network, n::Array{Float64, 1}, s::Array{Float64, 1}, e::ComplexF64)

    zr = transpose(network.wr) * n .+ network.br
    zi = transpose(network.wi) * n .+ network.bi
    c  = dresσ.(zr)
    d  = dresσ.(zi)
    dwr = transpose(s .* c) .* n
    dbr = s .* c
    dwi = transpose(s .* d) .* n
    dbi = s .* d
    o.wr  .+= dwr
    o.br  .+= dbr
    oe.wr .+= dwr * e
    oe.br .+= dbr * e
    oe.wi .+= dwi * e
    oe.bi .+= dbi * e
end

