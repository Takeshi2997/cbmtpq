include("./setup.jl")
include("./params.jl")
using LinearAlgebra

σ(x::Float64)     = 1.0 / (1.0 + exp(-x))
dσ(x::Float64)    = σ(x) * (1.0 - σ(x))
resσ(x::Float64)  = x + σ(x)
dresσ(x::Float64) = 1.0 + dσ(x)

function forward(n::Array{Float64, 1}, network::Network)

    z = transpose(network.w) * n .+ network.b
    return z
end

function backward(o::O, oe::OE, n::Array{Float64, 1}, s::Array{Float64, 1}, e::ComplexF64)

    dw = transpose(s) .* n
    db = s
    o.w  .+= dw
    o.b  .+= db
    oe.w .+= dw * e
    oe.b .+= db * e
end

