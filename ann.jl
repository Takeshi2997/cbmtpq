module ANN
include("./setup.jl")
using .Const, LinearAlgebra, Flux
using Flux.Optimise: update!
using BSON: @save

reallayer1 = Dense(Const.layer[1], Const.layer[2], tanh)
reallayer2 = Dense(Const.layer[2], Const.layer[3], tanh)
reallayer3 = Dense(Const.layer[3], Const.layer[4], tanh)
reallayer4 = Dense(Const.layer[4], Const.layer[5])
f = Chain(reallayer1, reallayer2, reallayer3, reallayer4)
realps = params(f)

imaglayer1 = Dense(Const.layer[1], Const.layer[2], tanh)
imaglayer2 = Dense(Const.layer[2], Const.layer[3], tanh)
imaglayer3 = Dense(Const.layer[3], Const.layer[4], tanh)
imaglayer4 = Dense(Const.layer[4], Const.layer[5])
g = Chain(imaglayer1, imaglayer2, imaglayer3, imaglayer4)
imagps = params(g)

mutable struct Network

    f::Any
    g::Any
end

network = Network(f, g)

function save(filename)

    @save filename f g
end

function forward(n::Array{Float32, 1})

    return network.f(n) .+ im * network.g(n)
end

realloss(s, n) = dot(s, f(n))
imagloss(s, n) = dot(s, g(n))

function setupbackward(n::Array{Float32, 1}, s::Array{Float32, 1})

    realgs = gradient(() -> realloss(s, n), realps)
    imaggs = gradient(() -> imagloss(s, n), imagps)
    return realgs, imaggs
end

function backward(realgs, imaggs, i::Integer)

    return realgs[f[i].W], realgs[f[i].b],
    imaggs[g[i].W], imaggs[g[i].b]
end

opt(lr::Float32) = ADAM(lr, (0.9, 0.999))

function update(ΔWreal::Array{Float32, 2}, Δbreal::Array{Float32, 1},
                ΔWimag::Array{Float32, 2}, Δbimag::Array{Float32, 1}, i::Integer, lr::Float32)

    update!(opt(lr), f[i].W, ΔWreal)
    update!(opt(lr), f[i].b, Δbreal)
    update!(opt(lr), g[i].W, ΔWimag)
    update!(opt(lr), g[i].b, Δbimag)
end

end
