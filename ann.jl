module ANN
include("./setup.jl")
using .Const, LinearAlgebra, Flux
using Flux.Optimise: update!
using BSON: @save

layer1 = Dense(Const.layer[1], Const.layer[2], relu)
layer2 = Dense(Const.layer[2], Const.layer[3], relu)
layer3 = Dense(Const.layer[3], Const.layer[4])
f = Chain(layer1, layer2, layer3)
ps = params(f)

mutable struct Network

    f::Any
end

network = Network(f)

function save(filename)

    @save filename f
end

function forward(n::Array{Float32, 1})

    out = network.f(n)
    return out[1:Const.dimS] .+ im * out[Const.dimS+1:end]
end

function realloss(s, n)
    
    out = f(n)
    return dot(s, out[1:Const.dimS])
end

function imagloss(s, n)
    
    out = f(n)
    return dot(s, out[Const.dimS+1:end])
end

function setupbackward(n::Array{Float32, 1}, s::Array{Float32, 1})

    realgs = gradient(() -> realloss(s, n), ps)
    imaggs = gradient(() -> imagloss(s, n), ps)
    return realgs, imaggs
end

function backward(realgs, imaggs, i::Integer)

    return realgs[f[i].W], realgs[f[i].b],
    imaggs[f[i].W], imaggs[f[i].b]
end

opt(lr::Float32) = ADAM(lr, (0.9, 0.999))

function update(ΔW::Array{Float32, 2}, Δb::Array{Float32, 1},
                i::Integer, lr::Float32)

    update!(opt(lr), f[i].W, ΔW)
    update!(opt(lr), f[i].b, Δb)
end

end
