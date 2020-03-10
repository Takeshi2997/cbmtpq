module ANN
include("./setup.jl")
using .Const, LinearAlgebra, Flux
using Flux.Optimise: update!
using BSON: @save
using CuArrays

const d  = CuArray([1.0f0, 1.0f0im])
const d1 = CuArray([1.0f0, 0.0f0])
const d2 = CuArray([0.0f0, 1.0f0])

layer1 = Dense(Const.layer[1], Const.layer[2], relu) |> gpu
layer2 = Dense(Const.layer[2], Const.layer[3], relu) |> gpu
layer3 = Dense(Const.layer[3], Const.layer[4]) |> gpu
f = Chain(layer1, layer2, layer3)
ps = params(f)

mutable struct Network

    f::Any
end

network = Network(f)

function save(filename)

    @save filename f
end

function forward(x::Array{Float32, 1})

    x = x |> gpu
    out = dot(network.f(x), d)
    out = out |> cpu
    return out
end

function realloss(x::Array{Float32, 1})

    x = x |> gpu
    return dot(f(x), d1)
end

function imagloss(x::Array{Float32, 1})

    x = x |> gpu 
    return dot(f(x), d2)
end

function setupbackward(x::Array{Float32, 1})

    realgs = gradient(() -> realloss(x), ps)
    imaggs = gradient(() -> imagloss(x), ps)
    return realgs, imaggs
end

function backward(realgs, imaggs, i::Integer)

    realgsW = realgs[f[i].W] |> cpu
    realgsb = realgs[f[i].b] |> cpu
    imaggsW = imaggs[f[i].W] |> cpu
    imaggsb = imaggs[f[i].b] |> cpu
    return realgsW, realgsb, imaggsW, imaggsb
end

opt(lr::Float32) = ADAM(lr, (0.9, 0.999))

function update(ΔW::Array{Float32, 2}, Δb::Array{Float32, 1},
                i::Integer, lr::Float32)

    ΔW = ΔW |> gpu
    Δb = Δb |> gpu
    update!(opt(lr), f[i].W, ΔW)
    update!(opt(lr), f[i].b, Δb)
end

end
