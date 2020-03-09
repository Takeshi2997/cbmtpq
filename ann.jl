module ANN
include("./setup.jl")
using .Const, LinearAlgebra, Flux
using Flux.Optimise: update!
using BSON: @save

reallayer1 = Dense(Const.layer[1], Const.layer[2], relu) |> f64
reallayer2 = Dense(Const.layer[2], Const.layer[3], relu) |> f64
reallayer3 = Dense(Const.layer[3], Const.layer[4]) |> f64
f = Chain(reallayer1, reallayer2, reallayer3)
ps = params(f)

mutable struct Network

    f::Any
end

network = Network(f)

function save(filename)

    @save filename f
end

function forward(x::Array{Float64, 1})

    return network.f(x)[1] .+ im * network.f(x)[2]
end

realloss(x::Array{Float64, 1}) = f(x)[1]
imagloss(x::Array{Float64, 1}) = f(x)[2]

function setupbackward(x::Array{Float64, 1})

    realgs = gradient(() -> realloss(x), ps)
    imaggs = gradient(() -> imagloss(x), ps)
    return realgs, imaggs
end

function backward(realgs, imaggs, i::Integer)

    return realgs[f[i].W], realgs[f[i].b],
    imaggs[f[i].W], imaggs[f[i].b]
end

opt(lr::Float64) = ADAM(lr, (0.9, 0.999))

function update(ΔWreal::Array{Float64, 2}, Δbreal::Array{Float64, 1},
                i::Integer, lr::Float64)

    update!(opt(lr), f[i].W, ΔWreal)
    update!(opt(lr), f[i].b, Δbreal)
end

end
