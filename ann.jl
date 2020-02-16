module ANN
include("./setup.jl")
using .Const, LinearAlgebra, Flux
using Flux.Optimise: update!
using BSON: @save

resσ(x::Float64)  = x + σ(x)

reallayer1 = Dense(Const.layer[1], Const.layer[2], relu) |> f64
reallayer2 = Dense(Const.layer[2], Const.layer[3], resσ) |> f64
f = Chain(reallayer1, reallayer2)
realps = params(f)

imaglayer1 = Dense(Const.layer[1], Const.layer[2], relu) |> f64
imaglayer2 = Dense(Const.layer[2], Const.layer[3], resσ) |> f64
g = Chain(imaglayer1, imaglayer2)
imagps = params(g)

function save(filename1, filename2)

    @save filename1 f
    @save filename2 g
end

function forward(n::Array{Float64, 1})

    return f(n) .+ im * g(n)
end

realloss(s, n) = dot(s, f(n))
imagloss(s, n) = dot(s, g(n))

function setupbackward(n::Array{Float64, 1}, s::Array{Float64, 1})

    realgs = gradient(() -> realloss(s, n), realps)
    imaggs = gradient(() -> imagloss(s, n), imagps)
    return realgs, imaggs
end

function backward(realgs, imaggs, i::Integer)

    return realgs[f[i].W], realgs[f[i].b],
    imaggs[g[i].W], imaggs[g[i].b]
end

opt = ADAM(Const.lr, (0.9, 0.999))

function update(ΔWreal::Array{Float64, 2}, Δbreal::Array{Float64, 1},
                ΔWimag::Array{Float64, 2}, Δbimag::Array{Float64, 1}, i::Integer)

    update!(opt, f[i].W, ΔWreal)
    update!(opt, f[i].b, Δbreal)
    update!(opt, g[i].W, ΔWimag)
    update!(opt, g[i].b, Δbimag)
end

end
