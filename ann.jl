module ANN
include("./setup.jl")
using .Const, LinearAlgebra, Flux, CuArrays
using Flux.Optimise: update!
using BSON: @save, @load

reallayer1 = Dense(Const.layer[1], Const.layer[2], relu) |> gpu
reallayer2 = Dense(Const.layer[2], Const.layer[3], relu) |> gpu
reallayer3 = Dense(Const.layer[3], Const.layer[4], relu) |> gpu
reallayer4 = Dense(Const.layer[4], Const.layer[5]) |> gpu
f = Chain(reallayer1, reallayer2, reallayer3, reallayer4)
realps = params(f)

imaglayer1 = Dense(Const.layer[1], Const.layer[2], relu) |> gpu
imaglayer2 = Dense(Const.layer[2], Const.layer[3], relu) |> gpu
imaglayer3 = Dense(Const.layer[3], Const.layer[4], relu) |> gpu
imaglayer4 = Dense(Const.layer[4], Const.layer[5]) |> gpu
g = Chain(imaglayer1, imaglayer2, imaglayer3, imaglayer4)
imagps = params(g)

function save(filename1::String, filename2::String)

    @save filename1 f
    @save filename2 g
end

function load(filename1::String, filename2::String)

    @load filename1 f
    @load filename2 g
end

function forward(n::CuArray{Float32, 1})

    return f(n) .+ im * g(n)
end

realloss(s::CuArray{Float32, 1}, n::CuArray{Float32, 1}) = dot(s, f(n))
imagloss(s::CuArray{Float32, 1}, n::CuArray{Float32, 1}) = dot(s, g(n))

function setupbackward(n::CuArray{Float32, 1}, s::CuArray{Float32, 1})

    realgs = gradient(() -> realloss(s, n), realps)
    imaggs = gradient(() -> imagloss(s, n), imagps)
    return realgs, imaggs
end

function backward(realgs, imaggs, i::Integer)

    return realgs[f[i].W], realgs[f[i].b],
    imaggs[g[i].W], imaggs[g[i].b]
end

opt = ADAM(Const.lr, (0.9, 0.999))

function update(ΔWreal::CuArray{Float32, 2}, Δbreal::CuArray{Float32, 1},
                ΔWimag::CuArray{Float32, 2}, Δbimag::CuArray{Float32, 1}, i::Integer)

    update!(opt, f[i].W, ΔWreal)
    update!(opt, f[i].b, Δbreal)
    update!(opt, g[i].W, ΔWimag)
    update!(opt, g[i].b, Δbimag)
end

end
