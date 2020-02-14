include("./params.jl")
using LinearAlgebra, Flux
using Flux.Optimise: update!
using BSON: @load
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
    return f, g
end

function load(filename1, filename2)
    
    @load filename1 f
    @load filename2 g
end

function forward(n::Array{Float64, 1})

    return f(n) .+ im * g(n)
end

realloss(s, n) = dot(s, f(n))
imagloss(s, n) = dot(s, g(n))

function backward(o::Array{DiffReal, 1}, oer::Array{DiffComplex, 1}, oei::Array{DiffComplex, 1},
                  n::Array{Float64, 1}, s::Array{Float64, 1}, e::ComplexF64)

    realgs = gradient(() -> realloss(s, n), realps)
    imaggs = gradient(() -> imagloss(s, n), imagps)
    for i in 1:2
        dwr = realgs[f[i].W]
        dbr = realgs[f[i].b]
        dwi = imaggs[g[i].W]
        dbi = imaggs[g[i].b]
        o[i].W   += dwr
        o[i].b   += dbr
        oer[i].W += dwr * e
        oer[i].b += dbr * e
        oei[i].W += dwi * e
        oei[i].b += dbi * e
    end
end

opt = ADAM(Const.lr, (0.9, 0.999))

function update(o::Array{DiffReal, 1}, oer::Array{DiffComplex, 1}, oei::Array{DiffComplex, 1},
                e::Float64, ϵ::Float64)

    for i in 1:2
        ΔWreal = 2.0 * (e - ϵ) * 2.0 * (real.(oer[i].W) - e * o[i].W) / Const.iters_num
        Δbreal = 2.0 * (e - ϵ) * 2.0 * (real.(oer[i].b) - e * o[i].b) / Const.iters_num
        ΔWimag = 2.0 * (e - ϵ) * 2.0 * imag.(oei[i].W) / Const.iters_num
        Δbimag = 2.0 * (e - ϵ) * 2.0 * imag.(oei[i].b) / Const.iters_num
        update!(opt, f[i].W, ΔWreal)
        update!(opt, f[i].b, Δbreal)
        update!(opt, g[i].W, ΔWimag)
        update!(opt, g[i].b, Δbimag)
    end
end
