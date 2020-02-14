include("./setup.jl")
using .Const, Flux
using BSON: @save

resσ(x::Float64)  = x + σ(x)

function network()

    # Make data file
    dirname = "./data"
    rm(dirname, force=true, recursive=true)
    mkdir(dirname)
    filenamereal = dirname * "/realparam_at_" * lpad(0, 3, "0") * ".bson"
    filenameimag = dirname * "/imagparam_at_" * lpad(0, 3, "0") * ".bson"

    reallayer1 = Dense(Const.layer[1], Const.layer[2], σ) |> f64
    reallayer2 = Dense(Const.layer[2], Const.layer[3], resσ) |> f64
    f = Chain(reallayer1, reallayer2)
    realps = params(f)

    imaglayer1 = Dense(Const.layer[1], Const.layer[2], σ) |> f64
    imaglayer2 = Dense(Const.layer[2], Const.layer[3], resσ) |> f64
    (wi1, bi1) = params(imaglayer1)
    (wi2, bi2) = params(imaglayer2)
    g = Chain(imaglayer1, imaglayer2)
    imagps = params(g)

    @save filenamereal f
    @save filenameimag g
end

network()
