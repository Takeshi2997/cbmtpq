include("./setup.jl")
include("./ml_core.jl")
using .Const, .MLcore
using LinearAlgebra
using Flux, Serialization
using BSON

const state = collect(-Const.dimB+1:2:Const.dimB-1)

function energy(β::Float32)

    ϵ = Const.t * abs.(cos.(π / Const.dimB * state))
    return -sum(ϵ .* tanh.(β * ϵ)) / Const.dimB 
end

function f(t::Float32)
    
    ϵ = Const.t * abs.(cos.(π / Const.dimB * state))
    return - t * sum(log.(cosh.(ϵ / t)))
end

function df(t::Float32)

    ϵ = Const.t * abs.(cos.(π / Const.dimB * state))
    return sum(-log.(cosh.(ϵ / t)) .+ (ϵ / t .* tanh.(ϵ / t)))
end

function s(u::Float32, t::Float32)

    return (u - f(t)) / t
end

function ds(u::Float32, t::Float32)

    return -(u - f(t)) / t^2 - df(t) / t
end

function translate(u::Float32)

    outputs = 0f0.0
    t = 5.0f0
    tm = 0.0f0
    tv = 0.0f0
    for n in 1:0
        dt = ds(u, t)
        lr_t = 0.1f0 * sqrt(1.0f0 - 0.999f0^n) / (1.0f0 - 0.9f0^n)
        tm += (1.0f0 - 0.9f0) * (dt - tm)
        tv += (1.0f0 - 0.999) * (dt.^2 - tv)
        t  -= lr_t * tm ./ (sqrt.(tv) .+ 1.0f0 * 10^(-7))
        outputs = s(u, t)
    end

    return 1 / t
end

function exact_energy()

    dirname = "./data"
    f = open("exact_energy.txt", "w")
    for iβ in 1:1000
        β = iβ * 0.01f0
   
        ϵ = energy(β)

        # Write energy
        write(f, string(β))
        write(f, "\t")
        write(f, string(ϵ + 1.0f0))
        write(f, "\t")
        write(f, string(-3.0f0 * Const.J / 8.0f0 * sinh(Const.J * β / 2.0f0) / 
                       (exp(Const.J * β / 2.0f0) + cosh(Const.J * β / 2.0f0)) + 0.125f0))
        write(f, "\n")
    end
    close(f)
end   

function calculate()

    dirname = "./data"
    f = open("energy_data.txt", "w")
    datamatrix = zeros(Float32, Const.iϵmax, 3)
    for iϵ in 1:Const.iϵmax

        filenameparams = dirname * "/params_at_" * lpad(iϵ, 3, "0") * ".bson"

        network = BSON.load(filenameparams)
        setfield!(MLcore.Func.ANN.network, :f, network[:f])
        setfield!(MLcore.Func.ANN.network, :g, network[:g])
        energyS, energyB, numberB = MLcore.calculation_energy()

        β = translate(energyB - Const.dimB * Const.t)
        # Write energy
        write(f, string(β))
        write(f, "\t")
        write(f, string(energyB / Const.dimB))
        write(f, "\t")
        write(f, string(energyS / Const.dimS))
        write(f, "\t")
        write(f, string(-3.0f0 * Const.J / 8.0f0 * sinh(Const.J * β / 2.0f0) / 
                       (exp(Const.J * β / 2.0f0) + cosh(Const.J * β / 2.0f0)) + 
                     0.125f0))
        write(f, "\t")
        write(f, string(numberB / Const.dimB))
        write(f, "\n")


        datamatrix[iϵ, :] = [β energyB/Const.dimB energyS/Const.dimS]
    end
    open(io -> serialize(io, datamatrix), "energy_data.dat", "w")
    close(f)
end

calculate()
exact_energy()


