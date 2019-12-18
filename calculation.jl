include("./setup.jl")
include("./ml_core.jl")
using .Const, .MLcore, LinearAlgebra, Serialization

const state = collect(-Const.dimB+1:2:Const.dimB-1)
const ϵ = Const.t * abs.(cos.(π / Const.dimB * state))

function energy(β)

    ϵ = Const.t * abs.(cos.(π / Const.dimB * state))
    return -sum(ϵ .* tanh.(β * ϵ)) / Const.dimB 
end

function f(t)
    
    ϵ = Const.t * abs.(cos.(π / Const.dimB * state))
    return - t * sum(log.(cosh.(ϵ / t)))
end

function df(t)

    ϵ = Const.t * abs.(cos.(π / Const.dimB * state))
    x = sign.(cos.(π / Const.dimB * state) .- μ / Const.t)
    return sum(-log.(cosh.(ϵ / t)) .+ (x .* ϵ / t .* tanh.(ϵ / t)))
end

function s(u, t)

    return (u - f(t)) / t
end

function ds(u, t)

    return -(u - f(t)) / t^2 - df(t) / t
end

function translate(u)

    outputs = 0.0
    t = 5.0
    tm = 0.0
    tv = 0.0
    for n in 1:5000
        dt = ds(u, t)
        lr_t = 0.1 * sqrt(1.0 - 0.999^n) / (1.0 - 0.9^n)
        tm += (1.0 - 0.9) * (dt - tm)
        tv += (1.0 - 0.999) * (dt.^2 - tv)
        t  -= lr_t * tm ./ (sqrt.(tv) .+ 1.0 * 10^(-7))
        outputs = s(u, t)
    end

    return 1 / t
end

function particle_number(β, μ)

    x = cos.(π / Const.dimB * state) .+ μ / Const.t
    ϵ = Const.t * abs.(x)
    return -sum(tanh.(β * ϵ) .* sign.(x)) + Const.dimB / 2.0
end

function dparticle_number(β, μ)

    ϵ = Const.t * abs.(cos.(π / Const.dimB * state) .+ μ / Const.t)
    return -sum(1.0 ./ (cosh.(β * ϵ)).^2)
end


function chemical_potential(β, n)
   
    δ = 1.0
    μ = 0.05
    while δ > 0.00000001
        numberB = particle_number(β, μ)
        dnumber = dparticle_number(β, μ)
        δ = abs(numberB / Const.dimB - n)
        μ -= 0.01 * (numberB - n * Const.dimB) / dnumber
    end
    
    return μ
end

function test2()

    dirname = "./data"
    f = open("energy_expected_value.txt", "w")
    for iβ in 1:1000
        β = iβ * 0.01
   
        ϵ = energy(β, 0.0)

        # Write energy
        write(f, string(β))
        write(f, "\t")
        write(f, string(ϵ))
        write(f, "\t")
        write(f, string(-3.0 * Const.J / 8.0 * sinh(Const.J * β / 2.0) / 
                        (exp(Const.J * β / 2.0) + cosh(Const.J * β / 2.0))))
        write(f, "\n")
    end
    close(f)
end   

function calculate()

    dirname = "./data"
    f = open("energy_data.txt", "w")
    datamatrix = zeros(Float64, Const.iϵmax, 3)
    for iϵ in 1:Const.iϵmax

        filename = dirname * "/param_at_" * lpad(iϵ, 3, "0") * ".dat"
        params = open(deserialize, filename)

        energyS, energyB, numberB = MLcore.forward(params...)

        β = translate(energyB - Const.dimB * Const.t)
        # Write energy
        write(f, string(β))
        write(f, "\t")
        write(f, string(energyB / Const.dimB))
        write(f, "\t")
        write(f, string(energyS / Const.dimS))
        write(f, "\t")
        write(f, string(-3.0 * Const.J / 8.0 * sinh(Const.J * β / 2.0) / 
                        (exp(Const.J * β / 2.0) + cosh(Const.J * β / 2.0)) + 
                       0.125))
        write(f, "\n")


        datamatrix[iϵ, :] = [β energyB/Const.dimB energyS/Const.dimS]
    end
    open(io -> serialize(io, datamatrix), "energy_data.dat", "w")
    close(f)
end

#calculate()
a = chemical_potential(100.0, 0.6)
println(a)
println(particle_number(100.0, a) / Const.dimB)
#test2()


