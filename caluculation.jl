include("./setup.jl")
include("./ml_core.jl")
using .Const, .MLcore, LinearAlgebra, Serialization

state = collect(1:Int64(Const.dimB/2)-1)
const ϵ = -2.0 * Const.t * cos.(2.0 * π / Const.dimB * state)

function f(t, μ)
    
    return -2.0 * t * sum(log.(1.0 .+ cosh.(ϵ / t))) - t * Const.dimB * log(2.0) -
    t * log(cosh(μ * Const.dimB / 2.0 / t)) - μ * Const.dimB / 2.0
end

function df(t, μ)

    return -2.0 * sum(log.(1.0 .+ cosh.(ϵ / t)) .- ϵ / t .* sinh.(ϵ / t) ./ (1 .+ cosh.(ϵ / t))) - 
    log(cosh(μ * Const.dimB / 2.0 / t)) + μ * Const.dimB / 2.0 / t * tanh(μ * Const.dimB / 2.0 / t) -
    Const.dimB * log(2.0)
end

function chemical_potential(n, t)

    return 2.0 * t / Const.dimB * atanh((2.0 * n - Const.dimB) / Const.dimB)
end

function s(u, n, t, μ)

    return (u - μ * n - f(t, μ)) / t
end

function ds(u, n, t, μ)

    return -(u - n * μ - f(t, μ)) / t^2 - df(t, μ) / t
end

function energy(β)

    return 2.0 * sum(ϵ .* sinh.(ϵ * β) ./ (1.0 .+ cosh.(ϵ * β)))
end

function translate(u, n)

    outputs = 0.0
    t = 5.0
    tm = 0.0
    tv = 0.0
    for i in 1:10000
        μ = chemical_potential(n, t)
        dt = ds(u, n, t, μ)
        lr_t = 0.1 * sqrt(1.0 - 0.999^n) / (1.0 - 0.9^n)
        tm += (1.0 - 0.9) * (dt - tm)
        tv += (1.0 - 0.999) * (dt.^2 - tv)
        t  -= lr_t * tm ./ (sqrt.(tv) .+ 1.0 * 10^(-7))
        outputs = s(u, n, t, μ)
    end

    return 1 / t
end

function test()

    dirname = "./data"
    f = open("energy-temperature.txt", "w")
    n = 0.5 * Const.dimB
    iϵ = 0
    β = 1.0
    while 1/β >0
        ϵ = - iϵ * 0.1
        β = translate(ϵ, n)
    
        # Write energy
        write(f, string(β))
        write(f, "\t")
        write(f, string(ϵ / Const.dimB))
        write(f, "\t")
        write(f, string(-3.0 * Const.J / 8.0 * sinh(Const.J * β / 2.0) / 
                        (exp(Const.J * β / 2.0) + cosh(Const.J * β / 2.0))))
        write(f, "\n")
        iϵ += 1
    end
    close(f)
end   

function test2()

    dirname = "./data"
    f = open("energy_expected_value.txt", "w")
    for iβ in 1:1000
        β = iβ * 0.01
   
        ϵ = energy(β)
        # Write energy
        write(f, string(β))
        write(f, "\t")
        write(f, string(ϵ / Const.dimB))
        write(f, "\n")
    end
    close(f)
end   

function calculate()

    dirname = "./data"
    f = open("energy_data.txt", "w")
    for iϵ in 1:1 # Const.iϵmax

        filename = dirname * "/param_at_" * lpad(iϵ, 3, "0") * ".dat"
        params = open(deserialize, filename)

        energy, energyS, energyB, numberB = MLcore.forward(params...)
        β = translate(real(energyB), numberB)

        # Write energy
        write(f, string(β))
        write(f, "\t")
        write(f, string(real(energyS) / Const.systemsize))
        write(f, "\t")
        write(f, string(real(energyB) / Const.dimB))
        write(f, "\t")
        write(f, string(-3.0 * Const.J / 8.0 * sinh(Const.J * β / 2.0) / 
                        (exp(Const.J * β / 2.0) + cosh(Const.J * β / 2.0))))
        write(f, "\n")
    end
    close(f)
end

#calculate()
test2()
