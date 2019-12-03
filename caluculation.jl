include("./setup.jl")
include("./ml_core.jl")
using .Const, .MLcore, LinearAlgebra, Serialization

state = collect(1:Int64(Const.dim/2)-1)
const ϵ = -2.0 * Const.t * cos.(2.0 * π / Const.dim * state)

function f(t)
    
    return -2.0 * t * sum(log.(1.0 .+ cosh.(ϵ / t))) - t * Const.dim * log(2.0)
end

function df(t)

    return -2.0 * sum(log.(1.0 .+ cosh.(ϵ / t)) .- ϵ / t .* sinh.(ϵ / t) ./ (1 .+ cosh.(ϵ / t))) - 
    Const.dim * log(2.0)
end

function s(u, t)

    return (u - f(t)) / t
end

function ds(u, t)

    return -(u - f(t)) / t^2 - df(t) / t
end

function energy(β)

    return -2.0 * sum(ϵ .* sinh.(ϵ * β) ./ (1.0 .+ cosh.(ϵ * β)))
end

function translate(u)

    outputs = 0.0
    t = 5.0
    tm = 0.0
    tv = 0.0
    for i in 1:10000
        dt = ds(u, t)
        lr_t = 0.1 * sqrt(1.0 - 0.999^i) / (1.0 - 0.9^i)
        tm += (1.0 - 0.9) * (dt - tm)
        tv += (1.0 - 0.999) * (dt.^2 - tv)
        t  -= lr_t * tm ./ (sqrt.(tv) .+ 1.0 * 10^(-7))
        outputs = s(u, t)
    end

    return 1 / t
end

function test()

    dirname = "./data"
    f = open("energy-temperature.txt", "w")
    iϵ = 0
    β = 1.0
    while 1/β >0
        ϵ = - iϵ * 0.1
        β = translate(ϵ)
    
        # Write energy
        write(f, string(β))
        write(f, "\t")
        write(f, string(ϵ / Const.dim))
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
        write(f, string(ϵ / Const.dim))
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

        energy, energyS, energyB = MLcore.forward(params...)
        β = translate(real(energyB))

        # Write energy
        write(f, string(β))
        write(f, "\t")
        write(f, string(real(energyS) / Const.systemsize))
        write(f, "\t")
        write(f, string(real(energyB) / Const.dim))
        write(f, "\t")
        write(f, string(-3.0 * Const.J / 8.0 * sinh(Const.J * β / 2.0) / 
                        (exp(Const.J * β / 2.0) + cosh(Const.J * β / 2.0))))
        write(f, "\n")
    end
    close(f)
end

#calculate()
test2()
