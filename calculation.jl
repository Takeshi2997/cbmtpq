include("./setup.jl")
include("./ml_core.jl")
using .Const, .MLcore, LinearAlgebra, Serialization

state = collect(1:Int64(Const.dimB/2)-1)
const ϵ = 4.0 * Const.t * cos.(2.0 * π / Const.dimB * state)

function f(t)
    
    return - t * sum(log.(2.0 .+ 2.0 * cosh.(ϵ / 2.0 / t)))
end

function df(t)

    return sum(-log.(2.0 .+ 2.0 * cosh.(ϵ / 2.0 / t)) .+
    (ϵ / 2.0 / t .* sinh.(ϵ / 2.0 / t)) ./ (1.0 .+ cosh.(ϵ / 2.0 / t)))
end

function s(u, t)

    return (u - f(t)) / t
end

function ds(u, t)

    return -(u - f(t)) / t^2 - df(t) / t
end

function energy(β)

    return -sum((ϵ .* sinh.(ϵ * β))./ (1.0 .+ cosh.(ϵ * β)))
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

function test()

    dirname = "./data"
    f = open("energy-temperature.txt", "w")
    iϵ = 0
    β = 1.0
    while 1/β >0
        ϵ = -iϵ * 0.01
        β = translate(ϵ)
    
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

        energy, energyS, energyB = MLcore.forward(params...)
        β = translate(real(energyB))

        # Write energy
        write(f, string(β))
        write(f, "\t")
        write(f, string(real(energyS) / Const.dimS))
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
#test2()
test()