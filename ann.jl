module ANN
    include("./setup.jl")
    using .Const, LinearAlgebra

    σ(x::Float64)   = 1.0 / (1.0 + exp(-x))
    dσ(x::Float64)  = σ(x) * (1.0 - σ(x))

    relu(x::Float64)  = x * (x > 0.0)
    drelu(x::Float64) = 1.0 * (x > 0.0)

    function forward(z)

        return z .+ σ.(real.(z)) .+ im * σ.(imag.(z))
 
    end

    function backward(z)

        return 1.0 .+ dσ.(real.(z)) .+ im * dσ.(imag.(z))
    end
end
