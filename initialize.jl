module Init
    include("./setup.jl")
    include("./functions.jl")
    include("./ml_core.jl")
    using .Const, .Func, .MLcore, LinearAlgebra, Serialization, ComplexValues

    function network()

        # Make data file
        dirname = "./data"
        rm(dirname, force=true, recursive=true)
        mkdir(dirname)
        filename = dirname * "/param_at_" * lpad(0, 3, "0") * ".dat"

        # Initialize weight, bias
        weight  = ones(Complex{Float64}, Const.dim, Const.dim) * 0.001
        weight += diagm(0 => ones(Complex{Float64}, Const.dim)) .* (im *  π / 4.0)
        biasB   = zeros(Complex{Float32}, Const.dim)
        biasS   = zeros(Complex{Float32}, Const.dim)
   
        # Define network
        network = (weight, biasB, biasS) 

        # Write
        open(io -> serialize(io, network), filename, "w")
    end
end

Init.network()
