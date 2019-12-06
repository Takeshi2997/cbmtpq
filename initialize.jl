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
        weight  = zeros(Complex{Float64}, Const.dimB, Const.dimS)
        weight[1:Const.dimS, 1:Const.dimS] += 
        diagm(0 => ones(Complex{Float64}, Const.dimS)) .* (im *  π / 4.0)
        biasB   = zeros(Complex{Float32}, Const.dimB)
        biasS   = zeros(Complex{Float32}, Const.dimS)
   
        # Define network
        network = (weight, biasB, biasS) 

        # Write
        open(io -> serialize(io, network), filename, "w")
    end
end

Init.network()
