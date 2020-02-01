module Init
    include("./setup.jl")
    using .Const, LinearAlgebra, Serialization, ComplexValues

    function network()

        # Make data file
        dirname = "./data"
        rm(dirname, force=true, recursive=true)
        mkdir(dirname)
        filename = dirname * "/param_at_" * lpad(0, 3, "0") * ".dat"

        # Initialize weight, bias
        weight  = rand(Float64, Const.dimH, Const.dimV) .* 
        exp.(2.0 * π * im * rand(Float64, Const.dimH, Const.dimV))
        biasH = zeros(Complex{Float64}, Const.dimH)
        biasV = zeros(Complex{Float64}, Const.dimV)
   
        network = (weight, biasH, biasV)
        # Write
        open(io -> serialize(io, network), filename, "w")
    end
end

Init.network()
