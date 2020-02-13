module Init
    include("./setup.jl")
    using LinearAlgebra, Serialization

    function network()

        # Make data file
        dirname = "./data"
        rm(dirname, force=true, recursive=true)
        mkdir(dirname)
        filename = dirname * "/param_at_" * lpad(0, 3, "0") * ".dat"

        # Initialize weight, bias
        weight  = -ones(Complex{Float64}, dimB, dimS) * 0.0001
        for n in 1:Int64(dimB/dimS)-1
            weight[dimS*n+1:(n+1)*dimS, :] += 
            diagm(0 => ones(Complex{Float64}, dimS)) * (im * π / 2.0)
        end
        bias = zeros(Complex{Float64}, dimB)
   
        network = (weight, bias)
        # Write
        open(io -> serialize(io, network), filename, "w")
    end
end

Init.network()
