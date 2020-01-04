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
        weight  = -ones(Complex{Float64}, Const.dimB, Const.dimS) * 0.0001
        for n in 1:Int64(Const.dimB/Const.dimS)-1
            weight[Const.dimS*n+1:(n+1)*Const.dimS, :] += 
            diagm(0 => ones(Complex{Float64}, Const.dimS)) * (im * π / 2.0)
        end
        biasB = zeros(Complex{Float64}, Const.dimB)
        biasS = zeros(Complex{Float64}, Const.dimS)
   
        network = (weight, biasB, biasS, 0.0)
        # Write
        open(io -> serialize(io, network), filename, "w")
    end
end

Init.network()
