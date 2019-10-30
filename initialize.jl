module Init
    include("./setup.jl")
    include("./functions.jl")
    using .Const, .Func, LinearAlgebra, ComplexValues

    function weight(i, j)

        return ones(Complex{Float32}, i, j) * 10^(-8)
    end

    function bias(i)

        return zeros(Complex{Float32}, i)
    end
end
