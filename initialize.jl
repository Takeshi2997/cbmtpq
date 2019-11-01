module Init
    include("./setup.jl")
    include("./functions.jl")
    using .Const, .Func, LinearAlgebra, ComplexValues

    function weight(i, j)

        return ones(Complex{Float32}, i, j) * 10^(-3)
    end
end
