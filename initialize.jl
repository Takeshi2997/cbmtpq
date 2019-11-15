module Init
    include("./setup.jl")
    include("./functions.jl")
    using .Const, .Func, LinearAlgebra, ComplexValues

    function w()

        return ones(Complex{Float64}, Const.dimB, Const.dimS) * 10.0^(-6)
    end
end
