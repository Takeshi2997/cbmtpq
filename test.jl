include("./setup.jl")
include("./ml_core.jl")
include("./initialize.jl")
include("./functions.jl")
using .Const, .MLcore, .Init, .Func, LinearAlgebra, Serialization

function f(boolianval)
    if boolianval == true
        return 1.0
    else
        return 0.0
    end
end
    
v = [1 2 3 4 5]
println([v .> 2])
println(convert.(Array{Float32}, [v .> 2]))
