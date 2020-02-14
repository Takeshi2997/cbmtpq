mutable struct DiffReal

    W::Array{Float64, 2}
    b::Array{Float64, 1}
end

mutable struct DiffComplex

    W::Array{ComplexF64, 2}
    b::Array{ComplexF64, 1}
end

