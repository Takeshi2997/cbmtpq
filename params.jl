using LinearAlgebra

mutable struct Network

    w::Array{Complex{Float64}, 2}
    b::Array{Complex{Float64}, 1}
end

mutable struct Moment

    w::Array{Complex{Float64}, 2}
    b::Array{Complex{Float64}, 1}
end
