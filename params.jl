mutable struct Network

    wr1::Array{Float64, 2}
    br1::Array{Float64, 1}
    wi1::Array{Float64, 2}
    bi1::Array{Float64, 1}
    wr2::Array{Float64, 2}
    br2::Array{Float64, 1}
    wi2::Array{Float64, 2}
    bi2::Array{Float64, 1}
end

function Network()

    Network(randn(Float64, layer[2], layer[1]), 
            randn(Float64, layer[2]),
            randn(Float64, layer[3], layer[2]), 
            randn(Float64, layer[3]),
            randn(Float64, layer[2], layer[1]), 
            randn(Float64, layer[2]),
            randn(Float64, layer[3], layer[2]), 
            randn(Float64, layer[3]))
end

mutable struct O

    wr1::Array{Float64, 2}
    br1::Array{Float64, 1}
    wr2::Array{Float64, 2}
    br2::Array{Float64, 1}
end

mutable struct OE

    wr1::Array{Float64, 2}
    br1::Array{Float64, 1}
    wi1::Array{Float64, 2}
    bi1::Array{Float64, 1}
    wr2::Array{Float64, 2}
    br2::Array{Float64, 1}
    wi2::Array{Float64, 2}
    bi2::Array{Float64, 1}
end


