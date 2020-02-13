include("./setup.jl")
using LinearAlgebra

function backward(o, oe, n, s, e)

    dw = transpose(s) .* n
    db = n
    o.w  .+= dw
    o.b  .+= db
    oe.w .+= dw * e
    oe.b .+= db * e
end

