module ANN
    include("./setup.jl")
    using .Const, LinearAlgebra

    function backward(o, oe, n, s, e)

        dw = transpose(s) .* n
        db = n
        o.w  .+= dw
        o.b  .+= db
        oe.w .+= dw * e
        oe.b .+= db * e
    end

    function updateparam(network, moment, e, ϵ, lr)

        Δw = 2.0 * (e - ϵ) * (oe.w - e * o.w) / Const.iters_num
        Δb = 2.0 * (e - ϵ) * (oe.b - e * o.b) / Const.iters_num
        moment.w   = 0.9 * moment.w - lr * Δw
        network.w += moment.w
        moment.b   = 0.9 * moment.b - lr * Δb
        network.b += moment.b        
    end
end
