include("./setup.jl")
include("./functions.jl")
include("./ann.jl")
include("./params.jl")

o  = O(zeros(ComplexF64, dimB, dimS), 
       zeros(ComplexF64, dimS))
oe = OE(zeros(ComplexF64, dimB, dimS), 
        zeros(ComplexF64, dimS))

function sampling(network::Network, ϵ::Float64)

    n = rand([1.0, 0.0],  dimB)
    s = rand([1.0, -1.0], dimS)
    energy  = 0.0
    energyS = 0.0
    energyB = 0.0
    numberB = 0.0

    for i in 1:burnintime

        s = updateS(s, n, network)
        n = updateB(n, s, network)
    end

    for i in 1:iters_num
        s     = updateS(s, n, network)
        nnext = updateB(n, s, network)

        eS = energyS_shift(s, n, network)
        eB = energyB_shift(n, s, network)
        e  = eS + eB
        energy    += e
        energyS   += eS
        energyB   += eB
        numberB   += sum(n)
        backward(o, oe, n, s, e)
        n = nnext
    end
    energy     = real(energy)  / iters_num
    energyS    = real(energyS) / iters_num
    energyB    = real(energyB) / iters_num
    numberB   /= iters_num
    error   = (energy - ϵ)^2

    return error, energy, energyS, energyB, numberB
end

function updateparams(network::Network, moment::Moment, e::Float64, ϵ::Float64, lr::Float64)

    Δw = 2.0 * (e - ϵ) * (oe.w - e * o.w) / iters_num
    Δb = 2.0 * (e - ϵ) * (oe.b - e * o.b) / iters_num
    moment.w   = 0.9 * moment.w - lr * Δw
    network.w += moment.w
    moment.b   = 0.9 * moment.b - lr * Δb
    network.b += moment.b        
    setfield!(o,  :w, zeros(ComplexF64, dimB, dimS))
    setfield!(o,  :b, zeros(ComplexF64, dimS))
    setfield!(oe, :w, zeros(ComplexF64, dimB, dimS))
    setfield!(oe, :b, zeros(ComplexF64, dimS))
end


