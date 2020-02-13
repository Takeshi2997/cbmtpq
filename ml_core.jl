include("./setup.jl")
include("./functions.jl")
include("./ann.jl")
include("./params.jl")

o  = O(zeros(Float64, dimB, dimS), 
       zeros(Float64, dimS),
       zeros(Float64, dimB, dimS), 
       zeros(Float64, dimS))
oe = OE(zeros(ComplexF64, dimB, dimS), 
        zeros(ComplexF64, dimS), 
        zeros(ComplexF64, dimB, dimS), 
        zeros(ComplexF64, dimS), 
        zeros(ComplexF64, dimB, dimS), 
        zeros(ComplexF64, dimS), 
        zeros(ComplexF64, dimB, dimS), 
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
        backward(o, oe, network, n, s, e)
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

    Δwr = 2.0 * (e - ϵ) * 2.0 * (real.(oe.wr) - e * o.wr) / iters_num
    Δbr = 2.0 * (e - ϵ) * 2.0 * (real.(oe.br) - e * o.br) / iters_num
    Δwi = 2.0 * (e - ϵ) * 2.0 * imag.(oe.wr) / iters_num
    Δbi = 2.0 * (e - ϵ) * 2.0 * imag.(oe.br) / iters_num
 
    moment.wr   = 0.9 * moment.wr - lr * Δwr
    network.wr += moment.wr
    moment.br   = 0.9 * moment.br - lr * Δbr
    network.br += moment.br
    moment.wi   = 0.9 * moment.wi - lr * Δwi
    network.wi += moment.wi
    moment.bi   = 0.9 * moment.bi - lr * Δbi
    network.bi += moment.bi
 
    setfield!(o,  :wr, zeros(Float64, dimB, dimS))
    setfield!(o,  :br, zeros(Float64, dimS))
    setfield!(oe, :wr, zeros(ComplexF64, dimB, dimS))
    setfield!(oe, :br, zeros(ComplexF64, dimS))
    setfield!(oe, :wi, zeros(ComplexF64, dimB, dimS))
    setfield!(oe, :bi, zeros(ComplexF64, dimS))
end


