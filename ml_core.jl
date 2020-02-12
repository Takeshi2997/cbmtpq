module MLcore
    include("./setup.jl")
    include("./functions.jl")
    include("./ann.jl")
    using .Const, .Func, .ANN, LinearAlgebra

    mutable struct O

        w::Array{Complex{Float64}, 2}
        b::Array{Complex{Float64}, 1}
    end

    mutable struct OE

        w::Array{Complex{Float64}, 2}
        b::Array{Complex{Float64}, 1}
    end

    o  = O(zeros(ComplexF64, Const.dimB, Const.dimS), 
           zeros(ComplexF64, Const.dimB))
    oe = OE(zeros(ComplexF64, Const.dimB, Const.dimS), 
            zeros(ComplexF64, Const.dimB))

    function sampling(network, ϵ)

        n = rand([1.0, 0.0], Const.dimB)
        s = rand([1.0, -1.0], Const.dimS)
        energy  = 0.0
        energyS = 0.0
        energyB = 0.0
        numberB = 0.0

        for i in 1:Const.burnintime

            s = Func.updateS(s, n, network)
            n = Func.updateB(n, s, network)
        end

        for i in 1:Const.iters_num
            s = Func.updateS(s, n, network)
            nnext = Func.updateB(n, s, network)
 
            eS = Func.energyS_shift(s, n, network)
            eB = Func.energyB_shift(n, s, network)
            e  = eS + eB
            energy    += e
            energyS   += eS
            energyB   += eB
            numberB   += sum(n)
            ANN.backward(o, oe, n, s, e)
            n = nnext
        end
        energy     = real(energy) / Const.iters_num
        energyS    = real(energyS) / Const.iters_num
        energyB    = real(energyB) / Const.iters_num
        numberB   /= Const.iters_num
        error   = (energy - ϵ)^2

        return error, energy, energyS, energyB, numberB
    end

    function updateparam(network, moment, e, ϵ, lr)

        Δw = 2.0 * (e - ϵ) * (oe.w - e * o.w) / Const.iters_num
        Δb = 2.0 * (e - ϵ) * (oe.b - e * o.b) / Const.iters_num
        moment.w   = 0.9 * moment.w - lr * Δw
        network.w += moment.w
        moment.b   = 0.9 * moment.b - lr * Δb
        network.b += moment.b        
        setfield!(o,  :w, zeros(ComplexF64, Const.dimB, Const.dimS))
        setfield!(o,  :b, zeros(ComplexF64, Const.dimB))
        setfield!(oe, :w, zeros(ComplexF64, Const.dimB, Const.dimS))
        setfield!(oe, :b, zeros(ComplexF64, Const.dimB))
    end

    function forward(weight, biasB, biasS)

        n = zeros(Float64, Const.dimB)
        s = -ones(Float64, Const.dimB)
        energy  = 0.0
        energyS = 0.0
        energyB = 0.0
        numberB = 0.0

        for i in 1:Const.burnintime
            activationB = transpose(weight) * n .+ biasS
            realactivationB = 2.0 * real.(activationB)
            s = Func.updateS(s, realactivationB)

            activationS = weight * s .+ biasB
            realactivationS = 2.0 * real.(activationS)
            n = Func.updateB(n, realactivationS)
        end

        for i in 1:Const.num
            activationB = transpose(weight) * n .+ biasS
            realactivationB = 2.0 * real.(activationB)
            s = Func.updateS(s, realactivationB)

            activationS = weight * s .+ biasB
            realactivationS = 2.0 * real.(activationS)
            nnext = Func.updateB(n, realactivationS)

            eS = Func.energyS_shift(s, activationB)
            eB = Func.energyB_shift(n, activationS)
            e  = eS + eB
            energy    += e
            energyS   += eS
            energyB   += eB
            numberB   += sum(n)

            n = nnext
        end
        energy   = real(energy) / Const.num
        energyS  = real(energyS) / Const.num
        energyB  = real(energyB) / Const.num
        numberB /= Const.num

        return energyS, energyB, numberB
   end
end
