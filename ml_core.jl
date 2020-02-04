module MLcore
    include("./setup.jl")
    include("./functions.jl")
    include("./ann.jl")
    using .Const, .Func, .ANN, LinearAlgebra

    mutable struct Network
    
        weight::Array{Complex{Float64}, 2}
        bias::Array{Complex{Float64}, 1}
    end

    function diff_error(network, ϵ)

        weight = network.weight
        bias  = network.bias

        n = rand([1.0, 0.0], Const.dimB)
        s = rand([1.0, -1.0], Const.dimS)
        energy  = 0.0
        energyS = 0.0
        energyB = 0.0
        numberB = 0.0
        dweight_h = zeros(Complex{Float64}, Const.dimB, Const.dimS)
        dweight   = zeros(Complex{Float64}, Const.dimB, Const.dimS)
        dbias_h   = zeros(Complex{Float64}, Const.dimB)
        dbias     = zeros(Complex{Float64}, Const.dimB)

        for i in 1:Const.burnintime
            s = Func.updateS(s, weight, bias, n)

            z = weight * s .+ bias
            activationS = ANN.forward(z)
            n = Func.updateB(n, activationS)
        end

        for i in 1:Const.iters_num
            s = Func.updateS(s, weight, bias, n)

            z = weight * s .+ bias
            activationS = ANN.forward(z)
            nnext = Func.updateB(n, activationS)

            factor = ANN.backward(z)
            eS = Func.energyS_shift(s, weight, bias, n)
            eB = Func.energyB_shift(n, activationS)
            e  = eS + eB
            energy    += e
            energyS   += eS
            energyB   += eB
            numberB   += sum(n)
            dweight_h += transpose(s) .* n .* factor * e
            dweight   += transpose(s) .* n .* factor 
            dbias_h   += n .* factor * e
            dbias     += n .* factor 

            n = nnext
        end
        energy     = real(energy) / Const.iters_num
        energyS    = real(energyS) / Const.iters_num
        energyB    = real(energyB) / Const.iters_num
        numberB   /= Const.iters_num
        dweight_h /= Const.iters_num
        dweight   /= Const.iters_num
        dbias_h   /= Const.iters_num
        dbias     /= Const.iters_num
        error   = (energy - ϵ)^2

        diff_weight = 2.0 * (energy - ϵ) * (dweight_h - energy * dweight)
        diff_bias   = 2.0 * (energy - ϵ) * (dbias_h - energy * dbias)

        return error, energyS, energyB, numberB, 
        diff_weight, diff_bias
    end

    function forward(weight, bias)

        n = zeros(Float64, Const.dimB)
        s = -ones(Float64, Const.dimB)
        energy  = 0.0
        energyS = 0.0
        energyB = 0.0
        numberB = 0.0

        for i in 1:Const.burnintime
            s = Func.updateS(s, weight, bias, n)

            activationS = weight * s .+ bias
            n = Func.updateB(n, activationS)
        end

        for i in 1:Const.num
            s = Func.updateS(s, weight, bias, n)

            activationS = weight * s .+ bias
            nnext = Func.updateB(n, activationS)

            eS = Func.energyS_shift(s, weight, bias, n)
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
