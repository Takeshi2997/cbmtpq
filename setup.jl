module Const

    struct Param

        # System Size
        dimB::Int64
        dimS::Int64

        # System Param
        ω::Float32
        J::Float32
        δ::Float32

        # Repeat Number
        burnintime::Int64
        iters_num::Int64
        it_num::Int64
        iϵmax::Int64
        num::Int64

        # Learning Rate
        lr::Float32
    end

    # System Size
    dimB = 20
    dimS = 10

    # System Param
    ω = 0.2
    J = 0.1
    δ = 1.0 * 10^(-6)

    # Repeat Number
    burnintime = 10
    iters_num = 200
    it_num = 2000
    iϵmax = 5
    num = 20000

    # Learning Rate
    lr = 0.00005
end
