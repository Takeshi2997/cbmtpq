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
    dimB = 64
    dimS = 8

    # System Param
    ω = 0.2
    J = 0.1
    δ = 0.01

    # Repeat Number
    burnintime = 100
    iters_num = 1000
    it_num = 1200
    iϵmax = 20
    num = 20000

    # Learning Rate
    lr = 0.01
end
