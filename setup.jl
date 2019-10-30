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

        # Learning Rate
        lr::Float32
        η::Float32

    end

    # System Size
    dimB = 50
    dimS = 2 * 10

    # System Param
    ω = 0.1 / dimB
    J = 0.02 / dimS
    δ = 0.01 / dimB / dimS

    # Repeat Number
    burnintime = 100
    iters_num = 1000
    it_num = 2000

    # Learning Rate
    lr = 0.001
end
