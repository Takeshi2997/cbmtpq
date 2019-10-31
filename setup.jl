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
    dimS = 2

    # System Param
    ω = 0.2
    J = 0.1
    δ = 1.0 * 10^(-8)

    # Repeat Number
    burnintime = 10
    iters_num = 200
    it_num = 500

    # Learning Rate
    lr = 0.0002
end
