module Const

    using Combinatorics
    struct Param

        # System Size
        dimB::Int64
        dimS::Int64
        sset::Array{Int64}

        # System Param
        ω::Float64
        J::Float64
        δ::Float64

        # Repeat Number
        burnintime::Int64
        iters_num::Int64
        it_num::Int64
        iϵmax::Int64
        num::Int64

        # Learning Rate
        lr::Float64
    end

    # System Size
    const dimB = 80
    const dimS = 2
    sarray = hcat(ones(Float64, 2^Const.dimS), -ones(Float64, 2^Const.dimS))
    const sset = collect(multiset_permutations(sarray, dimS))

    # System Param
    const ω = 2.0
    const J = 1.0
    const δ = 0.001

    # Repeat Number
    const burnintime = 50
    const iters_num = 200
    const it_num = 1000
    const iϵmax = 20
    const num = 2000

    # Learning Rate
    const lr = 0.0005
end
