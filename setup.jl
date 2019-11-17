module Const

    using Combinatorics

    # System Size
    const dimB = 80
    const dimS = 24
    sarray = hcat(ones(Float64, 4), -ones(Float64, 4))
    const sset = collect(multiset_permutations(sarray, 2))

    # System Param
    const ω = 2.0
    const J = 1.0
    const δ = 0.1

    # Repeat Number
    const burnintime = 100
    const iters_num = 1000
    const it_num = 1000
    const iϵmax = 20
    const num = 20000

    # Learning Rate
    const lr = 0.005
end
