module Const

    using Combinatorics

    # System Size
    const dimS = 16
    const dimB = 80
    sarray = hcat(ones(Float64, 4), -ones(Float64, 4))
    const sset = collect(multiset_permutations(sarray, 2))

    # System Param
    const ω = 2.0
    const J = 1.0
    const δ = 0.01

    # Repeat Number
    const burnintime = 200
    const iters_num = 500
    const it_num = 1000
    const iϵmax = 50
    const num = 20000

    # Learning Rate
    const lr = 0.00005
    const lr_repeat = 0.00001
end
