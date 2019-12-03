module Const

    using Combinatorics

    # System Size
    const copysize = 20
    const systemsize = 2
    const dimS = 40
    const dimB = 40
    sarray = hcat(ones(Float64, 4), -ones(Float64, 4))
    const sset = collect(multiset_permutations(sarray, 2))
    narray = hcat(ones(Float64, 4), zeros(Float64, 4))
    const nset = collect(multiset_permutations(narray, 2))

    # System Param
    const t = 1.0
    const J = 1.0
    const δ = 0.001

    # Repeat Number
    const burnintime = 200
    const iters_num = 500
    const it_num = 1000
    const iϵmax = 20
    const num = 20000

    # Learning Rate
    const lr = 0.00005
end
