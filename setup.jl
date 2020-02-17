module Const

# System Size
const dimS = 8
const dimB = 64

# System Param
const t = 1.0
const J = 1.0

# Repeat Number
const burnintime = 100
const iters_num = 200
const it_num = 1000
const iϵmax = 20
const num = 20000

# Network Params
const layer = [dimB, 16, 16, dimS]
const layers_num = size(layer)[1] - 1

# Learning Rate
const lr = 0.0001

end
