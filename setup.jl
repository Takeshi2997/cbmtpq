module Const

# System Size
const dimS = 8
const dimB = 64

# System Param
const t = 1.0f0
const J = 1.0f0

# Repeat Number
const burnintime = 100
const iters_num = 200
const it_num = 10000
const iϵmax = 20
const num = 20000

# Network Params
const layer = [dimB, 128, 128, 128, dimS]
const layers_num = size(layer)[1] - 1

# Learning Rate
const lr = 0.002f0

end
