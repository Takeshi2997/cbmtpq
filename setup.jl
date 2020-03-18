module Const

# System Size
const dimS = 8
const dimB = 48

# System Param
const t = 1.0f0
const J = 1.0f0

# Repeat Number
const burnintime = 100
const iters_num = 500
const it_num = 1000
const iϵmax = 20
const num = 2000

# Network Params
const layer = [dimB, 96, 96, dimS]
const layers_num = size(layer)[1] - 1

# Learning Rate
const lr = 0.0001f0

end
