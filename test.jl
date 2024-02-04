include("src/TensorNetworks.jl")
using .TensorNetworks
import Base: +, *, /, length
import .TensorNetworks: add!, MPO, OpList
include("src/structures/mps/umps.jl")
include("src/structures/mps/infiniteoplist.jl")
using LinearAlgebra
using KrylovKit

h = 1
J = 2



psi = randomUMPS(2, 10)

sh = spinhalf()
H = InfiniteOpList()
add!(H, "x", 1, h)
add!(H, ["z", "z"], [1, 2], J)

tensor = totensor(sh, H)
mpo = MPO(sh, H)
