include("src/TensorNetworks.jl")
using .TensorNetworks
using LinearAlgebra
using KrylovKit

h = 1
J = 2



psi = randomUMPS(2, 4)

sh = spinhalf()
H = InfiniteOpList()
add!(H, "x", 1, h)
add!(H, ["z", "z"], [1, 2], J)
add!(H, ["z", "z", "z"], [1, 2, 3], 1)

obs = inner(sh, psi, H)

H = InfiniteOpList()
add!(H, ["id","x","id"], [1, 2, 3], h/3)
add!(H, ["x","id","id"], [1, 2, 3], h/3)
add!(H, ["id","id","x"], [1, 2, 3], h/3)
add!(H, ["z", "z", "id"], [1, 2, 3], J/2)
add!(H, ["id", "z", "z"], [1, 2, 3], J/2)
add!(H, ["z", "z", "z"], [1, 2, 3], 1)

obs = inner(sh, psi, H)

ops = OpList(H)
mpo = MPO(sh, H)
obs2 = inner(psi, mpo)

A1 = contract(psi.Al, psi.C, 3, 1)
A2 = contract(psi.C, psi.Ar, 2, 1)