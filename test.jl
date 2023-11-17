include("src/TensorNetworks.jl")
using .TensorNetworks 
using KrylovKit

N = 10
s = 1.0
sh = spinhalf()
psi = productQS(sh, ["dn" for _ = 1:N])

Hlist = OpList(N)
add!(Hlist, "x", 1, -exp(-s))
add!(Hlist, "id", 1)
for i = 1:N-1
    add!(Hlist, ["pu", "x"], [i, i+1], -exp(-s))
    add!(Hlist, ["pu", "id"], [i, i+1])
end
H = QO(sh, Hlist)

gates = trotterize(sh, -1*Hlist, 0.1)

thermal = productQO(sh, ["id" for _ = 1:N])
for i = 1:100
    applygates!(thermal, gates)
    println(trace(H, thermal) / trace(thermal))
end