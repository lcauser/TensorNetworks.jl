include("src/TensorNetworks.jl")
using .TensorNetworks 

N = 10
s = 0.1
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

E1 = psi * H * psi
E2 = psi * (H * psi)

psiMPS = productMPS(sh, ["dn" for _ = 1:N])
HMPO = MPO(sh, Hlist)
E3 = inner(psiMPS, HMPO, psiMPS)

psitest = GMPS(psi; cutoff=1e-12)
Htest = GMPS(H; cutoff=1e-12)