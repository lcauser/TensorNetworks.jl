include("src/TensorNetworks.jl")

# Model parameters
N = 20
s = 0.1
c = 0.5
tmax = 500.0
save = 0.5
dt = 0.001

sh = spinhalf()

oplist = OpList(N)
add!(oplist, "x", 1, exp(-s)*sqrt(c*(1-c)))
add!(oplist, "pu", 1, -(1-c))
add!(oplist, "pd", 1, -c)
for i = 1:N-1
    add!(oplist, ["pu", "x"], [i, i+1], exp(-s)*sqrt(c*(1-c)))
    add!(oplist, ["pu", "pu"], [i, i+1], -(1-c))
    add!(oplist, ["pu", "pd"], [i, i+1], -c)
end

# Small gates
gates = trotterize(sh, oplist, dt)
evol = productMPO(sh, ["id" for i = 1:N])
for i = 1:Int(save / dt)
    applygates!(evol, gates; cutoff=0, maxdim=32)
    println(i)
end
evol2 = deepcopy(evol)
movecenter!(evol, 1)
movecenter!(evol, N; maxdim=4)

# Evolve psi
psi = productMPS(sh, ["s" for i = 1:N])
normal = 0
zs = [0.0]
for i = 1:Int(tmax / save)
    psi = applyMPO(evol, psi; cutoff=1e-12, maxdim=64)
    n = log(norm(psi))
    normal += n
    normalize!(psi)
    push!(zs, 2*normal)
    println(i)
    println(n)
end
