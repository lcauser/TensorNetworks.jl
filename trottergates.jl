include("src/TensorNetworks.jl")

# Model parameters
N = 20
s = -0.1
c = 0.5
tmax = 10000.0
save = 1.0

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

# Large gates
gates = trotterize(sh, oplist, 0.1)
evol1 = productMPO(sh, ["id" for i = 1:N])
applygates!(evol1, gates; cutoff=0, maxdim=128)

# Small gates
gates = trotterize(sh, oplist, 0.001)
evol2 = productMPO(sh, ["id" for i = 1:N])
for i = 1:500
    applygates!(evol2, gates; cutoff=0, maxdim=32)
    println(i)
end
largeerr = trace(evol1, evol2) / trace(evol2, evol2) - 1

errs = []
# Truncate smaller gate
movecenter!(evol2, 1)
for D = [1, 2, 4, 8, 16]
    evol3 = deepcopy(evol2)
    movecenter!(evol3, N; maxdim=D)
    push!(errs, abs(trace(evol3, evol2) / trace(evol2, evol2) - 1))
end


# Build large gate from directly truncated smaller gates
errs2 = []
for D = [1, 2, 4, 8, 16]
    evol3 = productMPO(sh, ["id" for i = 1:N])
    for i = 1:500
        applygates!(evol3, gates; cutoff=0, maxdim=D)
    end
    push!(errs2, abs(trace(evol3, evol2) / trace(evol2, evol2) - 1))
end


Ds = [1, 2, 4, 8, 16]
plot(Ds, [real(errs), real(errs2), real(largeerr)*ones(5)], xaxis=:log10, yaxis=:log,
     label=["After" "Before" "Trotter"], xticks = [1, 2, 4, 8, 16])
