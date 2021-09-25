include("src/TensorNetworks.jl")

N = 6
c = 0.5
s = 0.1
dt = 0.01
maxdim = 3
cutoff = 0

sh = spinhalf()
states = [["dn" for i = 1:N] for j = 1:N]
states[1][1] = "up"
states[N][N] = "s"
psi = productPEPS(sh, states)

H = OpList2d(N)
for i = 1:N
    for j = 1:N
        if i <= N-1
            add!(H, ["n", "x"], [i, j], true, sqrt(c*(1-c))*exp(-s))
            add!(H, ["n", "pu"], [i, j], true, -(1-c))
            add!(H, ["n", "pd"], [i, j], true, -c)
        end
        if j <= N-1
            add!(H, ["n", "x"], [i, j], false, sqrt(c*(1-c))*exp(-s))
            add!(H, ["n", "pu"], [i, j], false, -(1-c))
            add!(H, ["n", "pd"], [i, j], false, -c)
        end
    end
end
add!(H, ["id"], [1, 1], true, 1)
println("---")
env = Environment(psi, psi)

# Create the gate
gate = sqrt(c*(1-c))*exp(-s)*tensorproduct(op(sh, "n"), op(sh, "x"))
gate += -(1-c)*tensorproduct(op(sh, "n"), op(sh, "pu"))
gate += -c*tensorproduct(op(sh, "n"), op(sh, "pd"))
#gate = exp(dt*gate, [2, 4])

ops = Vector(Vector{String}[])
push!(ops, Vector(["n", "x"]))
push!(ops, Vector(["n", "pu"]))
push!(ops, Vector(["n", "pd"]))
coeffs = Vector(Number[sqrt(c*(1-c))*exp(-s), -(1-c), -c])

println("--------")
energies = []
# D = 1
psi, energy = simpleupdate(psi, 0.1, sh, ops, coeffs; maxiter=10000, maxdim=3)
psi0 = deepcopy(psi)
psi, energy = simpleupdate(psi, 0.01, sh, ops, coeffs; maxiter=10000, maxdim=3)
psi, energy = simpleupdate(psi, 0.001, sh, ops, coeffs; maxiter=10000, maxdim=3)
push!(energies, energy)
#psi, energy = fullupdate(psi, gate, 0.01, sh, ops, coeffs; maxdim=1, maxiter=1000, miniter=10, chi=200)
#psi, energy = fullupdate(psi, gate, 0.001, sh, ops, coeffs; maxdim=1, maxiter=1000, miniter=10, chi=200)
#push!(energies, energy)

# D = 2
psi = deepcopy(psi0)
#psi, energy = simpleupdate(psi, 0.1, sh, ops, coeffs; maxiter=10000, maxdim=2)
psi, energy = simpleupdate(psi, 0.01, sh, ops, coeffs; maxiter=10000, maxdim=2)
psi, energy = simpleupdate(psi, 0.001, sh, ops, coeffs; maxiter=10000, maxdim=2)
push!(energies, energy)
#psi, energy = fullupdate(psi, gate, 0.01, sh, ops, coeffs; maxdim=2, maxiter=1000, miniter=10, chi=200)
#psi, energy = fullupdate(psi, gate, 0.001, sh, ops, coeffs; maxdim=2, maxiter=1000, miniter=10, chi=200)
#push!(energies, energy)

# D = 3
psi = deepcopy(psi0)
#psi, energy = simpleupdate(psi, 0.1, sh, ops, coeffs; maxiter=10000, maxdim=3)
psi, energy = simpleupdate(psi, 0.01, sh, ops, coeffs; maxiter=10000, maxdim=3)
psi, energy = simpleupdate(psi, 0.001, sh, ops, coeffs; maxiter=10000, maxdim=3)
push!(energies, energy)
#psi, energy = fullupdate(psi, gate, 0.01, sh, ops, coeffs; maxdim=3, maxiter=1000, miniter=10, chi=200)
#psi, energy = fullupdate(psi, gate, 0.001, sh, ops, coeffs; maxdim=3, maxiter=1000, miniter=10, chi=200)
#push!(energies, energy)

# D = 4
psi = deepcopy(psi0)
#psi, energy = simpleupdate(psi, 0.1, sh, ops, coeffs; maxiter=10000, maxdim=4)
psi, energy = simpleupdate(psi, 0.01, sh, ops, coeffs; maxiter=10000, maxdim=4)
psi, energy = simpleupdate(psi, 0.001, sh, ops, coeffs; maxiter=10000, maxdim=4)
push!(energies, energy)
#psi, energy = fullupdate(psi, gate, 0.01, sh, ops, coeffs; maxdim=4, maxiter=1000, miniter=10, chi=200)
#psi, energy = fullupdate(psi, gate, 0.001, sh, ops, coeffs; maxdim=4, maxiter=1000, miniter=10, chi=200)
#push!(energies, energy)

ns = OpList2d(N)
for i = 1:N
    for j = 1:N
        add!(ns, ["n"], [i, j], false, 1)
    end
end
add!(ns, ["id"], [1, 1], false, 1)
ns = inner(sh, psi, ns, psi; maxchi=200)
#ns = [ns[i] / ns[end] for i = 1:length(ns)-1]
#ns = reshape(ns, (N, N))
