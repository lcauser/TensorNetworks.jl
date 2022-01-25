include("src/TensorNetworks.jl")

N = 6
c = 0.5
s = 0.1
dt = 0.01
maxdim = 3
cutoff = 0

# Create the Hamiltonian
H = OpList2d(N)
for i = 1:N
    for j = 1:N
        if i <= N-1
            add!(H, ["n", "x"], [i, j], true, sqrt(c*(1-c))*exp(-s))
            add!(H, ["n", "pu"], [i, j], true, -(1-c))
            add!(H, ["n", "pd"], [i, j], true, -c)
            add!(H, ["x", "n"], [i, j], true, sqrt(c*(1-c))*exp(-s))
            add!(H, ["pu", "n"], [i, j], true, -(1-c))
            add!(H, ["pd", "n"], [i, j], true, -c)
        end
        if j <= N-1
            add!(H, ["n", "x"], [i, j], false, sqrt(c*(1-c))*exp(-s))
            add!(H, ["n", "pu"], [i, j], false, -(1-c))
            add!(H, ["n", "pd"], [i, j], false, -c)
            add!(H, ["x", "n"], [i, j], false, sqrt(c*(1-c))*exp(-s))
            add!(H, ["pu", "n"], [i, j], false, -(1-c))
            add!(H, ["pd", "n"], [i, j], false, -c)
        end
    end
end

# Initiate spin half
sh = spinhalf()

# Create zero state
states = [["dn" for i = 1:N] for j = 1:N]
zero = productPEPS(sh, states)

@time begin
# Initial states
states = [["s" for i = 1:N] for j = 1:N]
psi = productPEPS(sh, states)
psi, energy = fullupdate(psi, H, 0.1, sh; maxdim=1, maxiter=10000, miniter=100, chi=1, saveiter=100)
psi, energy = fullupdate(psi, H, 0.1, sh; maxdim=2, maxiter=200, miniter=100, chi=1, chieval=16, saveiter=100, update_tol=1e-7)
if s > 0
    states = [["dn" for i = 1:N] for j = 1:N]
    states[1][1] = "up"
    psi2 = productPEPS(sh, states)
    psi2, energy2 = fullupdate(psi2, H, 0.1, sh; maxdim=1, maxiter=10000, miniter=100, chi=1, saveiter=100)
    psi2, energy2 = fullupdate(psi2, H, 0.1, sh; maxdim=2, maxiter=200, miniter=100, chi=1, chieval=16, saveiter=10, update_tol=1e-7)
    psi = energy2 > energy ? psi2 : psi
end

# Evolve fully
psi, energy = fullupdate(psi, H, 0.01, sh, [zero]; maxdim=2, maxiter=10000, miniter=200, chi=1, saveiter=100)
psi, energy = fullupdate(psi, H, 0.01, sh, [zero]; maxdim=3, maxiter=10000, miniter=200, chi=1, saveiter=100)
psi, energy = fullupdate(psi, H, 0.01, sh, [zero]; maxdim=4, maxiter=10000, miniter=200, chi=1, saveiter=100)
psi, energy = fullupdate(psi, H, 0.001, sh, [zero]; maxdim=4, maxiter=1000, miniter=200, chi=16, saveiter=100)
end


# Measure occupations
ns = OpList2d(N)
for i = 1:N
    for j = 1:N
        add!(ns, ["n"], [i, j], false, 1)
    end
end
add!(ns, ["id"], [1, 1], false, 1)
ns = inner(sh, psi, ns, psi; maxchi=200)
ns = [ns[i] / ns[end] for i = 1:length(ns)-1]
ns = reshape(ns, (N, N))
