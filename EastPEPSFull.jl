include("src/TensorNetworks.jl")

N = 10
c = 0.5
s = 1e-2
dt = 0.01
maxdim = 3
cutoff = 0

# Create the Hamiltonian
H = OpList2d(N)
for i = 1:N
    for j = 1:N
        if i <= N-1
            if i == 1 && j == 1
                add!(H, ["id", "x"], [i, j], true, sqrt(c*(1-c))*exp(-s))
                add!(H, ["id", "pu"], [i, j], true, -(1-c))
                add!(H, ["id", "pd"], [i, j], true, -c)
            else
                add!(H, ["n", "x"], [i, j], true, sqrt(c*(1-c))*exp(-s))
                add!(H, ["n", "pu"], [i, j], true, -(1-c))
                add!(H, ["n", "pd"], [i, j], true, -c)
            end
        end
        if j <= N-1
            if i == 1 && j == 1
                add!(H, ["id", "x"], [i, j], false, sqrt(c*(1-c))*exp(-s))
                add!(H, ["id", "pu"], [i, j], false, -(1-c))
                add!(H, ["id", "pd"], [i, j], false, -c)
            else
                add!(H, ["n", "x"], [i, j], false, sqrt(c*(1-c))*exp(-s))
                add!(H, ["n", "pu"], [i, j], false, -(1-c))
                add!(H, ["n", "pd"], [i, j], false, -c)
            end
        end
    end
end

# Initial states
sh = spinhalf()
states = [["s" for i = 1:N] for j = 1:N]
states[1][1] = "up"
states[N][N] = "s"
psi = productPEPS(sh, states)
psi, energy = fullupdate(psi, H, 0.1, sh; maxdim=1, maxiter=10000, miniter=100, chi=1, saveiter=100)
psi, energy = fullupdate(psi, H, 0.1, sh; maxdim=2, maxiter=10000, miniter=100, chi=1, saveiter=500, cutoff=1e-4, update_tol=1e-7)

if s > 0
    states = [["dn" for i = 1:N] for j = 1:N]
    states[1][1] = "up"
    states[N][N] = "s"
    psi2 = productPEPS(sh, states)
    psi2, energy2 = fullupdate(psi2, H, 0.1, sh; maxdim=1, maxiter=10000, miniter=100, chi=1, saveiter=100)
    psi2, energy2 = fullupdate(psi2, H, 0.1, sh; maxdim=2, maxiter=10000, miniter=100, chi=1, saveiter=500, cutoff=1e-4, update_tol=1e-7)
    psi = energy2 > energy ? psi2 : psi
end

# Evolve fully
psi, energy = fullupdate(psi, H, 0.01, sh; maxdim=2, maxiter=10000, miniter=1000, chi=1, saveiter=500)
psi, energy = fullupdate(psi, H, 0.01, sh; maxdim=3, maxiter=10000, miniter=1000, chi=1, saveiter=500)
psi, energy = fullupdate(psi, H, 0.01, sh; maxdim=4, maxiter=10000, miniter=1000, chi=1, saveiter=500)
psi, energy = fullupdate(psi, H, 0.001, sh; maxdim=4, maxiter=10000, miniter=200, chi=16, dropoff=4, saveiter=100)

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
