include("src/TensorNetworks.jl")

N = 10
c = 0.5
s = 2e-2
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

# Initiate spin half
sh = spinhalf()

# Create zero state
states = [["dn" for i = 1:N] for j = 1:N]
zero = productPEPS(sh, states)

# Initial states
sh = spinhalf()
states = [["dn" for i = 1:N] for j = 1:N]
states[1][1] = "up"
states[N][N] = "s"
psi = productPEPS(sh, states)
#psi, energy = fullupdate(psi, H, 0.1, sh; maxdim=1, maxiter=10000, miniter=100, chi=1, saveiter=100)


#psi, energy = fullupdate(psi, H, 0.1, sh; maxdim=2, maxiter=500, miniter=100, chi=1, chieval=16, saveiter=500, update_tol=1e-7)

# Evolve fully
psi, energy = simpleupdate(psi, 0.1, sh, H; maxiter=100000, miniter=100, maxdim=1, chi=50, cutoff=1e-8, saveiter=200)
psi, energy = simpleupdate(psi, 0.1, sh, H; maxiter=100000, miniter=100, maxdim=2, chi=50, cutoff=1e-8, saveiter=200)
psi, energy = simpleupdate(psi, 0.1, sh, H; maxiter=100000, miniter=100, maxdim=3, chi=50, cutoff=1e-8, saveiter=200)
psi, energy = simpleupdate(psi, 0.1, sh, H; maxiter=100000, miniter=100, maxdim=4, chi=50, cutoff=1e-8, saveiter=200)
psi, energy = simpleupdate(psi, 0.01, sh, H; maxiter=100000, miniter=100, maxdim=4, chi=50, cutoff=1e-8, saveiter=200)
psi, energy = fullupdate(psi, H, 0.01, sh; maxdim=4, maxiter=1000, miniter=1, chi=16, saveiter=10)


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

evalenv = Environment(psi, psi; chi=2)
normal = inner(evalenv)
energy = real(sum(inner(sh, evalenv, H) / normal))
