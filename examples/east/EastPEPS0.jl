include("src/TensorNetworks.jl")

# Parameters
N = 6
c = 0.5
s = 0.1

# Load spinhalf
sh = spinhalf()

# Create Hamiltonian
ops = OpList2d(N)
# Add kinetic terms
global kin = 0
for i = 1:N
    for j = 1:N
        p = (i == 1 && j == 1) ? "id" : "n"
        if i <= N-1
            if i == 1 && j == 1
                add!(ops, ["x"], [2, 1], true, sqrt(c*(1-c))*exp(-s))
            else
                add!(ops, [p, "x"], [i, j], true, sqrt(c*(1-c))*exp(-s))
            end
            global kin = kin + 1
        end
        if j <= N-1
            if i == 1 && j == 1
                add!(ops, ["x"], [1, 2], false, sqrt(c*(1-c))*exp(-s))
            else
                add!(ops, [p, "x"], [i, j], false, sqrt(c*(1-c))*exp(-s))
            end
            global kin = kin + 1
        end
    end
end

# Add escape rate terms
global es = 0
for i = 1:N
    for j = 1:N
        p = (i == 1 && j == 1) ? "id" : "n"
        if i <= N-1
            add!(ops, [p, "pu"], [i, j], true, -(1-c))
            add!(ops, [p, "pd"], [i, j], true, -c)
            global es = es + 2
        end
        if j <= N-1
            add!(ops, [p, "pu"], [i, j], false, -(1-c))
            add!(ops, [p, "pd"], [i, j], false, -c)
            global es = es + 2
        end
    end
end

# Create inital guess
spins = [["dn" for i = 1:N] for i = 1:N]
spins[1][1] = "up"
#psi = randomPEPS(2, N, 1)
psi = productPEPS(sh, spins)

# Apply simple update
println("----------------")
psi, energy = simpleupdate(psi, 0.1, sh, ops; maxiter=100000, miniter=2000, maxdim=1, chi=50, cutoff=1e-8)
psi, energy = simpleupdate(psi, 0.1, sh, ops; maxiter=100000, miniter=2000, maxdim=2, chi=50, cutoff=1e-8)
psi, energy = simpleupdate(psi, 0.1, sh, ops; maxiter=100000, miniter=2000, maxdim=3, chi=50, cutoff=1e-8)
psi, energy = simpleupdate(psi, 0.1, sh, ops; maxiter=100000, miniter=2000, maxdim=4, chi=50, cutoff=1e-8)
psi, energy = simpleupdate(psi, 0.01, sh, ops; maxiter=100000, miniter=2000, maxdim=4, chi=50, cutoff=1e-8)
psi, energy = simpleupdate(psi, 0.001, sh, ops; maxiter=100000, miniter=2000, maxdim=4, chi=50, cutoff=1e-8)

# Calculate energy with better environment
energy = inner(sh, psi, ops, psi; maxchi=200)
