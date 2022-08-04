using HDF5
include("../../src/TensorNetworks.jl")
#using .TensorNetworks

N = 10
c = 0.5
s = -0.1
D = 2


# Create hamiltonian
sh = spinhalf()
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

# Random guess
psi = randomPEPS(2, N, N, 1)

# Simple update
psi, energy = simpleupdate(psi, 0.1, sh, ops; maxiter=5000, maxdim=1, saveiter=1000, chi=100, cutoff=1e-6)
psi, energy = simpleupdate(psi, 0.01, sh, ops; maxiter=5000, maxdim=D, saveiter=1000, chi=100, cutoff=1e-8)
#psi, energy = simpleupdate(psi, 0.01, sh, ops; maxiter=2000, maxdim=D, saveiter=1000, chi=100, cutoff=1e-10)
#psi, energy = simpleupdate(psi, 0.001, sh, ops; maxiter=2000, maxdim=D, saveiter=1000, chi=100, cutoff=1e-10)
psi, energy = fullupdate(psi, ops, 0.01, sh; maxdim=2, maxiter=1000, miniter=1, chi=4*2^2, saveiter=1, cutoff=1e-6)
psi, energy = fullupdate(psi, ops, 0.001, sh; maxdim=2, maxiter=1000, miniter=1, chi=4*2^2, saveiter=1, cutoff=1e-6)



# Add occupations
global occ = 0
for i = 1:N
    for j = 1:N
        add!(ops, ["n"], [i, j], false, 1)
        global occ = occ + 1
    end
end

# Calculate expectations, and then calculate energy from this
expectations = inner(sh, psi, ops, psi; maxchi=200)

# Calculate scgf
scgf = sum(expectations[1:kin+es])

# Calculate activity
activity = sum(expectations[1:kin])

# Occupations
occs = expectations[kin+es+1:kin+es+occ]

