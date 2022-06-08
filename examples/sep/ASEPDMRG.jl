#using .TensorNetworks

# Model parameters
N = 20
p = 0.7

# Create lattice type
sh = spinhalf()

# Create hamiltonian
H = OpList(N)
H = MPO(sh, H) # Convert to MPO

# Create initial guess
psi = randomMPS(2, N, 1)
movecenter!(psi, 1)

# Do DMRG
psi, energy = dmrg(psi, H; maxsweeps=1000, cutoff=1e-12, maxdim=16, nsites=2)


# Measure Occupations and Correlations
oplist = OpList(N)
for i = 1:N
    add!(oplist, ["pu"], [i])
end
for i = 1:N-1
    add!(oplist, ["pu", "pu"], [i, i+1])
end
expectations = inner(sh, psi, oplist, psi)
occupations = expectations[1:N]
correlations = expectations[N+1:end]
