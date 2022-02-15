include("src/TensorNetworks.jl")

# Model parameters
N = 10
s = 0.0
c = 0.2 / 1.2

# Create lattice type
sh = spinhalf()

# Create hamiltonian
H = OpList(N)
add!(H, "x", 1, -exp(-s)*sqrt(c*(1-c)))
add!(H, "pu", 1, (1-c))
add!(H, "pd", 1, c)
for i = 1:N-1
    add!(H, ["pu", "x"], [i, i+1], -exp(-s)*sqrt(c*(1-c)))
    add!(H, ["pu", "pu"], [i, i+1], (1-c))
    add!(H, ["pu", "pd"], [i, i+1], c)
end
H = MPO(sh, H)

# Create initial guess
#psi = productMPS(sh, ["s" for i = 1:N])
#psi = randomMPS(2, N, 8)
psi = productMPS(N, [sqrt(c), sqrt(1-c)])
movecenter!(psi, 1)

# Do DMRG
@time psi, energy = dmrg(psi, H; maxsweeps=1000, cutoff=1e-12, maxdim=128, nsites=2)

# Find excited state
psi2 = randomMPS(2, N, 2)
@time psi2, energy2 = dmrg(psi2, H, psi; maxsweeps=10000, cutoff=1e-12, maxdim=128, nsites=2)

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
