include("src/TensorNetworks.jl")

# Model parameters
N = 100
s = -1.0
c = 0.5

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
H = MPO(H, sh)

# Create initial guess
psi = productMPS(sh, ["dn" for i = 1:N])
movecenter!(psi, 1)

# Do DMRG
@time psi1, energy1 = dmrg(psi, H; maxsweeps=100, cutoff=1e-16)

# Find excited state
psi2 = randomMPS(2, N, 1)
@time psi1, energy2 = dmrg(psi2, H, psi1; maxsweeps=100, cutoff=1e-16)

# Measure Occupations and Correlations
oplist = OpList(N)
for i = 1:N
    add!(oplist, ["pu"], [i])
end
for i = 1:N-1
    add!(oplist, ["pu", "pu"], [i, i+1])
end
expectations = inner(sh, psi1, oplist, psi1)
occupations = expectations[1:N]
correlations = expectations[N+1:end]
