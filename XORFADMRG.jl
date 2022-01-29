include("src/TensorNetworks.jl")

# Model parameters
N = 10
s = 1.0
c = 0.5

# Create lattice type
sh = spinhalf()

# Create hamiltonian
H = OpList(N)
add!(H, "x", 1, -0.25*exp(-s)*sqrt(c*(1-c)))
add!(H, "pu", 1, 0.25*(1-c))
add!(H, "pd", 1, 0.25*c)
for i = 1:N-2
    add!(H, ["pu", "x", "pd"], [i, i+1, i+2], -exp(-s)*sqrt(c*(1-c)))
    add!(H, ["pu", "pu", "pd"], [i, i+1, i+2], (1-c))
    add!(H, ["pu", "pd", "pd"], [i, i+1, i+2], c)
    add!(H, ["pd", "x", "pu"], [i, i+1, i+2], -exp(-s)*sqrt(c*(1-c)))
    add!(H, ["pd", "pu", "pu"], [i, i+1, i+2], (1-c))
    add!(H, ["pd", "pd", "pu"], [i, i+1, i+2], c)
end
add!(H, "x", N, -0.25*exp(-s)*sqrt(c*(1-c)))
add!(H, "pu", N, 0.25*(1-c))
add!(H, "pd", N, 0.25*c)


println("----")
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
