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
states = ["dn" for i = 1:N]
states[Int(floor(N/2))] = "up"
psi = productMPS(sh, states)
movecenter!(psi, 1)

# Do DMRG
@time psi, energy = dmrgx(psi, H; maxsweeps=100, cutoff=1e-16)
oplist = OpList(N)
for i = 1:N
    add!(oplist, ["pu"], [i])
end
expectations = inner(sh, psi, oplist, psi)
occupations = expectations[1:N]
