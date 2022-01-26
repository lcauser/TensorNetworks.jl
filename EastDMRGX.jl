include("src/TensorNetworks.jl")

# Model parameters
N = 20
s = 1.0
c = 0.5

# Create lattice type
sh = spinhalf()

# Create hamiltonian
A = -exp(-s)*sqrt(c*(1-c))*op(sh, "x") + c*op(sh, "pd") + (1-c)*op(sh, "pu")
M = zeros(ComplexF64, (3, 2, 2, 3))
M[1, :, :, 1] = op(sh, "id")
M[2, :, :, 1] = A
M[3, :, :, 2] = op(sh, "pu")
M[3, :, :, 3] = op(sh, "id")
H = productMPO(N, M)
M1 = copy(M[3:3, :, :, :])
M1[1, :, :, 1] = A
H[1] = M1

# Activity operator
A = exp(-s)*sqrt(c*(1-c))*op(sh, "x")
M = zeros(ComplexF64, (3, 2, 2, 3))
M[1, :, :, 1] = op(sh, "id")
M[2, :, :, 1] = A
M[3, :, :, 2] = op(sh, "pu")
M[3, :, :, 3] = op(sh, "id")
K = productMPO(N, M)
M1 = copy(M[3:3, :, :, :])
M1[1, :, :, 1] = A
K[1] = M1

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
