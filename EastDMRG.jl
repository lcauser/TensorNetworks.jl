include("src/TensorNetworks.jl")

# Model parameters
N = 10
s = 1.0
c = 1/3

# Create lattice type
sh = spinhalf()

# Create hamiltonian
A = -exp(-s)*sqrt(c*(1-c))*op(sh, "x") + c*op(sh, "pd") + (1-c)*op(sh, "pu")
M = zeros((3, 2, 2, 3))
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
M = zeros((3, 2, 2, 3))
M[1, :, :, 1] = op(sh, "id")
M[2, :, :, 1] = A
M[3, :, :, 2] = op(sh, "pu")
M[3, :, :, 3] = op(sh, "id")
K = productMPO(N, M)
M1 = copy(M[3:3, :, :, :])
M1[1, :, :, 1] = A
K[1] = M1

# Create initial guess
psi = productMPS(sh, ["dn" for i = 1:N])
#psi = randomMPS(2, N, 1)
movecenter!(psi, 1)

# Do DMRG
@time psi1, energy1 = dmrg(psi, H)

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
