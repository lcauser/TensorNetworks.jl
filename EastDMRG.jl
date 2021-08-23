include("src/TensorNetworks.jl")

# Model parameters
N = 400
s = -1.0
c = 0.5

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

# Create initial guess
psi = productMPS(sh, ["s" for i = 1:N])
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
