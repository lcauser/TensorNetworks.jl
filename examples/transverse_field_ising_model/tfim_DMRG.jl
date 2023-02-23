using TensorNetworks
include("tfimUtils.jl")

# Model parameters
N = 100
h = 1.0
J = 1.0

# Create lattice type
sh = spinhalf()

# Create hamiltonian
H = TFIMHamiltonian(N, h, J) # Create op list
H = MPO(sh, H) # Convert to MPO

# Create initial guess
psi = randomMPS(2, N, 1) # 2 is the physical dimension, 1 is the bond dimension
movecenter!(psi, 1) # Moves the canonical center to site 1

# Do DMRG; nsites (default 2) is the number of tensors in each update, 
# cutoff is the acceptable SVD, maxdim is the maximum bond dimension
psi, energy = dmrg(psi, H; nsites=2, cutoff=1e-12, maxdim=32, maxsweeps=100)


# Measure magnetizations
oplist = OpList(N)
for i = 1:N
    add!(oplist, "z", i)
end

for i = 1:N
    add!(oplist, "x", i)
end

for i = 1:N-1
    add!(oplist, ["z", "z"], [i, i+1])
end
expectations = inner(sh, psi, oplist, psi)
Zs = expectations[1:N]
Xs = expectations[N+1:2*N]
ZZs = expectations[2*N+1:end]
