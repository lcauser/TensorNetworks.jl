#= 
    Example code for performing DMRG, or variational MPS, on the transverse field Ising model.
    DMRG is a method to target area-law ground states for gapped 1D systems, although 
    there are numerous extensions for excited states, 2D systems and time-evolution.
=#
include("../src/TensorNetworks.jl")
using .TensorNetworks

### Model parameters
N = 40 # System size 
h = 1.0 # Transverse field 
g = 0.00 # Longitudinal field 
J = 0.9 # Coupling


### Create the hamiltonian as an MPO 
sh = spinhalf() # Tells the algorithms that we are using a spin half systems
H = OpList(N) # Creates a list of operators contained in the hamiltonian
for i = 1:N 
    add!(H, "x", i, -h) # Add transverse field
    add!(H, "z", i, -g) # Add a magnetic field
end
for i = 1:N-1
    add!(H, ["z", "z"], [i, i+1], -J) # Adds coupling
end
H = MPO(sh, H) # Convert to MPO

### Create initial guess
# Random guess
psi = randomMPS(2, N, 1) # 2 is the physical dimension, 1 is the bond dimension

# Alternatively, start with all spins up
#psi = productMPS(sh, ["up" for _ = 1:N])

### Perform DMRG
# nsites (default 2) is the number of tensors in each update; nsites=1 is single site vMPS, nsites=2 is DMRG 
# cutoff is the acceptable SVD, maxdim is the maximum bond dimension
psi, energy = dmrg(psi, H; nsites=2, cutoff=1e-12, maxdim=128, maxsweeps=20, minsweeps=20);

psi2 = randomMPS(2, N, 1) # 2 is the physical dimension, 1 is the bond dimension
psi2, energy2 = dmrg(psi, H, 10*psi; nsites=2, cutoff=1e-12, maxdim=128, maxsweeps=100)