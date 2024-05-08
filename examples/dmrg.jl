#= 
    Example code for performing DMRG, or variational MPS, on the transverse field Ising model.
    DMRG is a method to target area-law ground states for gapped 1D systems, although 
    there are numerous extensions for excited states, 2D systems and time-evolution.
=#
using TensorNetworks

### Model parameters
N = 100 # System size 
h = 1.0 # Transverse field 
g = 0.05 # Longitudinal field 
J = 1.2 # Coupling


### Create the hamiltonian as an MPO 
sh = spinhalf() # Tells the algorithms that we are using a spin half systems
H = OpList(N) # Creates a list of operators contained in the hamiltonian
for i = 1:N 
    add!(H, "x", i, h) # Add transverse field
    add!(H, "z", i, g) # Add a magnetic field
end
for i = 1:N-1
    add!(H, ["z", "z"], [i, i+1], J) # Adds coupling
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
psi, energy = dmrg(psi, H; nsites=2, cutoff=1e-12, maxdim=32, maxsweeps=100)

### Measure observables
# Measure magnetizations
oplist = OpList(N)
for i = 1:N
    add!(oplist, "z", i)
end
for i = 1:N
    add!(oplist, "x", i)
end

# Measure couplings
for i = 1:N-1
    add!(oplist, ["z", "z"], [i, i+1])
end

# Take measurements
expectations = inner(sh, psi, oplist, psi)
Zs = expectations[1:N]
Xs = expectations[N+1:2*N]
ZZs = expectations[2*N+1:end]
