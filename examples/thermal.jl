#= 
    Thermal states of Hamiltonians can be approximated by evolving the identity
    matrix in (imaginary) time. Here, we use Trotterization to evolve an MPO 
    to \beta / 2, and then enforce it to be unitary by taking the matrix product 
    with its adjoint.
=#
using TensorNetworks 
using Printf

### Simulation parameters
# System 
N = 40 # System size 
h = 1.0 # Transverse field 
g = 0.0 # Longitudinal field 
J = 1.0 # Coupling 

# TEBD parameters 
beta = 4.0 # total run time 
dt = 5e-3 # Trotter time step 
maxdim = 64 # Maximum bond dimension
cutoff = 1e-10 # Singular value decomposition cutoff

### Write down the Hamiltonian & Trotterize
sh = spinhalf()
H = OpList(N) # Creates a list of operators contained in the hamiltonian
for i = 1:N 
    add!(H, "x", i, h) # Add transverse field
    add!(H, "z", i, g) # Add a magnetic field
end
for i = 1:N-1
    add!(H, ["z", "z"], [i, i+1], J) # Adds coupling
end
gates = trotterize(sh, -1*H, dt) # Second order trotter by default

### Evolve the identity matrix
U = productMPO(sh, ["id" for i = 1:N])
for t = dt:dt:beta/2
    applygates!(U, gates; cutoff=cutoff, maxdim=maxdim)
    @printf("time=%.3f, maxbonddim=%d \n", 2*t, maxbonddim(U))
end

### Measure the energy 
H = MPO(sh, H)
energy = trace(H, adjoint(U), U) / trace(adjoint(U), U)
