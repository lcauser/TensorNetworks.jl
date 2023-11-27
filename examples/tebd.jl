#= 
    Example code for performing TEBD on the transverse field Ising model.
    TEBD is a method of performing time-evolution on 1D systems with an MPS as
    a variational ansatz. It can be used for imaginary-time evolution to target
    ground states or thermal states, or real-time evolution for quantum quench
    dynamics.
=#
using TensorNetworks 
using Plots

### Simulation parameters
# System 
N = 21 # System size 
h = 1.0 # Transverse field 
g = 20.0 # Longitudinal field 
J = 10.0 # Coupling 

# TEBD parameters 
T = 10.0 # total run time 
t = 5e-2 # How frequent to take measurements 
dt = 5e-3 # Trotter time step 
maxdim = 64 # Maximum bond dimension
cutoff = 1e-10 # Singular value decomposition cutoff

### Write down the Hamiltonian
sh = spinhalf()
H = OpList(N) # Creates a list of operators contained in the hamiltonian
for i = 1:N 
    add!(H, "x", i, h) # Add transverse field
    add!(H, "z", i, g) # Add a magnetic field
end
for i = 1:N-1
    add!(H, ["z", "z"], [i, i+1], J) # Adds coupling
end

### Create an "observer" to measure local observables
magnetization_ops = OpList(N)
for i = 1:N
    add!(magnetization_ops, "z", i)
end
observer = TEBDOperators(sh, magnetization_ops)

### Peform real time evolution
# Uses a second-order Trotter decomposition to implement dynamics
psi = productMPS(sh, [isodd(i) ? "up" : "dn" for i = 1:N]) # Start from Z2 state
psi, energy = tebd(sh, psi, -1im*H, dt, T, t, [observer]; cutoff=cutoff, maxdim=maxdim)

### Take measurements & plot
Zs = real(observer.measurements) # Take measurements from observer 
Zs = mapreduce(permutedims, vcat, Zs) # Make an array 
heatmap(0:t:T, 1:N, Zs') # Plot 
xlabel!("Time")
yaxis!("Space")

# The parameters have been choosen to implement a PXP-like dynamics using
# the TFIM (up to boundary terms). The observed dynamics are explained by 
# a phenomena called quantum many-body scars...
# For added fun, we some ``edge mode'' like behaviour
