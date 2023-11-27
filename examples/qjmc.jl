#= 
    Example code for quantum jump Monte Carlo on the TFIM with an MPS ansatz.
    QJMC is an unravelling of open quantum systems onto a stochastic 
    dynamics. It allows us to simulate open dynamics by averaging over the
    evolution of pure states.
=#
using TensorNetworks 
using Plots

### Simulation parameters
# System 
N = 21 # System size 
h = 1.0 # Transverse field 
g = 20.0 # Longitudinal field 
J = 10.0 # Coupling 
gamma = 0.1 # Dissipation rate

# TEBD parameters 
T = 10.0 # total run time 
t = 5e-2 # How frequent to take measurements 
dt = 5e-3 # Trotter time step 
maxdim = 64 # Maximum bond dimension
cutoff = 1e-10 # Singular value decomposition cutoff

### Create the Hamiltonian 
sh = spinhalf()
H = OpList(N) # Creates a list of operators contained in the hamiltonian
for i = 1:N 
    add!(H, "x", i, h) # Add transverse field
    add!(H, "z", i, g) # Add a magnetic field
end
for i = 1:N-1
    add!(H, ["z", "z"], [i, i+1], J) # Adds coupling
end

### Add dissipation 
jump_operators = OpList(N)
for i = 1:N 
    add!(jump_operators, "s-", i, sqrt(gamma))
end

### Measure some magnetizations 
# Observers
observer_list = OpList(N)
for i = 1:N
    add!(observer_list, "z", i)
end
observer = QJMCOperators(observer_list, sh)

### Run a simulation 
psi = productMPS(sh, [isodd(i) ? "up" : "dn" for i = 1:N]) # Start from Z2 state
jumps, times = qjmc_simulation(sh, psi, H, jump_operators, T, dt, [observer]; save=t, cutoff=cutoff, maxdim=maxdim)

### Take measurements & plot
Zs = real(observer.measurements) # Take measurements from observer 
Zs = mapreduce(permutedims, vcat, Zs) # Make an array 
heatmap(0:t:T, 1:N, Zs') # Plot 
xlabel!("Time")
yaxis!("Space")
