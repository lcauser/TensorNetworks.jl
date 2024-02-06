include("../src/TensorNetworks.jl")
using .TensorNetworks

# Parameters
h = 1.0 # transverse field 
J = 1.0 # coupling 
D = 64 # bond dimension 

# Create uMPS 
psi = randomUMPS(2, D) # 2 is the physical dimension of the lattice site 

# Create Hamiltonian
sh = spinhalf()
H = InfiniteOpList()
add!(H, "x", 1, -h)
add!(H, ["z", "z"], [1, 2], -J)

# Optimise with VUMPS 
psi, energy = vumps(sh, psi, H)

# Take some measurements 
measurements = InfiniteOpList()
add!(measurements, "z", 1)
add!(measurements, "x", 1)
add!(measurements, ["z", "z"], [1, 2])
x, z, zz = inner(sh, psi, measurements)