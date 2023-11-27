#= 
    Example code for finding the ground state of the 2D TFIM.
    Uses PEPS with the simple update and full update.
=#

using TensorNetworks 
using Plots

### Simulation parameters
# System 
N = 6 # Size of square lattice
h = 1.0 # Transverse field 
g = 0.0 # Longitudinal field 
J = 1.0 # Coupling 

### Create the Hamiltonian
sh = spinhalf()
H = OpList2d(N)
for i = 1:N 
    for j = 1:N 
        # Add fields
        add!(H, ["z"], [i, j], false, g)
        add!(H, ["x"], [i, j], false, h)

        # Add coupling 
        if i < N
            add!(H, ["z", "z"], [i, j], true, J)
        end
        if j < N 
            add!(H, ["z", "z"], [i, j], false, J)
        end
    end
end

### Find the ground state with imaginary time evolution
# Random guess
psi = randomPEPS(2, N, N, 1)

# Simple update
psi, energy = simpleupdate(psi, 1e-2, sh, -1*H; maxiter=5000, maxdim=1, saveiter=1000, chi=1, cutoff=1e-6)
psi, energy = simpleupdate(psi, 1e-3, sh, -1*H; maxiter=5000, maxdim=2, saveiter=1000, chi=16, cutoff=1e-8)
psi, energy = simpleupdate(psi, 1e-3, sh, -1*H; maxiter=5000, maxdim=3, saveiter=1000, chi=36, cutoff=1e-8)

# Full update
psi, energy = fullupdate(psi, -1*H, 1e-3, sh; maxdim=3, maxiter=1000, miniter=1, chi=100, saveiter=1, cutoff=1e-8)

### Measurements & plot
magnetizations = OpList2d(N)
for i = 1:N
    for j = 1:N
        add!(magnetizations, ["z"], [i, j], false, 1)
    end
end
Zs = real(inner(sh, psi, magnetizations, psi; maxchi=200))
Zs = reshape(Zs, (N, N))
heatmap(Zs)