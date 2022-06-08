using .TensorNetworks
using Printf
include("EastUtils.jl")

# Model parameters
N = 20
c = 0.05
dt = 0.01
tmax = 1.0
cutoff = 0

# Create lattice type
sh = spinhalf()

# Create hamiltonian
H = EastGenerator(N, c, 0.0) # Create op list
gates = trotterize(sh, H, dt)

total_Zs = []

@time for D = [8]
    # Create initial MPO
    U = productMPO(sh, ["id" for i = 1:N])
    movecenter!(U, 1)

    # Measure norm
    norms = []
    norm_measure = trace(U)
    normal = log(norm_measure)
    U /= norm_measure
    push!(norms, copy(normal))

    # Evolve with trotter gates
    for i = 1:Int(tmax / dt)
        applygates!(U, gates; cutoff=cutoff, maxdim=D)
        norm_measure = trace(U)
        normal += log(norm_measure)
        U /= norm_measure
        push!(norms, copy(normal))
        @printf("time=%.3f, norm=%.12f, maxbonddim=%d \n",
                i*dt, real(normal), maxbonddim(U))
    end
    push!(total_Zs, norms)
end
