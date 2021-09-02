include("src/TensorNetworks.jl")

N = 20
J = 1.0
g = 1.0
kappa = 0.1

dt = 0.01
save = 0.1
tmax = 20.0

# Get the state space
sh = qKCMS(omega, gamma, kappa)

# Create initial state
psi = productMPS(sh, ["dn" for i = 1:N])

# Create the Hamiltonian
H = OpList(N)
add!(H, "x", 1, -J*g)
for i = 1:N-1
    add!(H, ["x"], [i+1], -J*g)
    add!(H, ["z", "z"], [i, i+1], -J)
end

# Create jump operators
jumpops = OpList(N)
for i = 1:N
    add!(jumpops, ["n"], [i], sqrt(kappa))
end

# Observers
obslist = OpList(N)
for i = 1:N
    add!(obslist, ["pu"], [i], sqrt(kappa))
end

println("------------")
gates = trotterize(H, sh, dt; evol="real")

jumps, times = qjmc_simulation(psi, gates, jumpops, sh, tmax, dt; save=save)
