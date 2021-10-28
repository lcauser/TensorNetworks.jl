include("src/TensorNetworks.jl")

N = 20
kappa = 1.0
omega = 0.4
gamma = 0.05
dt = 0.01
save = 0.1
tmax = 100.0

# Get the state space
sh = qKCMS(omega, gamma, kappa)

# Create initial state
psi = productMPS(sh, ["l" for i = 1:N])

# Create the Hamiltonian
H = OpList(N)
add!(H, "x", 1, omega)
for i = 1:N-1
    add!(H, ["pl", "x"], [i, i+1], omega)
end

# Create jump operators
jumpops = OpList(N)
add!(jumpops, "s-", 1, sqrt(kappa))
add!(jumpops, "s+", 1, sqrt(gamma))
for i = 1:N-1
    add!(jumpops, ["pl", "s-"], [i, i+1], sqrt(kappa))
    add!(jumpops, ["pl", "s+"], [i, i+1], sqrt(gamma))
end

# Observers
obslist = OpList(N)
for i = 1:N
    add!(obslist, ["pu"], [i], sqrt(kappa))
end
observer = QJMCOperators(obslist, sh)

println("------------")
gates = trotterize(H, sh, dt; evol="real")

jumps, times = qjmc_simulation(psi, H, jumpops, sh, tmax, dt, [observer]; save=save, update=true)
