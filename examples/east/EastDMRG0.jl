#include("src/TensorNetworks.jl")
#using .TensorNetworks


"""
    EastHamiltonian(N::Int, c::Float64, s::Float64)

Create the operator list for the Hamiltonian of the East model for system size
N, temperature c and activity bias s.
"""
function EastHamiltonian(N::Int, c::Float64, s::Float64)
    H = OpList(N)
    add!(H, "x", 1, -exp(-s)*sqrt(c*(1-c)))
    add!(H, "pu", 1, (1-c))
    add!(H, "pd", 1, c)
    for i = 1:N-1
        add!(H, ["id", "x"], [i, i+1], -exp(-s)*sqrt(c*(1-c)))
        add!(H, ["id", "pu"], [i, i+1], (1-c))
        add!(H, ["id", "pd"], [i, i+1], c)
    end
    return H
end


"""
    EastGroundState(N::Int, c::Float64)

Create the ground state to the East model as an MPS.
"""
function EastGroundState(N::Int, c::Float64)
    psi = productMPS(N, [sqrt(c), sqrt(1-c)])
    return psi
end


"""
    EastStationaryState(N::Int, c::Float64)

Create the stationary state to the East model as an MPS.
"""
function EastStationaryState(N::Int, c::Float64)
    psi = productMPS(N, [c, 1-c])
    return psi
end


"""
    EastFlatState(N::Int)

Create the flat state to the East model as an MPS.
"""
function EastFlatState(N::Int)
    psi = productMPS(N, [1, 1])
    return psi
end

"""
    EastActivity(N::Int, c::Float64, s::Float64)

Create the operator list for the Hamiltonian of the East model for system size
N, temperature c and activity bias s.
"""
function EastActivity(N::Int, c::Float64, s::Float64)
    H = OpList(N)
    add!(H, "x", 1, exp(-s)*sqrt(c*(1-c)))
    for i = 1:N-1
        add!(H, ["id", "x"], [i, i+1], exp(-s)*sqrt(c*(1-c)))
    end
    return H
end



# Model parameters
N = 20
c = 0.5

# Create lattice type
sh = spinhalf()

thetas = []
activities = []
rates = []
ss = collect(-2.0:0.05:2.0)

# Create initial guess
psi = randomMPS(2, N, 1)
movecenter!(psi, 1)

for s = ss
    # Create hamiltonian
    H = EastHamiltonian(N, c, s) # Create op list
    H = MPO(sh, H) # Convert to MPO

    # Do DMRG
    psi, energy = dmrg(psi, H; maxsweeps=1000, cutoff=1e-12, maxdim=32, nsites=2)

    # Measure Occupations and Correlations
    activity = real(sum(inner(sh, psi, EastActivity(N, c, s), psi)))

    push!(thetas, -energy)
    push!(activities, activity)
end
rates = -1 .* thetas  .- (ss .* activities)
writedlm("C:/Users/lukec/OneDrive - The University of Nottingham/unconstrained.csv", [ss, thetas, activities, rates], ",")
