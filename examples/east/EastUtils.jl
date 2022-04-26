using .TensorNetworks

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
        add!(H, ["pu", "x"], [i, i+1], -exp(-s)*sqrt(c*(1-c)))
        add!(H, ["pu", "pu"], [i, i+1], (1-c))
        add!(H, ["pu", "pd"], [i, i+1], c)
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
    EastGenerator(N::Int, c::Float64, s::Float64)

Create the operator list for the generator of the East model for system size
N, temperature c and activity bias s.
"""
function EastGenerator(N::Int, c::Float64, s::Float64)
    H = OpList(N)
    add!(H, "s+", 1, exp(-s)*c)
    add!(H, "s-", 1, exp(-s)*(1-c))
    add!(H, "pu", 1, -(1-c))
    add!(H, "pd", 1, -c)
    for i = 1:N-1
        add!(H, ["pu", "s+"], [i, i+1], exp(-s)*c)
        add!(H, ["pu", "s-"], [i, i+1], exp(-s)*(1-c))
        add!(H, ["pu", "pu"], [i, i+1], -(1-c))
        add!(H, ["pu", "pd"], [i, i+1], -c)
    end
    return H
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
