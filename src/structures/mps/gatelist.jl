"""
    GateList(length::Int, sites::Vector{Vector{Int}}, gates::Vector{Vector{Array}})
    GateList(length::Int)

Create a gatelist, which contains sequences of non-overlapping gates
to apply to an MPS.
"""
mutable struct GateList
    length::Int
    sites::Vector{Vector{Int}}
    gates::Vector{Vector{Array}}
end
GateList(length::Int) = GateList(length, [], [])

"""
    length(gl::GateList)

Return the number of sites the gatelist acts on.
"""
length(gl::GateList) = gl.length


"""
    gatesize(gate)

Determine the number of sites a gate acts on.
"""
function gatesize(gate)
    sz = length(size(gate))
    sz % 2 == 1 && error("The gate must have even number of dimensions.")
    return Int(sz / 2)
end


"""
    add!(gl::GateList, sites::Vector{Int}, gates::Vector{Array})

Add a sequences of gates to a gatelist.
"""
function add!(gl::GateList, sites::Vector{Int}, gates::Vector{Array})
    # Sort the gates by the site order
    perm = sortperm(sites)
    sites = sites[perm]
    gates = gates[perm]

    # Check they are the same length
    length(sites) != length(gates) && error("The site and gate list must be the same length.")

    # Check there are no overlapping gates
    for i = 1:length(sites)-1
        if sites[i] + gatesize(gates[i]) - 1 >= sites[i+1]
            error("Gates in one sequence cannot be shared by sites.")
        end
    end
    if sites[end] + gatesize(gates[end]) - 1 > length(gl)
        error("The bond gates cannot exceed the length of the lattice.")
    end

    # Add to gatelist
    push!(gl.sites, sites)
    push!(gl.gates, gates)
end


"""
    trotterize(st::Sitetypes, ops::OpList, dt::Float64; kwargs...)

Trotterize an operator list to produce a sequence of gates to approximately
evolve an MPS.

Key arguments:
    - order::Int : Trotter order
    - evol::String : "real" or "imag" time evolution.
"""
function trotterize(st::Sitetypes, ops::OpList, dt::Float64; kwargs...)
    # Create a gate list and find the interaction range
    gl = GateList(length(ops), [], [])
    rng = siterange(ops)

    # Decide on the trotter order
    order::Int = get(kwargs, :order, 2)
    order = rng == 1 ? 1 : order
    (0 > order || order > 2) && error("Only trotter order 1 and 2 are supported.")

    # Choose the time
    evol::String = get(kwargs, :evol, "imag")
    dt = evol == "real" ? -1im*dt : dt

    # Calculate the gates
    for i = 1:rng
        time = (i < rng && order == 2) ? dt / 2 : dt
        gates = []
        sites::Vector{Int} = []
        site::Int = i
        while site <= length(ops)
            # Calculate the operator
            gate = sitetensor(ops, st, site)
            if gate != false
                # Exponentiate
                gate = exp(time * gate, [Int(2*i) for i = 1:gatesize(gate)])

                # Add to list
                push!(gates, gate)
                push!(sites, site)
            end
            site = site + rng
        end

        # Add to gatelist
        add!(gl, convert(Vector{Int}, sites), convert(Vector{Array}, gates))
    end

    # Add the first rng - 1 gates backwards
    if order == 2
        for i = 1:rng-1
            add!(gl, gl.sites[rng-i], gl.gates[rng-i])
        end
    end

    return gl
end

"""
    applygate(psi::GMPS, site::Int, gate, direction::Bool = false; kwargs...)

Apply a gate to the MPS at a starting site. Specify a direction to move the
gauge after truncation. Returns the error from truncation.

Key arguments:
    - cutoff::Float64 : truncation cutoff error
    - maxdim::Int : maximum truncation bond dimension
    - mindim::Int : minimum truncation bond dimension.
"""
function applygate(psi::GMPS, site::Int, gate, direction::Bool = false; kwargs...)
    # Find the interaction range of the gate
    rng = gatesize(gate)

    # Calculate the product of all sites which the gate is applied too.
    prod = psi[site]
    for i = 1:rng-1
        prod = contract(prod, psi[site+i], length(size(prod)), 1)
    end

    # Contract with the gate
    r = rank(psi)
    prod = contract(prod, gate, [2+r*(i-1) for i = 1:rng], [2*i for i = 1:rng])

    # Move idxs to the correct place
    for i = 1:rng
        prod = moveidx(prod, 2+(r-1)*rng+i, 2+r*(i-1))
    end

    # Replace the tensors
    replacesites!(psi, prod, site, direction, false; kwargs...)
    
    # Calculate error
    prod_err = psi[site]
    for i = 1:rng-1
        prod_err = contract(prod_err, psi[site+i], length(size(prod_err)), 1)
    end
    error = contract(conj(prod), prod_err, [i for i=1:length(size(prod))], [i for i=1:length(size(prod))])
    return abs.(error)^2
end


"""
    applygate!(psi::GMPS, site::Int, gate, direction::Bool = false; kwargs...)

Apply a gate to the MPS at a starting site. Specify a direction to move the
gauge after truncation.

Key arguments:
    - cutoff::Float64 : truncation cutoff error
    - maxdim::Int : maximum truncation bond dimension
    - mindim::Int : minimum truncation bond dimension.
"""
function applygate!(psi::GMPS, site::Int, gate, direction::Bool = false; kwargs...)
    error = applygate(psi, site, gate, direction; kwargs...)
end



function applygates(psi::GMPS, gates::GateList; kwargs...)
    # Error terms 
    error = 1

    # Apply the sequences in order
    for row = 1:length(gates.gates)
        # Determine the first and last site to be acted on
        firstsite = gates.sites[row][1]
        lastsite = gates.sites[row][end] + gatesize(gates.gates[row][end]) - 1

        # Determine the distance to the center from each and decide where to go
        direction = abs(center(psi) - firstsite) < abs(center(psi) - lastsite) ? false : true

        # Apply the gates in the row
        for i = 1:length(gates.gates[row])
            # Determine the gate
            gate = direction ? length(gates.gates[row]) + 1 - i : i

            # Move orthogonal center
            if !direction
                ctr = gates.sites[row][gate]
            else
                ctr = gates.sites[row][gate] + gatesize(gates.gates[row][gate]) - 1
            end
            movecenter!(psi, ctr; kwargs...)

            # Apply the gate
            error *= applygate(psi, gates.sites[row][gate], gates.gates[row][gate],
                               direction; kwargs...)
        end
    end

    return error
end

function applygates!(psi::GMPS, gates::GateList; kwargs...)
    error = applygates(psi, gates; kwargs...)
end
