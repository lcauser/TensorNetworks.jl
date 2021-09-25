mutable struct Environment
    psi::PEPS
    phi::PEPS
    chi::Int
    cutoff::Float64
    direction::Bool
    blocks::Vector
    center::Int
    blocks2::Vector
    center2::Int
end

function Environment(psi::PEPS, phi::PEPS; kwargs...)
    # Check PEPS are the same length
    (height1, length1) = size(psi)
    (height2, length2) = size(phi)
    height1 != height2 && error("PEPS must be equal size.")
    length1 != length2 && error("PEPS must be equal size.")

    # Get the truncation criteria
    chi::Int = get(kwargs, :chi, 0)
    chi = chi == 0 ? 2*maxbonddim(psi)*maxbonddim(phi) : chi
    cutoff::Real = get(kwargs, :cutoff, 1e-12)

    # Build up the environment
    direction::Bool = get(kwargs, :direction, false)
    center::Int = get(kwargs, :center, 1)
    center2::Int = get(kwargs, :center2, 1)

    blocks = [bMPO(length(psi)) for i = 1:length(psi)]
    blocks2 = Any[ones(ComplexF64, 1, 1, 1, 1) for i = 1:length(psi)]
    env = Environment(psi, phi, chi, cutoff, direction, blocks, 0, blocks2, 0)
    build!(env, center, center2, direction)
    return env
end

function block(env::Environment, idx::Int)
    if idx < 1 || idx > length(env.psi)
        return bMPO(length(env.psi))
    else
        return env.blocks[idx]
    end
end

function block2(env::Environment, idx::Int)
    if idx < 1 || idx > length(env.psi)
        return ones(ComplexF64, 1, 1, 1, 1)
    else
        return env.blocks2[idx]
    end
end

function findsite(env::Environment)
    if !env.direction
        site1 = env.center
        site2 = env.center2
    else
        site1 = env.center2
        site2 = env.center
    end
    return site1, site2
end

### Build the environment
function buildup!(env::Environment, idx::Int)
    # Retrieve the previous block
    prev = deepcopy(block(env, idx+1))

    # Loop through each tensor
    for i = 1:length(env.psi)
        # Retrieve the tensors
        M = prev[i]
        A = env.psi[idx, i]
        B = env.phi[idx, i]

        # Contract tensors
        prod = contract(M, conj(A), 2, 3)
        prod = contract(prod, B, 2, 3)
        prod = trace(prod, 6, 10)

        # Reshape into bMPO tensor
        prod, cmb = combineidxs(prod, [1, 3, 6])
        prod = moveidx(prod, 6, 1)
        prod, cmb = combineidxs(prod, [2, 4, 6])

        # Update the tensor
        prev[i] = prod
    end

    # Apply variational sweeps to limit the bond dimension
    prev = vbMPO(prev; chi=env.chi, cutoff=env.cutoff)

    # Save the block
    env.blocks[idx] = prev
end


function builddown!(env::Environment, idx::Int)
    # Retrieve the previous block
    prev = deepcopy(block(env, idx-1))

    # Loop through each tensor
    for i = 1:length(env.psi)
        # Retrieve the tensors
        M = prev[i]
        A = env.psi[idx, i]
        B = env.phi[idx, i]

        # Contract tensors
        prod = contract(M, conj(A), 2, 2)
        prod = contract(prod, B, 2, 2)
        prod = trace(prod, 6, 10)

        # Reshape into bMPO tensor
        prod, cmb = combineidxs(prod, [1, 3, 6])
        prod = moveidx(prod, 6, 1)
        prod, cmb = combineidxs(prod, [2, 4, 6])

        # Update the tensor
        prev[i] = prod
    end

    # Apply variational sweeps to limit the bond dimension
    prev = vbMPO(prev; chi=env.chi, cutoff=env.cutoff)

    # Save the block
    env.blocks[idx] = prev
end


function buildright!(env::Environment, idx::Int)
    # Retrieve the previous block
    prev = deepcopy(block(env, idx-1))

    # Loop through each tensor
    for i = 1:length(env.psi)
        # Retrieve the tensors
        M = prev[i]
        A = env.psi[i, idx]
        B = env.phi[i, idx]

        # Contract tensors
        prod = contract(M, conj(A), 2, 1)
        prod = contract(prod, B, 2, 1)
        prod = trace(prod, 6, 10)

        # Reshape into bMPO tensor
        prod, cmb = combineidxs(prod, [1, 3, 6])
        prod = moveidx(prod, 6, 1)
        prod, cmb = combineidxs(prod, [2, 3, 5])

        # Update the tensor
        prev[i] = prod
    end

    # Apply variational sweeps to limit the bond dimension
    prev = vbMPO(prev; chi=env.chi, cutoff=env.cutoff)

    # Save the block
    env.blocks[idx] = prev
end


function buildleft!(env::Environment, idx::Int)
    # Retrieve the previous block
    prev = deepcopy(block(env, idx+1))

    # Loop through each tensor
    for i = 1:length(env.psi)
        # Retrieve the tensors
        M = prev[i]
        A = env.psi[i, idx]
        B = env.phi[i, idx]

        # Contract tensors
        prod = contract(M, conj(A), 2, 4)
        prod = contract(prod, B, 2, 4)
        prod = trace(prod, 6, 10)

        # Reshape into bMPO tensor
        prod, cmb = combineidxs(prod, [1, 4, 7])
        prod = moveidx(prod, 6, 1)
        prod, cmb = combineidxs(prod, [2, 4, 6])

        # Update the tensor
        prev[i] = prod
    end

    # Apply variational sweeps to limit the bond dimension
    prev = vbMPO(prev; chi=env.chi, cutoff=env.cutoff)

    # Save the block
    env.blocks[idx] = prev
end


function buildmpsright!(env::Environment, idx::Int)
    # Fetch left block
    left = block2(env, idx-1)

    # Fetch new blocks
    M1 = block(env, env.center-1)[idx]
    M2 = block(env, env.center+1)[idx]
    A1 = !env.direction ? env.psi[env.center, idx] : env.psi[idx, env.center]
    A2 = !env.direction ? env.phi[env.center, idx] : env.phi[idx, env.center]

    # Contract them to grow the block
    if env.direction == false
        prod = contract(left, M1, 1, 1)
        prod = contract(prod, conj(A1), 1, 1)
        prod = trace(prod, 3, 6)
        prod = contract(prod, A2, 1, 1)
        prod = trace(prod, 2, 7)
        prod = trace(prod, 5, 8)
        prod = contract(prod, M2, 1, 1)
        prod = trace(prod, 2, 6)
        prod = trace(prod, 3, 5)
    else
        prod = contract(left, M1, 1, 1)
        prod = contract(prod, conj(A1), 1, 2)
        prod = trace(prod, 3, 6)
        prod = contract(prod, A2, 1, 2)
        prod = trace(prod, 2, 7)
        prod = trace(prod, 5, 8)
        prod = contract(prod, M2, 1, 1)
        prod = trace(prod, 3, 6)
        prod = trace(prod, 4, 5)
    end

    # Store the block
    env.blocks2[idx] = prod
end


function buildmpsleft!(env::Environment, idx::Int)
    # Fetch left block
    right = block2(env, idx+1)

    # Fetch new blocks
    M1 = block(env, env.center-1)[idx]
    M2 = block(env, env.center+1)[idx]
    A1 = !env.direction ? env.psi[env.center, idx] : env.psi[idx, env.center]
    A2 = !env.direction ? env.phi[env.center, idx] : env.phi[idx, env.center]

    # Contract them to grow the block
    if env.direction == false
        prod = contract(M2, right, 4, 4)
        prod = contract(A2, prod, 4, 6)
        prod = trace(prod, 3, 7)
        prod = contract(conj(A1), prod, 4, 7)
        prod = trace(prod, 3, 9)
        prod = trace(prod, 3, 6)
        prod = contract(M1, prod, 4, 6)
        prod = trace(prod, 3, 7)
        prod = trace(prod, 2, 4)
    else
        prod = contract(M2, right, 4, 4)
        prod = contract(A2, prod, 3, 6)
        prod = trace(prod, 3, 7)
        prod = contract(conj(A1), prod, 3, 7)
        prod = trace(prod, 3, 9)
        prod = trace(prod, 3, 6)
        prod = contract(M1, prod, 4, 6)
        prod = trace(prod, 3, 6)
        prod = trace(prod, 2, 3)
    end

    # Store the block
    env.blocks2[idx] = prod
end


function build!(env::Environment, idx1::Int, idx2::Int, direction::Bool=false)
    # Check for a change in direction for which the environment is built
    if direction != env.direction
        env.direction = direction
        env.blocks = [bMPO(length(psi)) for i = 1:length(psi)]
        env.center = 0
        env.blocks2 = [ones(ComplexF64, 1, 1, 1, 1) for i = 1:length(psi)]
        env.center2 = 0
    end

    # Build the blocks
    rebuild = true # Book keeping term for rebuilding MPS blocks
    if direction == false
        if env.center == 0
            # The center isn't set; build from left and right
            for i = 1:idx1-1
                builddown!(env, i)
            end
            for i = 1:length(env.psi) - idx1
                buildup!(env, length(env.psi) + 1 - i)
            end
        elseif env.center < idx1
            # Build from above
            for i = 1:idx1-env.center
                builddown!(env, env.center-1+i)
            end
        elseif env.center > idx1
            # Build from below
            for i = 1:env.center-idx1
                buildup!(env.center+1-i)
            end
        else
            rebuild = false
        end
        env.center = idx1
    else
        if env.center == 0
            # The center isn't set; build from left and right
            for i = 1:idx2-1
                buildright!(env, i)
            end
            for i = 1:length(env.psi) - idx2
                buildleft!(env, length(env.psi) + 1 - i)
            end
        elseif env.center < idx2
            # Build from left
            for i = 1:idx2-env.center
                buildright!(env, env.center-1+i)
            end
        elseif env.center > idx2
            # Build from right
            for i = 1:env.center-idx2
                buildleft!(env, env.center+1-i)
            end
        else
            rebuild = false
        end
        env.center = idx2
    end

    # Check too see if blocks should be rebuilt
    if rebuild
        env.blocks2 = [ones(ComplexF64, 1, 1, 1, 1) for i = 1:length(psi)]
        env.center2 = 0
    end

    # Decide which idx to build MPS blocks too
    idx = env.direction ? idx1 : idx2

    # Build MPS-like blocks
    if env.center2 == 0
        # Build from left and right
        for i = 1:idx-1
            buildmpsright!(env, i)
        end
        for i = 1:length(env.psi)-idx
            buildmpsleft!(env, length(env.psi)+1-i)
        end
    elseif env.center2 < idx
        # Build from the left
        for i = 1:idx-env.center2
            buildmpsright!(env, env.center2-1+i)
        end
    elseif env.center2 > idx
        # Build from the right
        for i = 1:env.center2-idx
            buildmpsleft!(env, env.center2 + 1 - i)
        end
    end
    env.center2 = idx
end

function maxbonddim(env::Environment)
    chi = 0
    for b in env.blocks
        chi = max(chi, maxbonddim(b))
    end
    return chi
end


function inner(env::Environment, ops::Vector{Array{ComplexF64, 2}},
               sites::Vector{Int}, direction::Bool=false)
    # Build the environment to the correct site
    build!(env, sites[1], sites[2], direction)

    # Fetch the left and right blocks
    prod = block2(env, env.center2-1)
    right = block2(env, env.center2+length(ops))

    # Loop through each site in the operator
    for i = 1:length(ops)
        # Fetch relevent tensors
        A = !env.direction ? env.psi[env.center, env.center2+i-1] : env.psi[env.center2+i-1, env.center]
        B = !env.direction ? env.phi[env.center, env.center2+i-1] : env.phi[env.center2+i-1, env.center]

        M1 = block(env, env.center-1)[env.center2-1+i]
        M2 = block(env, env.center+1)[env.center2-1+i]

        # Apply the gate
        A = contract(A, ops[i], 5, 1)

        # Do the contraction
        if !direction
            prod = contract(prod, M1, 1, 1)
            prod = contract(prod, conj(A), 1, 1)
            prod = trace(prod, 3, 6)
            prod = contract(prod, B, 1, 1)
            prod = trace(prod, 2, 7)
            prod = trace(prod, 5, 8)
            prod = contract(prod, M2, 1, 1)
            prod = trace(prod, 2, 6)
            prod = trace(prod, 3, 5)
        else
            prod = contract(prod, M1, 1, 1)
            prod = contract(prod, conj(A), 1, 2)
            prod = trace(prod, 3, 6)
            prod = contract(prod, B, 1, 2)
            prod = trace(prod, 2, 7)
            prod = trace(prod, 5, 8)
            prod = contract(prod, M2, 1, 1)
            prod = trace(prod, 3, 6)
            prod = trace(prod, 4, 5)
        end
    end

    # Contract with right block
    prod = contract(prod, right, 1, 1)
    prod = trace(prod, 3, 6)
    prod = trace(prod, 1, 3)
    prod = trace(prod, 1, 2)

    return prod[1]
end

function inner(env::Environment, st::Sitetypes, ops::Vector{String},
               sites::Vector{Int}, direction::Bool=false)
    # Calculate the operator list
    newops = [op(st, ops[i]) for i = 1:length(ops)]
    return inner(env, newops, sites, direction)
end

function inner(st::Sitetypes, psi::PEPS, ops::OpList2d, phi::PEPS; kwargs...)
    expectations = ComplexF64[0 for i = 1:length(ops.sites)]


    # Calculate environment
    env = Environment(psi, phi; kwargs...)

    # Loop through each direction
    for direction = [false, true]
        # Loop through each site
        for center = 1:length(psi)
            for center2 = 1:length(psi)
                site1 = !direction ? center : center2
                site2 = !direction ? center2 : center
                sites = [site1, site2]

                # Find all the indexs which correspond to site and direction
                idxs = siteindexs(ops, sites, direction)

                # Find the expectation of each
                for idx in idxs
                    expectations[idx] = inner(env, st, ops.ops[idx], sites, direction)
                    expectations[idx] *= ops.coeffs[idx]
                end
            end
        end
    end

    return expectations
end

function inner(env::Environment)
    # Fetch the blocks
    site1 = !env.direction ? env.center : env.center2
    site2 = !env.direction ? env.center2 : env.center
    prod = block2(env, env.center2-1)
    right = block2(env, env.center2+1)
    M1 = block(env, env.center-1)[env.center2]
    M2 = block(env, env.center+1)[env.center2]
    A1 = env.psi[site1, site2]
    A2 = env.phi[site1, site2]

    # Contract
    prod = contract(prod, M1, 1, 1)
    if !env.direction
        prod = contract(prod, conj(A1), [1, 4], [1, 2])
        prod = contract(prod, A2, [1, 3, 7], [1, 2, 5])
        prod = contract(prod, M2, [1, 3, 5], [1, 2, 3])
    else
        prod = contract(prod, conj(A1), [1, 4], [2, 1])
        prod = contract(prod, A2, [1, 3, 7], [2, 1, 5])
        prod = contract(prod, M2, [1, 4, 6], [1, 2, 3])
    end
    prod = contract(prod, right, [1, 2, 3, 4], [1, 2, 3, 4])[1]
    return prod
end

function inner(psi::PEPS, phi::PEPS; kwargs...)
    env = Environment(psi, phi; kwargs...)
    return inner(env)
end


function inner(env::Environment, ops::Vector{Vector{Array{ComplexF64, 2}}}, coeffs::Vector{Number})
    # Loop through each direction
    prod = 0
    for direction = [false, true]
        # Loop through each site
        for center = 1:length(psi)
            for center2 = 1:length(psi)
                site1 = !direction ? center : center2
                site2 = !direction ? center2 : center
                sites = [site1, site2]
                for i = 1:length(ops)
                    if center2 + length(ops[i])-1 <= length(psi)
                        prod += coeffs[i]*inner(env, ops[i], sites, direction)
                    end
                end
            end
        end
    end
    return prod
end

function inner(psi::PEPS, phi::PEPS, ops::Vector{Vector{Array{ComplexF64, 2}}},
               coeffs::Vector{Number}; kwargs...)
    env = Environement(psi, phi; kwargs...)
    return inner(env, ops, coeffs)
end

### Reduced tensors environment
function ReducedTensorEnv(env::Environment, site1, site2, dir, A1, A2)
    # Determine which centers
    if !dir
        center = site1
        center2 = site2
    else
        center = site2
        center2 = site1
    end

    # Retrieve relevent MPS blocks and tensors
    left = block2(env, center2-1)
    right = block2(env, center2+2)
    M1 = block(env, center-1)[center2]
    M2 = block(env, center+1)[center2]
    M3 = block(env, center-1)[center2+1]
    M4 = block(env, center+1)[center2+1]

    # Contract to reduced tensor environment
    if !dir
        # Grow the left block
        left = contract(left, M1, 1, 1)
        left = contract(left, conj(A1), 1, 1)
        left = trace(left, 3, 6)
        left = contract(left, A1, 1, 1)
        left = trace(left, 2, 6)
        left = contract(left, M2, 1, 1)
        left = trace(left, 2, 6)
        left = trace(left, 3, 5)

        # Grow the right block
        right = contract(M4, right, 4, 4)
        right = contract(A2, right, 4, 6)
        right = trace(right, 3, 6)
        right = contract(conj(A2), right, 4, 6)
        right = trace(right, 3, 7)
        right = contract(M3, right, 4, 6)
        right = trace(right, 3, 7)
        right = trace(right, 2, 4)
    else
        # Grow the left block
        left = contract(left, M1, 1, 1)
        left = contract(left, conj(A1), 1, 2)
        left = trace(left, 3, 6)
        left = contract(left, A1, 1, 2)
        left = trace(left, 2, 6)
        left = contract(left, M2, 1, 1)
        left = trace(left, 3, 6)
        left = trace(left, 4, 5)

        # Grow the right block
        right = contract(M4, right, 4, 4)
        right = contract(A2, right, 3, 6)
        right = trace(right, 3, 6)
        right = contract(conj(A2), right, 3, 6)
        right = trace(right, 3, 7)
        right = contract(M3, right, 4, 6)
        right = trace(right, 2, 4)
        right = trace(right, 2, 4)
    end
    prod = contract(left, right, 1, 1)
    prod = trace(prod, 3, 6)

    return prod
end
