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
    dropoff::Int
end

function Environment(psi::PEPS, phi::PEPS; kwargs...)
    # Check PEPS are the same length
    (height1, length1) = size(psi)
    (height2, length2) = size(phi)
    height1 != height2 && error("PEPS must be equal size.")
    length1 != length2 && error("PEPS must be equal size.")

    # Get the truncation criteria
    chi::Int = get(kwargs, :chi, 0)
    chi = chi == 0 ? 4*maxbonddim(psi)*maxbonddim(phi) : chi
    cutoff::Real = get(kwargs, :cutoff, 1e-16)

    # Build up the environment
    direction::Bool = get(kwargs, :direction, false)
    center::Int = get(kwargs, :center, 1)
    center2::Int = get(kwargs, :center2, 1)
    dropoff::Int = get(kwargs, :dropoff, 0)

    blocks = [bMPO(length(psi)) for i = 1:length(psi)]
    blocks2 = Any[ones(ComplexF64, 1, 1, 1, 1) for i = 1:length(psi)]
    env = Environment(psi, phi, chi, cutoff, direction, blocks, 0, blocks2, 0, dropoff)
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
    # Fetch the upwards bond dimensions
    bonddims1 = []
    bonddims2 = []
    for i = 1:length(env.psi)
        push!(bonddims1, size(env.psi[idx, i])[2])
        push!(bonddims2, size(env.phi[idx, i])[2])
    end

    # Create a random bMPO
    bMPO = randombMPO(length(env.psi), env.chi, bonddims1, bonddims2)

    # Apply variational sweeps to limit the bond dimension
    bMPO = vbMPO(bMPO, env, true, idx; chi=env.chi, cutoff=env.cutoff)

    # Save the block
    env.blocks[idx] = bMPO
end


function builddown!(env::Environment, idx::Int)
    # Fetch the upwards bond dimensions
    bonddims1 = []
    bonddims2 = []
    for i = 1:length(env.psi)
        push!(bonddims1, size(env.psi[idx, i])[3])
        push!(bonddims2, size(env.phi[idx, i])[3])
    end

    # Create a random bMPO
    bMPO = randombMPO(length(env.psi), env.chi, bonddims1, bonddims2)

    # Apply variational sweeps to limit the bond dimension
    bMPO = vbMPO(bMPO, env, false, idx; chi=env.chi, cutoff=env.cutoff)

    # Save the block
    env.blocks[idx] = bMPO
end


function buildright!(env::Environment, idx::Int)
    # Fetch the upwards bond dimensions
    bonddims1 = []
    bonddims2 = []
    for i = 1:length(env.psi)
        push!(bonddims1, size(env.psi[i, idx])[4])
        push!(bonddims2, size(env.phi[i, idx])[4])
    end

    # Create a random bMPO
    bMPO = randombMPO(length(env.psi), env.chi, bonddims1, bonddims2)

    # Apply variational sweeps to limit the bond dimension
    bMPO = vbMPO(bMPO, env, false, idx; chi=env.chi, cutoff=env.cutoff)

    # Save the block
    env.blocks[idx] = bMPO
end


function buildleft!(env::Environment, idx::Int)
    # Fetch the upwards bond dimensions
    bonddims1 = []
    bonddims2 = []
    for i = 1:length(env.psi)
        push!(bonddims1, size(env.psi[i, idx])[1])
        push!(bonddims2, size(env.phi[i, idx])[1])
    end

    # Create a random bMPO
    bMPO = randombMPO(length(env.psi), env.chi, bonddims1, bonddims2)

    # Apply variational sweeps to limit the bond dimension
    bMPO = vbMPO(bMPO, env, true, idx; chi=env.chi, cutoff=env.cutoff)

    # Save the block
    env.blocks[idx] = bMPO
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
    prod = contract(left, M1, 1, 1)
    if env.direction == false
        prod = contract(prod, conj(A1), [1, 4], [1, 2])
        prod = contract(prod, A2, [1, 3, 7], [1, 2, 5])
        prod = contract(prod, M2,[1, 3, 5], [1, 2, 3])
    else
        prod = contract(prod, conj(A1), [1, 4], [2, 1])
        prod = contract(prod, A2, [1, 3, 7], [2, 1, 5])
        prod = contract(prod, M2, [1, 4, 6], [1, 2, 3])
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
    prod = contract(M2, right, 4, 4)
    if env.direction == false
        prod = contract(A2, prod, [3, 4], [3, 6])
        prod = contract(conj(A1), prod, [3, 4, 5], [5, 7, 3])
        prod = contract(M1, prod, [2, 3, 4], [2, 4, 6])
    else
        prod = contract(A2, prod, [3, 4], [6, 3])
        prod = contract(conj(A1), prod, [3, 4, 5], [7, 5, 3])
        prod = contract(M1, prod, [2, 3, 4], [1, 3, 6])
    end

    # Store the block
    env.blocks2[idx] = prod
end


function build!(env::Environment, idx1::Int, idx2::Int, direction::Bool=false)
    # Check for a change in direction for which the environment is built
    if direction != env.direction
        env.direction = direction
        env.blocks = [bMPO(length(env.psi)) for i = 1:length(env.psi)]
        env.center = 0
        env.blocks2 = [ones(ComplexF64, 1, 1, 1, 1) for i = 1:length(env.psi)]
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
                buildup!(env, env.center+1-i)
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
        env.blocks2 = [ones(ComplexF64, 1, 1, 1, 1) for i = 1:length(env.psi)]
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
        prod = contract(prod, M1, 1, 1)
        if !direction
            prod = contract(prod, conj(A), [1, 4], [1, 2])
            prod = contract(prod, B, [1, 3, 7], [1, 2, 5])
            prod = contract(prod, M2, [1, 3, 5], [1, 2, 3])
        else
            prod = contract(prod, conj(A), [1, 4], [2, 1])
            prod = contract(prod, B, [1, 3, 7], [2, 1, 5])
            prod = contract(prod, M2, [1, 4, 6], [1, 2, 3])
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
    # Calculate environment
    env = Environment(psi, phi; kwargs...)

    return inner(st, env, ops; kwargs...)
end

function inner(st::Sitetypes, env::Environment, ops::OpList2d; kwargs...)
    expectations = ComplexF64[0 for i = 1:length(ops.sites)]
    # Loop through each direction
    for direction = [false, true]
        # Loop through each site
        for center = 1:length(env.psi)
            for center2 = 1:length(env.psi)
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
        for center = 1:length(env.psi)
            for center2 = 1:length(env.psi)
                site1 = !direction ? center : center2
                site2 = !direction ? center2 : center
                sites = [site1, site2]
                for i = 1:length(ops)
                    if center2 + length(ops[i])-1 <= length(env.psi)
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
        left = contract(left, conj(A1), [1, 4], [1, 2])
        left = contract(left, A1, [1, 3], [1, 2])
        left = contract(left, M2, [1, 3, 5], [1, 2, 3])

        # Grow the right block
        right = contract(M4, right, 4, 4)
        right = contract(A2, right, [3, 4], [3, 6])
        right = contract(conj(A2), right, [3, 4], [4, 6])
        right = contract(M3, right, [2, 3, 4], [2, 4, 6])
    else
        # Grow the left block
        left = contract(left, M1, 1, 1)
        left = contract(left, conj(A1), [1, 4], [2, 1])
        left = contract(left, A1, [1, 3], [2, 1])
        left = contract(left, M2, [1, 4, 6], [1, 2, 3])

        # Grow the right block
        right = contract(M4, right, 4, 4)
        right = contract(A2, right, [3, 4], [6, 3])
        right = contract(conj(A2), right, [3, 4], [6, 4])
        right = contract(M3, right, [2, 3, 4], [1, 3, 6])
    end
    prod = contract(left, right, [1, 4], [1, 4])
    #return prod

    # Find the closest semi-positive hermitian
    dims = size(prod)
    prod2, cmb1 = combineidxs(prod, [1, 3])
    prod2, cmb1 = combineidxs(prod2, [1, 2])
    prod2 = 0.5*real(prod2 + conj(transpose(prod2)))

    # Decompose into spectrum
    F = eigen(prod2)
    vals = diagm([real(val) < 0 ? 0.0 : val for val in F.values])

    # Find the X matric, and apply LQ/QR decompositon
    #X = contract(F.vectors, sqrt.(vals), 2, 1)
    #X = moveidx(X, 1, 2)
    #X = reshape(X, (size(X)[1], dims[1], dims[3]))
    #Xprime, R = qr(X, 2)
    #L, Xprime = lq(X, 3)
    #X = contract(X, inv(R), 2, 1)
    #X = moveidx(X, 3, 2)
    #X = contract(inv(L), X, 2, 3)
    #X = moveidx(X, 1, 3)

    # Multiply X with hermitian conjugate
    #prod2 = contract(X, X, 1, 1)

    prod2 = contract(F.vectors, vals, 2, 1)
    prod2 = contract(prod2, conj(transpose(F.vectors)), 2, 1)
    dims = size(prod)
    prod2 = reshape(prod2, (dims[1]*dims[3], dims[2], dims[4]))
    prod2 = moveidx(prod2, 1, 3)
    prod2 = reshape(prod2, (dims[2], dims[4], dims[1], dims[3]))
    prod2 = moveidx(prod2, 3, 1)
    prod2 = moveidx(prod2, 4, 3)
    #sum(abs.(prod-prod2)) > 1e-5 && println(sum(abs.(prod-prod2)))
    #sum(abs.(prod-prod2)) > 1e-5 && println(F.values)
    return real(prod2)
end
