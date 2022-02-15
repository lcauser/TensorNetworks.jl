mutable struct Environment
    objects::Vector{GPEPS}
    MPS::Vector{GMPS}
    blocks::Vector{Array{ComplexF64}}
    direction::Bool
    center::Int
    centerMPS::Int
    chi::Int
    dropoff::Int
end

function Environment(args::GPEPS...; kwargs...)
    # Check the GPEPS share the same properies
    checks = false
    checks = sum([size(arg)[2]!=size(args[1])[2] for arg in args]) > 0 ? true : checks
    checks = sum([size(arg)[1]!=size(args[1])[1] for arg in args]) > 0 ? true : checks
    checks = sum([dim(arg)!=dim(args[1]) for arg in args]) > 0 ? true : checks
    checks && error("GPEPS must have the same properties.")

    # Check the inner product is of correct form
    length(args) < 2 && error("The environment must construct an inner product.")
    (rank(args[1]) != 1 || rank(args[end]) != 1) && error("The inner product must start and end with a GPEPS of rank 1.")
    sum([rank(args[i])!=2 for i=2:length(args)-1])>0 && error("The inner GPEPS must be of rank 2.")

    # Get the truncation criteria
    chi::Int = get(kwargs, :chi, 0)
    if chi == 0
        chi = 1
        for i = 1:length(args)
            chi *= maxbonddim(args[i])
        end
    end
    dropoff::Int = get(kwargs, :dropoff, 0)

    # Build up the environment
    direction::Bool = get(kwargs, :direction, false)
    center::Int = get(kwargs, :center, 1)
    centerMPS::Int = get(kwargs, :centerMPS, 1)

    # Create environment blocks
    nummps = direction ? size(args[1])[2] : size(args[1])[1]
    numblocks = direction ? size(args[1])[1] : size(args[1])[2]
    MPSblocks = GMPS[GMPS(length(args), 1, numblocks) for i = 1:nummps]
    blocks = Any[ones(ComplexF64, [1 for j=1:2+length(args)]...) for i = 1:numblocks]
    env = Environment(collect(args), MPSblocks, blocks, direction, 0, 0, chi, dropoff)
    build!(env, center, centerMPS, direction)
    return env
end

size(env::Environment) = size(env.objects[1])

function blocksizes(env::Environment)
    N1 = env.direction ? size(env)[2] : size(env)[1]
    N2 = env.direction ? size(env)[1] : size(env)[2]

    return N1, N2
end


function block(env::Environment, idx::Int)
    N1, N2 = blocksizes(env)
    if idx < 1 || idx > N1
        return bGMPSOnes(length(env.objects), N2)
    else
        return env.MPS[idx]
    end
end

function MPSblock(env::Environment, idx::Int)
    N1, N2 = blocksizes(env)
    if idx < 1 || idx > N2
        return ones(ComplexF64, [1 for i = 1:2+length(env.objects)]...)
    else
        return env.blocks[idx]
    end
end

function findsite(env::Environment)
    if !env.direction
        site1 = env.center
        site2 = env.centerMPS
    else
        site1 = env.centerMPS
        site2 = env.center
    end
    return site1, site2
end

### Build the environment
function buildup!(env::Environment, idx::Int)
    # Find the size
    N1, N2 = size(env)

    # Fetch the upwards bond dimensions
    bonddims = [Int[] for i = 1:length(env.objects)]
    for i = 1:length(env.objects)
        for j = 1:N2
            push!(bonddims[i], size(env.objects[i][idx, j])[2])
        end
    end

    # Create a random bMPO
    bMPO = bGMPS(env.chi, bonddims...)

    # Apply variational sweeps to limit the bond dimension
    proj = ProjbMPS(bMPO, env, true, idx)
    bMPO = vmps(bMPO, proj; maxdim=env.chi, nsites=1)

    # Save the block
    env.MPS[idx] = bMPO
end


function builddown!(env::Environment, idx::Int)
    # Find the size
    N1, N2 = size(env)

    # Fetch the upwards bond dimensions
    bonddims = [Int[] for i = 1:length(env.objects)]
    for i = 1:length(env.objects)
        for j = 1:N2
            push!(bonddims[i], size(env.objects[i][idx, j])[3])
        end
    end

    # Create a random bMPO
    bMPO = bGMPS(env.chi, bonddims...)

    # Apply variational sweeps to limit the bond dimension
    proj = ProjbMPS(bMPO, env, false, idx)
    bMPO = vmps(bMPO, proj; maxdim=env.chi, nsites=1)

    # Save the block
    env.MPS[idx] = bMPO
end


function buildright!(env::Environment, idx::Int)
    # Find the size
    N1, N2 = size(env)

    # Fetch the upwards bond dimensions
    bonddims = [Int[] for i = 1:length(env.objects)]
    for i = 1:length(env.objects)
        for j = 1:N1
            push!(bonddims[i], size(env.objects[i][j, idx])[4])
        end
    end

    # Create a random bMPO
    bMPO = bGMPS(env.chi, bonddims...)

    # Apply variational sweeps to limit the bond dimension
    proj = ProjbMPS(bMPO, env, false, idx)
    bMPO = vmps(bMPO, proj; maxdim=env.chi, nsites=1)

    # Save the block
    env.MPS[idx] = bMPO
end


function buildleft!(env::Environment, idx::Int)
    # Find the size
    N1, N2 = size(env)

    # Fetch the upwards bond dimensions
    bonddims = [Int[] for i = 1:length(env.objects)]
    for i = 1:length(env.objects)
        for j = 1:N1
            push!(bonddims[i], size(env.objects[i][j, idx])[1])
        end
    end

    # Create a random bMPO
    bMPO = bGMPS(env.chi, bonddims...)

    # Apply variational sweeps to limit the bond dimension
    proj = ProjbMPS(bMPO, env, true, idx)
    bMPO = vmps(bMPO, proj; maxdim=env.chi, nsites=1)

    # Save the block
    env.MPS[idx] = bMPO
end


function buildmpsright!(env::Environment, idx::Int)
    # Fetch left block
    left = MPSblock(env, idx-1)

    # Determine sites
    site1 = env.direction ? idx : env.center
    site2 = env.direction ? env.center : idx

    # Fetch new blocks
    A1 = block(env, env.center-1)[idx]
    A2 = block(env, env.center+1)[idx]
    M1 = conj(env.objects[1][site1, site2])
    M2 = env.objects[end][site1, site2]

    # Determine rank
    r = length(env.objects)

    # Contract them to grow the block
    prod = contract(left, A1, 1, 1)
    if env.direction == false
        prod = contract(prod, M1, [1, 2+r], [1, 2])
        for i = 1:r-2
            M = env.objects[1+i][site1, site2]
            prod = contract(prod, M, [1, 2+r-i, length(size(prod))], [1, 2, 5])
        end
        prod = contract(prod, M2, [1, 3, length(size(prod))], [1, 2, 5])
        prod = contract(prod, A2, [2*i-1 for i = 1:1+r], [i for i = 1:1+r])
    else
        prod = contract(prod, M1, [1, 2+r], [2, 1])
        for i = 1:r-2
            M = env.objects[1+i][site1, site2]
            prod = contract(prod, M, [1, 2+r-i, length(size(prod))], [2, 1, 5])
        end
        prod = contract(prod, M2, [1, 3, length(size(prod))], [2, 1, 5])
        prod = contract(prod, A2, [1, [2+2*i for i = 1:r]...], [i for i = 1:1+r])
    end

    # Store the block
    env.blocks[idx] = prod
end


function buildmpsleft!(env::Environment, idx::Int)
    # Fetch right block
    right = MPSblock(env, idx+1)

    # Determine sites
    site1 = env.direction ? idx : env.center
    site2 = env.direction ? env.center : idx

    # Fetch new blocks
    A1 = block(env, env.center-1)[idx]
    A2 = block(env, env.center+1)[idx]
    M1 = conj(env.objects[1][site1, site2])
    M2 = env.objects[end][site1, site2]

    # Determine rank
    r = length(env.objects)

    # Contract them to grow the block
    prod = contract(A2, right, 2+r, 2+r)
    if env.direction == false
        prod = contract(M2, prod, [3, 4], [1+r, 2+2*r])
        for i = 1:r-2
            M = env.objects[1+i][site1, site2]
            prod = contract(M, prod, [3, 4, 6], [length(size(prod))-(r+1-i), length(size(prod)), 3])
        end
        prod = contract(M1, prod, [3, 4, 5], [length(size(prod))-2, length(size(prod)), 3])
        prod = contract(A1, prod, [1+i for i=1:1+r], [2*i for i=1:1+r])
    else
        prod = contract(M2, prod, [4, 3], [1+r, 2+2*r])
        for i = 1:r-2
            M = env.objects[1+i][site1, site2]
            prod = contract(M, prod, [4, 3, 6], [length(size(prod))-(r+1-i), length(size(prod)), 3])
        end
        prod = contract(M1, prod, [4, 3, 5], [length(size(prod))-2, length(size(prod)), 3])
        prod = contract(A1, prod, [1+i for i=1:1+r], [[2*i-1 for i = 1:r]..., length(size(prod))])
    end

    # Store the block
    env.blocks[idx] = prod
end


function build!(env::Environment, idx1::Int, idx2::Int, direction::Bool=false)
    # Check for a change in direction for which the environment is built
    if direction != env.direction
        env.direction = direction
        nummps = direction ? size(env.objects[1])[2] : size(env.objects[1])[1]
        numblocks = direction ? size(env.objects[1])[1] : size(env.objects[1])[2]
        env.MPS = GMPS[GMPS(length(env.objects), 1, numblocks) for i = 1:nummps]
        env.center = 0
        env.blocks = Any[ones(ComplexF64, [1 for j=1:2+length(env.objects)]...) for i = 1:numblocks]
        env.centerMPS = 0
    end

    # Build the blocks
    rebuild = true # Book keeping term for rebuilding MPS blocks
    if direction == false
        if env.center == 0
            # The center isn't set; build from left and right
            for i = 1:idx1-1
                builddown!(env, i)
            end
            for i = 1:size(env.objects[1])[1] - idx1
                buildup!(env, size(env.objects[1])[1] + 1 - i)
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
            for i = 1:size(env.objects[1])[1] - idx2
                buildleft!(env, size(env.objects[1])[1] + 1 - i)
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
        env.blocks = [ones(ComplexF64, 1, 1, 1, 1) for i = 1:size(env.objects[1])[1]]
        env.centerMPS = 0
    end

    # Decide which idx to build MPS blocks too
    idx = env.direction ? idx1 : idx2

    # Build MPS-like blocks
    if env.centerMPS == 0
        # Build from left and right
        for i = 1:idx-1
            buildmpsright!(env, i)
        end
        for i = 1:size(env.objects[1])[1]-idx
            buildmpsleft!(env, size(env.objects[1])[1]+1-i)
        end
    elseif env.centerMPS < idx
        # Build from the left
        for i = 1:idx-env.centerMPS
            buildmpsright!(env, env.centerMPS-1+i)
        end
    elseif env.centerMPS > idx
        # Build from the right
        for i = 1:env.centerMPS-idx
            buildmpsleft!(env, env.centerMPS + 1 - i)
        end
    end
    env.centerMPS = idx
end

function maxbonddim(env::Environment)
    chi = 0
    for b in env.MPS
        chi = max(chi, maxbonddim(b))
    end
    return chi
end


function inner(env::Environment, ops::Vector{Array{ComplexF64, 2}},
               sites::Vector{Int}, direction::Bool=false)
    # Build the environment to the correct site
    build!(env, sites[1], sites[2], direction)

    # Fetch the left and right blocks
    prod = MPSblock(env, env.centerMPS-1)
    right = MPSblock(env, env.centerMPS+length(ops))

    # Loop through each site in the operator
    for i = 1:length(ops)
        # Fetch relevent tensors
        A = !env.direction ? env.objects[1][env.center, env.centerMPS+i-1] : env.objects[1][env.centerMPS+i-1, env.center]
        B = !env.direction ? env.objects[end][env.center, env.centerMPS+i-1] : env.objects[end][env.centerMPS+i-1, env.center]

        M1 = block(env, env.center-1)[env.centerMPS-1+i]
        M2 = block(env, env.center+1)[env.centerMPS-1+i]

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
    prod = contract(prod, right, [1, 2, 3, 4], [1, 2, 3, 4])
    return prod[1]
end

function inner(env::Environment, st::Sitetypes, ops::Vector{String},
               sites::Vector{Int}, direction::Bool=false)
    # Calculate the operator list
    newops = [op(st, ops[i]) for i = 1:length(ops)]
    return inner(env, newops, sites, direction)
end

function inner(st::Sitetypes, psi::GPEPS, ops::OpList2d, phi::GPEPS; kwargs...)
    # Calculate environment
    env = Environment(psi, phi; kwargs...)

    return inner(st, env, ops; kwargs...)
end

function inner(st::Sitetypes, env::Environment, ops::OpList2d; kwargs...)

    expectations = ComplexF64[0 for i = 1:length(ops.sites)]
    # Loop through each direction
    for direction = [false, true]
        # Loop through each site
        for center = 1:size(env.objects[1])[1]
            for centerMPS = 1:size(env.objects[1])[1]
                site1 = !direction ? center : centerMPS
                site2 = !direction ? centerMPS : center
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
    site1 = !env.direction ? env.center : env.centerMPS
    site2 = !env.direction ? env.centerMPS : env.center
    prod = MPSblock(env, env.centerMPS-1)
    right = MPSblock(env, env.centerMPS+1)
    A1 = block(env, env.center-1)[env.centerMPS]
    A2 = block(env, env.center+1)[env.centerMPS]
    M1 = conj(env.objects[1][site1, site2])
    M2 = env.objects[end][site1, site2]

    # Determine rank
    r = length(env.objects)

    # Contract
    prod = contract(prod, A1, 1, 1)
    if !env.direction
        prod = contract(prod, M1, [1, 2+r], [1, 2])
        for i = 1:r-2
            M = env.objects[1+i][site1, site2]
            prod = contract(prod, M, [1, 2+r-i, length(size(prod))], [1, 2, 5])
        end
        prod = contract(prod, M2, [1, 3, length(size(prod))], [1, 2, 5])
        prod = contract(prod, A2, [2*i-1 for i = 1:1+r], [i for i = 1:1+r])
    else
        prod = contract(prod, M1, [1, 2+r], [2, 1])
        for i = 1:r-2
            M = env.objects[1+i][site1, site2]
            prod = contract(prod, M, [1, 2+r-i, length(size(prod))], [2, 1, 5])
        end
        prod = contract(prod, M2, [1, 3, length(size(prod))], [2, 1, 5])
        prod = contract(prod, A2, [1, [2+2*i for i = 1:r]...], [i for i = 1:1+r])
    end
    prod = contract(prod, right, [i for i=1:2+r], [i for i=1:2+r])[1]
    return prod
end

function inner(objects::GPEPS...; kwargs...)
    env = Environment(objects; kwargs...)
    return inner(env)
end


function inner(env::Environment, ops::Vector{Vector{Array{ComplexF64, 2}}}, coeffs::Vector{Number})
    # Loop through each direction
    prod = 0
    for direction = [false, true]
        # Loop through each site
        for center = 1:size(env.objects[1])[1]
            for centerMPS = 1:size(env.objects[1])[1]
                site1 = !direction ? center : centerMPS
                site2 = !direction ? centerMPS : center
                sites = [site1, site2]
                for i = 1:length(ops)
                    if centerMPS + length(ops[i])-1 <= size(env.objects[1])[1]
                        prod += coeffs[i]*inner(env, ops[i], sites, direction)
                    end
                end
            end
        end
    end
    return prod
end

function inner(psi::GPEPS, phi::GPEPS, ops::Vector{Vector{Array{ComplexF64, 2}}},
               coeffs::Vector{Number}; kwargs...)
    env = Environement(psi, phi; kwargs...)
    return inner(env, ops, coeffs)
end

### Reduced tensors environment
function ReducedTensorEnv(env::Environment, site1, site2, dir, A1, A2)
    # Determine which centers
    if !dir
        center = site1
        centerMPS = site2
    else
        center = site2
        centerMPS = site1
    end

    # Retrieve relevent MPS blocks and tensors
    left = MPSblock(env, centerMPS-1)
    right = MPSblock(env, centerMPS+2)
    M1 = block(env, center-1)[centerMPS]
    M2 = block(env, center+1)[centerMPS]
    M3 = block(env, center-1)[centerMPS+1]
    M4 = block(env, center+1)[centerMPS+1]

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

function ReducedTensorSingleEnv(env::Environment, site1, site2, dir, A1, A2)
    # Determine which centers and the sites
    if !dir
        center = site1
        centerMPS = site2
        site11 = site1
        site12 = site2
        site21 = site1
        site22 = site2 + 1
    else
        center = site2
        centerMPS = site1
        site11 = site1
        site12 = site2
        site21 = site1 + 1
        site22 = site2
    end

    # Retrieve relevent MPS blocks and tensors
    left = MPSblock(env, centerMPS-1)
    right = MPSblock(env, centerMPS+2)
    M1 = block(env, center-1)[centerMPS]
    M2 = block(env, center+1)[centerMPS]
    M3 = block(env, center-1)[centerMPS+1]
    M4 = block(env, center+1)[centerMPS+1]
    B1 = env.psi[site11, site12]
    B2 = env.psi[site21, site22]

    # Contract to reduced tensor environment
    if !dir
        # Grow the left block
        left = contract(left, M1, 1, 1)
        left = contract(left, conj(B1), [1, 4], [1, 2])
        left = contract(left, A1, [1, 3], [1, 2])
        left = contract(left, M2, [1, 3, 6], [1, 2, 3])

        # Grow the right block
        right = contract(M4, right, 4, 4)
        right = contract(A2, right, [3, 4], [3, 6])
        right = contract(conj(B2), right, [3, 4], [4, 6])
        right = contract(M3, right, [2, 3, 4], [2, 5, 7])
    else
        # Grow the left block
        left = contract(left, M1, 1, 1)
        left = contract(left, conj(B1), [1, 4], [2, 1])
        left = contract(left, A1, [1, 3], [2, 1])
        left = contract(left, M2, [1, 4, 7], [1, 2, 3])

        # Grow the right block
        right = contract(M4, right, 4, 4)
        right = contract(A2, right, [3, 4], [6, 3])
        right = contract(conj(B2), right, [3, 4], [6, 4])
        right = contract(M3, right, [2, 3, 4], [1, 4, 7])
    end
    prod = contract(left, right, [1, 2, 5], [1, 2, 5])


    return prod
end
