function fullupdate(psi::PEPS, gate, dt::Real, st::Sitetypes, ops, coeffs; kwargs...)
    # Get convergence arguments
    maxiter = get(kwargs, :maxiter, 0)
    miniter = get(kwargs, :miniter, 2)
    tol = get(kwargs, :tol, 1e-8)
    saveiter = get(kwargs, :saveiter, 10)

    # Get psi properties
    maxdim = get(kwargs, :maxdim, maxbonddim(psi))
    N = length(psi)

    # Create the gate
    gate = exp(dt*creategate(st, ops, coeffs), [2, 4])

    # Create the environment
    env = Environment(psi, psi; kwargs...)

    # Measure energies
    ops = [[op(st, name) for name in op1] for op1 in ops]
    function calculateenergy(psi)
        return real(inner(env, ops, coeffs) / inner(env))
    end

    # Loop through until convergence
    converge = false
    iter = 0
    lastenergy = calculateenergy(psi)
    energy = 0
    maxchi = 0
    while !converge
        # Keep track of change in norm for energy
        energy = 0
        maxcost = 0

        # Update rows
        for i = 1:N
            j = 1
            while j < N
                #println(i, ", ", j)
                # Optimize
                build!(env, i, j, false)
                normal, cost = optimize(env, gate, i, j, false; kwargs...)
                energy += log(normal)/dt
                maxcost = max(cost, maxcost)
                j += 2
            end
            build!(env, i, N, false)

            j = N - 2
            while j > 1
                #println(i, ", ", j)
                # Optimize
                build!(env, i, j+1, false)
                normal, cost = optimize(env, gate, i, j, false; kwargs...)
                energy += log(normal)/dt
                maxcost = max(cost, maxcost)
                j -= 2
            end
            maxchi = max(maxchi, maxbonddim(env))
        end

        # Update columns
        for i = 1:N
            j = 1
            while j < N
                #println(j, ", ", i)
                # Optimize
                build!(env, j, i, true)
                normal, cost = optimize(env, gate, j, i, true; kwargs...)
                energy += log(normal)/dt
                maxcost = max(cost, maxcost)
                j += 2
            end
            build!(env, N, i, true)

            j = N - 2
            while j > 1
                #println(j, ", ", i)
                # Optimize
                build!(env, j+1, i, true)
                normal, cost = optimize(env, gate, j, i, true; kwargs...)
                energy += log(normal)/dt
                maxcost = max(cost, maxcost)
                j -= 2
            end
            maxchi = max(maxchi, maxbonddim(env))
        end

        # Rescale tensors
        rescale!(psi)

        # Check convergence
        energy = real(energy)
        iter += 1
        converge = (iter >= maxiter && maxiter != 0) ? true : converge
        if iter % saveiter == 0
            build!(env, 1, 1, false)
            energy = calculateenergy(psi)
            converge = ((energy-lastenergy) / abs(energy) < tol) ? true : converge
            lastenergy = energy
        @printf("iter=%d, energy=%.12f, maxdim=%d, maxchi=%d, cost=%.12f \n", iter, energy, maxbonddim(psi), maxchi, maxcost)
        end
        converge = iter < miniter ? false : converge
    end
    return psi, energy
end


function optimize(env::Environment, gate, site11, site12, dir; kwargs...)
    # Find the sites
    site21 = site11 + dir
    site22 = site12 + !dir

    # Retrieve relevent tensors
    A1 = env.psi[site11, site12]
    A2 = env.psi[site21, site22]

    # Find the full-updated tensor
    A1full, A2full = updatetensor(A1, A2, gate, dir)

    # Find a truncated initial guess
    maxdim = get(kwargs, :maxdim, size(A1)[!dir ? 4 : 3])
    if !dir
        A1 = A1full[:, :, :, 1:min(maxdim, size(A1full)[4]), :]
        A2 = A2full[1:min(maxdim, size(A2full)[1]), :, :, :, :]
    else
        A1 = A1full[:, :, 1:min(maxdim, size(A1full)[3]), :, :]
        A2 = A2full[:, 1:min(maxdim, size(A2full)[2]), :, :, :]
    end

    # Find the norm of the fully updated tensor
    fullnormal = calculatenorm(env, site11, site12, dir, A1full, A2full)

    # Make a cost function and evaluate the initial cost
    function calculatecost(A1, A2)
        normal = calculatenorm(env, site11, site12, dir, A1, A2)
        overlap = calculateoverlap(env, site11, site12, dir, A1full, A2full, A1, A2)
        return real(fullnormal + normal - overlap - conj(overlap))
    end
    cost = calculatecost(A1, A2)

    # Do an alternating least squares scheme
    converge = cost < 1e-16
    iters = 0
    while !converge
        for site = [false, true]
            # Calculate effecitve dot and define effective norm
            doteff = partialoverlap(env, site11, site12, dir, site, A1full, A2full, A1, A2)
            function normeff(x)
                return partialnorm(env, site11, site12, dir, site, site ? A1 : x, site ? x : A2)
            end

            # Optimize the site
            A, info = linsolve(normeff, doteff, site ? A2 : A1)
            A2 = site ? A : A2
            A1 = site ? A1 : A
        end
        iters += 1
        oldcost = cost
        cost = calculatecost(A1, A2)
        converge = cost < 1e-10
        converge = converge ? true : abs((cost-oldcost) / cost) < 1e-5
        converge = iters >= 10000 ? true : converge
        iters == 10000 && println("yes")
    end

    # Renormalize tensors
    normal = calculatenorm(env, site11, site12, dir, A1, A2)
    A1 = A1 / normal^0.25
    A2 = A2 / normal^0.25

    # Update sites
    psi[site11, site12] = A1
    psi[site21, site22] = A2
    #println("-------")
    #println(site11, " ", site12, " ", site21, " ", site22)
    #println(normalFull)
    #println(normal)
    #println(cost)
    return normal^0.5, cost
end


function updatetensor(A1, A2, gate, dir)
    # Get the original dimensions
    dims1 = size(A1)
    dims2 = size(A2)

    # Contract to make full tensors
    if !dir
        fulltensor = contract(A1, A2, 4, 1)
    else
        fulltensor = contract(A1, A2, 3, 2)
    end
    fulltensor = contract(fulltensor, gate, [4, 8], [2, 4])
    fulltensor = moveidx(fulltensor, 7, 4)

    # Apply a SVD to split into two tensors
    fulltensor, cmb1 = combineidxs(fulltensor, [1, 2, 3, 4])
    fulltensor, cmb1 = combineidxs(fulltensor, [1, 2, 3, 4])
    A1full, S, A2full = svd(fulltensor, 2; cutoff=1e-16)
    A1full = contract(A1full, sqrt.(S), 2, 1)
    A2full = contract(sqrt.(S), A2full, 2, 1)

    # Reshape into the correct forms
    A1full = moveidx(A1full, 1, 2)
    if !dir
        A1full = reshape(A1full, (size(S)[1], dims1[1], dims1[2], dims1[3], dims1[5]))
        A1full = moveidx(A1full, 1, 4)
        A2full = reshape(A2full, (size(S)[1], dims2[2], dims2[3], dims2[4], dims2[5]))
    else
        A1full = reshape(A1full, (size(S)[1], dims1[1], dims1[2], dims1[4], dims1[5]))
        A1full = moveidx(A1full, 1, 3)
        A2full = reshape(A2full, (size(S)[1], dims2[1], dims2[3], dims2[4], dims2[5]))
        A2full = moveidx(A2full, 1, 2)
    end

    return A1full, A2full
end


function expandleft(left, M1, M2, A1, A2, dir)
    # Contract
    left = contract(left, M1, 1, 1)
    if !dir
        left = contract(left, conj(A1), [1, 4], [1, 2])
        left = contract(left, A2, [1, 3, 7], [1, 2, 5])
        left = contract(left, M2, [1, 3, 5], [1, 2, 3])
    else
        left = contract(left, conj(A1), [1, 4], [2, 1])
        left = contract(left, A2, [1, 3, 7], [2, 1, 5])
        left = contract(left, M2, [1, 4, 6], [1, 2, 3])
    end
end


function expandright(right, M1, M2, A1, A2, dir)
    right = contract(M2, right, 4, 4)
    if !dir
        right = contract(A2, right, [3, 4], [3, 6])
        right = contract(conj(A1), right, [3, 4, 5], [5, 7, 3])
        right = contract(M1, right, [2, 3, 4], [2, 4, 6])
    else
        right = contract(A2, right, [3, 4], [6, 3])
        right = contract(conj(A1), right, [3, 4, 5], [7, 5, 3])
        right = contract(M1, right, [2, 3, 4], [1, 3, 6])
    end
end


function calculateoverlap(env, site1, site2, dir, A1full, A2full, A1, A2)
    # Determine where the centers are
    center = !dir ? site1 : site2
    center2 = !dir ? site2 : site1

    # Fetch the relevent blocks
    left = block(env, center-1)
    right = block(env, center+1)
    leftMPS = block2(env, center2-1)
    rightMPS = block2(env, center2+2)
    As = [A1, A2]
    Afulls = [A1full, A2full]

    # Loop through both sites and grow the block
    prod = leftMPS
    for i = 1:2
        A = As[i]
        Afull = Afulls[i]
        M1 = left[center2-1+i]
        M2 = right[center2-1+i]

        # Contract
        prod = expandleft(prod, M1, M2, Afull, A, dir)
    end

    # Contract with right blocks
    prod = contract(prod, rightMPS, [1, 2, 3, 4], [1, 2, 3, 4])[1]

    return prod
end
calculatenorm(env, site1, site2, dir, A1, A2) = calculateoverlap(env, site1, site2, dir, A1, A2, A1, A2)

function partialoverlap(env, site1, site2, dir, site, A1full, A2full, A1, A2)
    # Determine where the centers are
    center = !dir ? site1 : site2
    center2 = !dir ? site2 : site1

    # Fetch the relevent blocks
    left = block(env, center-1)
    right = block(env, center+1)
    leftMPS = block2(env, center2-1)
    rightMPS = block2(env, center2+2)

    # Grow the left or right block
    if site
        leftMPS = expandleft(leftMPS, left[center2], right[center2], A1full, A1, dir)
    else
        rightMPS = expandright(rightMPS, left[center2+1], right[center2+1], A2full, A2, dir)
    end

    # Get the middle blocks
    Afull = site ? A2full : A1full
    M1 = site ? left[center2+1] : left[center2]
    M2 = site ? right[center2+1] : right[center2]

    # Contract left with middle and right
    prod = contract(leftMPS, M1, 1, 1)
    if !dir
        prod = contract(prod, conj(Afull), [1, 4], [1, 2])
        prod = contract(prod, M2, [2, 5], [1, 2])
        prod = contract(prod, rightMPS, [3, 4, 7], [1, 2, 4])
        prod = moveidx(prod, 3, 5)
    else
        prod = contract(prod, conj(Afull), [1, 4], [2, 1])
        prod = contract(prod, M2, [2, 6], [1, 2])
        prod = contract(prod, rightMPS, [3, 4, 7], [1, 2, 4])
        prod = moveidx(prod, 2, 1)
        prod = moveidx(prod, 3, 5)
        prod = moveidx(prod, 4, 3)
    end

    return prod
end
partialnorm(env, site1, site2, dir, site, A1, A2) = partialoverlap(env, site1, site2, dir, site, A1, A2, A1, A2)
