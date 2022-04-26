function fullupdate(psi::GPEPS, H::OpList2d, dt::Number, st::Sitetypes, projectors::Vector{GPEPS} = []; kwargs...)
    # Get convergence arguments
    maxiter = get(kwargs, :maxiter, 10000)
    miniter = get(kwargs, :miniter, 1000)
    tol = get(kwargs, :tol, 1e-6)
    saveiter = get(kwargs, :saveiter, 500)

    # Get psi properties
    maxdim = get(kwargs, :maxdim, maxbonddim(psi))
    N = size(psi)[1]
    chi = get(kwargs, :chi, maxdim^2)
    chieval = get(kwargs, :chi_evaluate, 100)
    chiproj::Int = get(kwargs, :chi_proj, maxdim)
    dropoff::Int = get(kwargs, :dropoff, 0)

    # Create the gates
    gates = trotterize(st, H, dt)

    # Create the environment
    env = Environment(psi, psi; chi=chi, dropoff=dropoff)
    renv = ReducedEnvironment(env; hermitian=true, posdef=true)
    projEnvs = [Environment(proj, psi; chi=chiproj) for proj = projectors]
    projRenvs = [ReducedEnvironment(projEnv) for projEnv in projEnvs]

    # Measure energies
    function calculateenergy(psi)
        evalenv = Environment(psi, psi; chi=chieval)
        normal = inner(evalenv)
        return real(sum(inner(st, evalenv, H) / normal))
    end

    function buildenv!(i::Int, j::Int, dir::Bool, dir2::Bool)
        build!(env, i, j, dir)
        build!(renv, dir2)
        for k = 1:length(projEnvs)
            build!(projEnvs[k], i, j, dir)
            build!(projRenvs[k], dir2)
        end
    end

    # Loop through until convergence
    converge = false
    iter = 0
    lastenergy = calculateenergy(psi)
    energy = 0
    maxchi = 0
    maxcost = 0
    while !converge
        # Update rows
        for i = 1:N
            j = 1
            while j < N
                # Optimize
                buildenv!(i, j, false, false)
                maxchi = max(maxchi, maxbonddim(env))
                gate = getgate(gates, i, j, false)
                if gate != 0
                    cost = optimize(env, gate, i, j, false, projEnvs; chi=chi, kwargs...)
                    maxcost = max(cost, maxcost)
                else
                    println("nope")
                end
                j += 2
            end
            buildenv!(i, N, false, true)

            j = N - 2
            while j > 1
                # Optimize
                buildenv!(i, j+1, false, true)
                maxchi = max(maxchi, maxbonddim(env))
                gate = getgate(gates, i, j, false)
                if gate != 0
                    cost = optimize(env, gate, i, j, false, projEnvs; chi=chi, kwargs...)
                    maxcost = max(cost, maxcost)
                end
                j -= 2
            end
        end

        # Update columns
        for i = 1:N
            j = 1
            while j < N
                # Optimize
                buildenv!(j, i, true, false)
                maxchi = max(maxchi, maxbonddim(env))
                gate = getgate(gates, j, i, true)
                if gate != 0
                    cost = optimize(env, gate, j, i, true, projEnvs; chi=chi, kwargs...)
                    maxcost = max(cost, maxcost)
                end
                j += 2
            end
            buildenv!(N, i, true, true)
            maxchi = max(maxchi, maxbonddim(env))

            j = N - 2
            while j > 1
                # Optimize
                buildenv!(j+1, i, true, true)
                maxchi = max(maxchi, maxbonddim(env))
                gate = getgate(gates, j, i, true)
                if gate != 0
                    cost = optimize(env, gate, j, i, true, projEnvs; chi=chi, kwargs...)
                    maxcost = max(cost, maxcost)
                end
                j -= 2
            end
        end

        # Rescale tensors
        rescale!(psi)

        # Check convergence
        energy = real(energy)
        iter += 1
        converge = (iter >= maxiter && maxiter != 0) ? true : converge
        if iter % saveiter == 0
            buildenv!(1, 1, false, false)
            maxchi = max(maxchi, maxbonddim(env))
            energy = calculateenergy(psi)
            diff = (energy-lastenergy) / (saveiter * dt)
            diff = abs(energy) < 1e-10 ? diff : diff / abs(energy)
            converge = (diff < tol) ? true : converge
            lastenergy = energy
            @printf("iter=%d, energy=%.12f, maxdim=%d, maxchi=%d, cost=%.6e \n", iter, energy, maxbonddim(psi), maxchi, maxcost)
            maxcost = 0
        end
        converge = iter < miniter ? false : converge
    end
    return psi, energy
end

function fullupdate(psi::GPEPS, H::OpList2d, dt::Number, st::Sitetypes; kwargs...)
    return fullupdate(psi, H, dt, st, GPEPS[]; kwargs...)
end

function optimize(env::Environment, gate, site11::Int, site12::Int, dir::Bool, projEnvs::Vector{Environment}; kwargs...)
    # Key arguments
    maxiter::Int = get(kwargs, :update_iterations, 10)
    tol::Float64 = get(kwargs, :update_tol, 1e-10)
    cutoff::Float64 = get(kwargs, :cutoff, 1e-6)
    maxdim = get(kwargs, :maxdim, 1)

    # Find the sites
    site21 = site11 + dir
    site22 = site12 + !dir

    # Find the reduced tensors
    A1, R1 = reducedtensor(env.objects[end], site11, site12, !dir ? 4 : 3)
    A2, R2 = reducedtensor(env.objects[end], site21, site22, !dir ? 1 : 2)

    # Construct the environment for the reduced tensors
    renv = ReducedTensorEnv(env, site11, site12, dir, A1, A2)
    renvProjs = [ReducedTensorSingleEnv(projEnvs[i], site11, site12, dir, A1, A2) for i = 1:length(projEnvs)]

    # Find the full-updated reduced tensors
    fulltensor = contract(R1, R2, 2, 1)
    fulltensor = contract(fulltensor, gate, [2, 4], [2, 4])
    fulltensor = moveidx(fulltensor, 2, 4)

    # Find the norm of the fully updated tensor
    normalFull = contract(renv, conj(fulltensor), [1, 3], [1, 4])
    normalFull = contract(normalFull, fulltensor, [1, 2, 3, 4], [1, 4, 2, 3])[1]

    # Find the cost of the original tensors
    function calculatecost(R1, R2)
        cost = normalFull + norm(renv, R1, R2)
        cost -= 2*overlap(renv, R1, R2, fulltensor)
        for i = 1:length(projEnvs)
            cost += overlapproj(renvProjs[i], R1, R2)
        end
        return real(cost)
    end
    costoriginal = calculatecost(R1, R2)

    # Find initial guesses via svd
    prod, cmb1 = combineidxs(fulltensor, [1, 2])
    prod, cmb1 = combineidxs(prod, [1, 2])
    R1, S, R2 = svd(prod, 2; cutoff=cutoff, maxdim=maxdim)
    R1 = contract(R1, sqrt.(S), 2, 1)
    R2 = contract(sqrt.(S), R2, 2, 1)
    R1 = moveidx(R1, 1, 2)
    R1 = reshape(R1, (size(S)[1], size(fulltensor)[1], size(fulltensor)[2]))
    R1 = moveidx(R1, 1, 2)
    R2 = reshape(R2, (size(S)[2], size(fulltensor)[3], size(fulltensor)[4]))
    R2 = moveidx(R2, 2, 3)

    # Make copies of R1 and R2
    R1svd = deepcopy(R1)
    R2svd = deepcopy(R2)
    costsvd = calculatecost(R1, R2)

    # Do an alternating least squares scheme
    cost = deepcopy(costsvd)
    converge = cost < tol
    converge = (maxdim == 1 && length(projEnvs) == 0) ? true : converge
    iters = 0
    D = size(R1)[2]
    costs = [cost]
    while !converge
        for site = [false, true]
            # More gauge to the tensor which is being updated
            if site
                R1, R = qr(R1, 2)
                R2 = contract(R, R2, 2, 1)
            else
                L, R2 = lq(R2, 1)
                R1 = contract(R1, L, 2, 1)
                R1 = moveidx(R1, 3, 2)
            end

            # Calculate effecitvenorm
            Neff = effectivenorm(renv, R1, R2, site)
            Projeffs = [effectiveproj(renvProjs[i], R1, R2, site) for i = 1:length(projEnvs)]

            # Define the cost matrix
            function normeff(x)
                if site
                    f = contractnorm(Neff, R1, x, site)
                    for i = 1:length(projEnvs)
                        f += contractproj(Projeffs[i], R1, x, site)
                    end
                else
                    f = contractnorm(Neff, x, R2, site)
                    for i = 1:length(projEnvs)
                        f += contractproj(Projeffs[i], x, R2, site)
                    end
                end
                return f
            end

            # Calculate the effective dot
            doteff = effectivedot(renv, site ? R1 : R2, fulltensor, site)

            # Optimize the site
            try
                R, info = linsolve(normeff, doteff, site ? R2 : R1)
                R1 = site ? R1 : R
                R2 = site ? R : R2
            catch e
            end
        end

        # Check convergence
        iters += 1
        oldcost = cost
        cost = calculatecost(R1, R2)
        push!(costs, cost)
        converge = cost < tol
        diff = abs((oldcost-cost) / (oldcost + cost))
        converge = converge ? true : diff < 1e-4
        converge = iters >= maxiter ? true : converge

        # Increase bond dimension
        if converge && (D < maxdim) && (cost > tol)
            D += 1
            R1, R2 = increasedim(R1, R2, D)
            iters = 0
            converge = false
        end
    end

    # Decide to use SVD or ALS
    normal = norm(renv, R1, R2)
    if costsvd < cost || real(normal) < 1e-16
        R1 = R1svd
        R2 = R2svd
        cost = costsvd
        normal = norm(renv, R1, R2)
    end

    # Renormalize tensors
    if costoriginal >= cost && real(normal) > 1e-16
        #println("----")
        #println(costoriginal)
        #println(costsvd)
        #println(cost)
        #println(overlapproj(renvProjs[1], R1, R2))
        #println(normal)
        R1, R = qr(R1, 2)
        L, R2 = lq(R2, 1)
        R = contract(R, L, 2, 1)
        R = R / normal^0.5
        U, S, V = svd(R, 2)
        U = contract(U, sqrt.(S), 2, 1)
        V = contract(sqrt.(S), V, 2, 1)
        R1 = contract(R1, U, 2, 1)
        R1 = moveidx(R1, 3, 2)
        R2 = contract(V, R2, 2, 1)

        # Restore tensor
        if dir
            A1 = contract(A1, R1, 3, 1)
            A1 = moveidx(A1, 4, 3)
            A2 = contract(A2, R2, 2, 2)
            A2 = moveidx(A2, 4, 2)
        else
            A1 = contract(A1, R1, 4, 1)
            A2 = contract(A2, R2, 1, 2)
            A2 = moveidx(A2, 4, 1)
        end

        # Update sites
        env.objects[end][site11, site12] = A1
        env.objects[end][site21, site22] = A2
    else
        cost = costoriginal
    end
    return cost
end


function contractnorm(norm, R1, R2, site::Bool)
    if site == true
        # Second site
        prod = contract(norm, R2, [2, 4], [2, 1])
        prod = moveidx(prod, 1, 2)
    else
        # First site
        prod = contract(R1, norm, [1, 2], [4, 2])
        prod = moveidx(prod, 3, 1)
        prod = moveidx(prod, 3, 2)
    end

    return prod
end

function effectivenorm(renv, R1, R2, site::Bool)
    if site == true
        # Second site
        prod = contract(renv, conj(R1), 1, 1)
        prod = contract(prod, R1, [1, 5], [1, 3])
    else
        # First site
        prod = contract(R2, renv, 2, 4)
        prod = contract(conj(R2), prod, [2, 3], [5, 2])
    end

    return prod
end

function effectiveproj(renv, R1, R2, site::Bool)
    if site == true
        # second site
        prod = contract(renv, R1, [1, 2], [3, 1])
        prod = moveidx(prod, 1, 3)
        prod = moveidx(prod, 1, 2)
    else
        # first site
        prod = contract(R2, renv, [2, 3], [4, 3])
        prod = moveidx(prod, 3, 2)
        prod = moveidx(prod, 1, 2)
    end
    return prod
end

function contractproj(proj, R1, R2, site::Bool)
    prod = contract(proj, site ? R2 : R1, [1, 2, 3], [1, 2, 3])[1]
    prod = prod * conj(proj)
    return prod
end

function overlapproj(renv, R1, R2)
    prod = contract(renv, R1, [1, 2], [3, 1])
    prod = contract(prod, R2, [1, 2, 3], [3, 2, 1])[1]
    return prod*conj(prod)
end

function inversenorm(renv, R1, R2, site::Bool)
    Dprime1, D1, d = size(R1)
    D2, Dprime2, d = size(R2)

    # Make into matrix
    prod, cmb1 = combineidxs(renv, [1, 3])
    prod, cmb1 = combineidxs(prod, [1, 2])

    #prod = inv(prod)

    # Find the decomposition
    F = eigen(prod)
    vals = real(F.values)
    vals = [val/maximum(vals) > 0.0 ? 1/val : 0.0 for val in vals]
    prod = contract(F.vectors, diagm(vals), 2, 1)
    prod = contract(prod, conj(transpose(F.vectors)), 2, 1)

    # Reshape into the correct form
    if site == true
        prod = reshape(prod, (Dprime2*D2, Dprime2, D2))
        prod = moveidx(prod, 1, 3)
        prod = reshape(prod, (Dprime2, D2, Dprime2, D2))
    else
        prod = reshape(prod, (D1*Dprime1, D1, Dprime1))
        prod = moveidx(prod, 1, 3)
        prod = reshape(prod, (D1, Dprime1, D1, Dprime1))
    end
    prod = moveidx(prod, 2, 1)
    prod = moveidx(prod, 4, 3)
    return prod
end

function effectivedot(renv, R, fulltensor, site::Bool)
    if site == true
        # Second site
        prod = contract(renv, conj(R), 1, 1)
        prod = contract(prod, fulltensor, [1, 3, 5], [1, 4, 2])
        prod = moveidx(prod, 2, 1)
    else
        # First site
        prod = contract(renv, conj(R), 3, 2)
        prod = contract(prod, fulltensor, [2, 3, 5], [1, 4, 3])
    end

    return prod
end

function overlap(renv, R1, R2, fulltensor)
    prod = contract(renv, conj(R1), 1, 1)
    prod = contract(prod, conj(R2), [2, 4], [2, 1])
    prod = contract(prod, fulltensor, [1, 2, 3, 4], [1, 4, 2, 3])
    return prod[1]
end

function norm(renv, R1, R2)
    prod = contract(renv, conj(R1), 1, 1)
    prod = contract(prod, R1, [1, 5], [1, 3])
    prod = contract(prod, conj(R2), [1, 3], [2, 1])
    prod = contract(prod, R2, [1, 2, 3], [2, 1, 3])
    return prod[1]
end

function increasedim(R1, R2, dim)
    dims1 = size(R1)
    dims2 = size(R2)

    F1 = zeros(ComplexF64, (dims1[1], dim, dims1[3])) + 1e-4*randn(Float64, dims1[1], dim, dims1[3])
    F2 = zeros(ComplexF64, (dim, dims2[2], dims2[3])) + 1e-4*randn(Float64, dim, dims2[2], dims2[3])
    F1[:, 1:dims1[2], :] = R1
    F2[1:dims2[1], :, :] = R2

    return F1, F2
end
