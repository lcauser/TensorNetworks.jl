function fullupdate(psi::PEPS, gate, dt::Real, energylist::OpList2d, st::Sitetypes; kwargs...)
    # Get convergence arguments
    maxiter = get(kwargs, :maxiter, 0)
    miniter = get(kwargs, :miniter, 2)
    tol = get(kwargs, :tol, 1e-7)
    saveiter = get(kwargs, :saveiter, 1)

    # Get psi properties
    maxdim = get(kwargs, :maxdim, maxbonddim(psi))
    N = length(psi)

    # Exponentiate the gate
    gate = exp(dt*gate, [2, 4])

    # Create the environment
    env = Environment(psi, psi; kwargs...)

    # Loop through until convergence
    converge = false
    iter = 0
    lastenergy = 0
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
        if iter >= miniter
            converge = (iter >= maxiter && maxiter != 0) ? true : converge
        end
        if iter % saveiter == 0 && iter > miniter
            build!(env, 1, 1, false)
            #energy = calculateenergy(psi, st, energylist)
            converge = ((energy-lastenergy) / abs(energy) < tol) ? true : converge
            lastenergy = energy
        end
        @printf("iter=%d, energy=%.12f, maxdim=%d, maxchi=%d, cost=%.12f \n", iter, energy, maxbonddim(psi), maxchi, maxcost)
    end
    #energy = calculateenergy(psi, st, energylist)
    return psi, energy
end


function optimize(env::Environment, gate, site11, site12, dir; kwargs...)
    # Find the sites
    site21 = site11 + dir
    site22 = site12 + !dir
    #println(site11, site12, site21, site22)

    #Retrieve relevent tensors
    A1 = env.psi[site11, site12]
    A2 = env.psi[site21, site22]

    # Find the reduced tensors
    A1, R1 = reducedtensor(env.psi, site11, site12, !dir ? 4 : 3)
    A2, R2 = reducedtensor(env.psi, site21, site22, !dir ? 1 : 2)

    # Construct the environment for the reduced tensors
    renv = ReducedTensorEnv(env, site11, site12, dir, A1, A2)

    ### Make the norm better conditioned

    # Find the full-updated reduced tensors and initial guess
    fulltensor = contract(R1, R2, 2, 1)
    fulltensor = contract(fulltensor, gate, [2, 4], [2, 4])
    fulltensor = moveidx(fulltensor, 2, 4)
    prod, cmb1 = combineidxs(fulltensor, [1, 2])
    prod, cmb1 = combineidxs(prod, [1, 2])
    #R1, S, R2 = svd(prod, 2; kwargs...)
    R1, S, R2 = svd(prod, 2; cutoff=1e-16, kwargs...)
    R1 = contract(R1, sqrt.(S), 2, 1)
    R1 = moveidx(R1, 1, 2)
    R1 = reshape(R1, (size(S)[1], size(fulltensor)[1], size(fulltensor)[2]))
    R1 = moveidx(R1, 1, 2)
    R2 = contract(sqrt.(S), R2, 2, 1)
    R2 = reshape(R2, (size(S)[2], size(fulltensor)[3], size(fulltensor)[4]))
    R2 = moveidx(R2, 2, 3)

    # Find the norm of the fully updated tensor
    normalFull = contract(renv, conj(fulltensor), [1, 3], [1, 4])
    normalFull = contract(normalFull, fulltensor, [1, 2, 3, 4], [1, 4, 2, 3])[1]

    # Make a cost function and evaluate the initial cost
    calculatecost(R1, R2) = abs(normalFull + norm(renv, R1, R2) - 2*overlap(renv, R1, R2, fulltensor))
    cost = calculatecost(R1, R2)

    # Do an alternating least squares scheme
    converge = cost < 1e-16
    iters = 0
    while !converge
        for site = [false, true]
            # Calculate effecitve dot
            normeff(x) = site ? effectivenorm(renv, R1, x, site) : effectivenorm(renv, x, R2, site)
            doteff = effectivedot(renv, site ? R1 : R2, fulltensor, site)

            # Optimize the site
            R, info = linsolve(normeff, doteff, site ? R2 : R1)
            R1 = site ? R1 : R
            R2 = site ? R : R2
        end
        iters += 1
        oldcost = cost
        cost = calculatecost(R1, R2)
        #println(cost)
        converge = cost < 1e-10
        converge = converge ? true : abs((cost-oldcost) / cost) < 1e-5
        converge = iters >= 10000 ? true : converge
        iters == 10000 && println("yes")
    end

    # Renormalize tensors
    normal = norm(renv, R1, R2)
    R1 = R1 / normal^0.25
    R2 = R2 / normal^0.25

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
    psi[site11, site12] = A1
    psi[site21, site22] = A2
    #println("-------")
    #println(site11, " ", site12, " ", site21, " ", site22)
    #println(normalFull)
    #println(normal)
    #println(cost)

    return normal^0.5, cost
end


function effectivenorm(renv, R1, R2, site::Bool)
    if site == true
        # Second site
        prod = contract(renv, conj(R1), 1, 1)
        prod = contract(prod, R1, 1, 1)
        prod = trace(prod, 4, 6)
        prod = contract(prod, R2, 2, 2)
        prod = trace(prod, 3, 4)
        prod = moveidx(prod, 1, 2)
    else
        # First site
        prod = contract(R2, renv, 2, 4)
        prod = contract(conj(R2), prod, 2, 5)
        prod = trace(prod, 2, 4)
        prod = contract(R1, prod, 1, 4)
        prod = trace(prod, 1, 4)
        prod = moveidx(prod, 3, 1)
        prod = moveidx(prod, 3, 2)
    end

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
