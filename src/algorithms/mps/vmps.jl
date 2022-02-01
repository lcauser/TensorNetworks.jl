function vmps(psi::GMPS, Vs::AbstractProjMPS; kwargs...)
    # Convergence criteria
    minsweeps::Int = get(kwargs, :minsweeps, 1)
    maxsweeps::Int = get(kwargs, :maxsweeps, 100)
    tol::Float64 = get(kwargs, :tol, 1e-8)
    numconverges::Float64 = get(kwargs, :numconverges, 1)
    verbose::Bool = get(kwargs, :verbose, 0)

    # Truncation
    nsites::Int = get(kwargs, :nsites, 2)
    cutoff::Float64 = get(kwargs, :cutoff, 1e-12)
    maxdim::Int = get(kwargs, :maxdim, 1000)
    mindim::Int = get(kwargs, :mindim, 1)

    # Calculate the cost & bond dimension
    cost = calculate(Vs)
    cost += conj(cost)
    lastcost = copy(cost)
    D = maxbonddim(psi)
    lastD = copy(D)

    #  Loop through until convergence
    direction = false
    converged = false
    convergedsweeps = 0
    sweeps = 0
    while !converged
        for j = 1:length(psi)+1-nsites
            # Determine the site
            site = direction ? length(psi) + 1 - j : j
            site1 = direction ? site + 1 - nsites : site

            # Build the projector blocks
            movecenter!(Vs, site)

            # Get the contracted tensors
            A0 = psi[site1]
            for i = 1:nsites-1
                A0 = contract(A0, psi[site1+i], 2+i, 1)
            end

            # Calculate the vector
            vec = project(Vs, A0, direction, nsites)

            # Project out
            #f(x) = project(Ms, x, direction, nsites)
            #println(f(vec))
            #vec, info = linsolve(f, vec, A0, maxiter=kryloviter,
            #                     krylovdim=krylovdim)

            # Replace
            replacesites!(psi, vec, site1, direction; cutoff=cutoff,
                          maxdim=maxdim, mindim=mindim)
        end
        # Reverse direction, build projector blocks to the end
        movecenter!(Vs, direction ? 1 : length(psi))
        direction = !direction

        # Check convergence
        sweeps += 1
        D = maxbonddim(psi)
        cost = calculate(Vs)
        cost += conj(cost)
        if sweeps >= minsweeps
            diff(x, y) = abs(x) < 1e-10 ? abs(x-y) : abs((x - y) / x)
            if diff(cost, lastcost) < tol && lastD == D
                convergedsweeps += 1
            else
                convergedsweeps == 0
            end
            if convergedsweeps >= numconverges
                converged = true
            end
            if sweeps >= maxsweeps && maxsweeps != 0
                converged = true
            end
        end
        lastcost = copy(cost)
        lastD = copy(D)

        # Output information
        if verbose
            @printf("Sweep=%d, energy=%.12f, maxbonddim=%d \n", sweeps,
                   real(cost), D)
        end
    end

    return psi
end


function vmps(psis::GMPS...; kwargs...)
    # Orthogonalize psi and make a copy
    psi0 = deepcopy(psis[1])
    movecenter!(psi0, 1)

    # Create the projections
    projPsis = ProjMPSSum([ProjMPS(psi, psi0) for psi in psis]; kwargs...)

    return vmps(psi0, projPsis; kwargs...)
end
