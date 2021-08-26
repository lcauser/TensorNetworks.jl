function vmps(psi::MPS, Ms::ProjMPSSum, Vs::ProjMPSSum; kwargs...)
    # DMRG options
    nsites::Int = get(kwargs, :nsites, 2)
    krylovdim::Int = get(kwargs, :krylovdim, 1)
    kryloviter::Int = get(kwargs, :kryloviter, 1)

    # Convergence criteria
    minsweeps::Int = get(kwargs, :minsweeps, 1)
    maxsweeps::Int = get(kwargs, :maxsweeps, 100)
    tol::Float64 = get(kwargs, :tol, 1e-10)
    numconverges::Float64 = get(kwargs, :numconverges, 4)
    verbose::Bool = get(kwargs, :verbose, 1)

    # Truncation
    cutoff::Float64 = get(kwargs, :cutoff, 1e-12)
    maxdim::Int = get(kwargs, :maxdim, 1000)
    mindim::Int = get(kwargs, :mindim, 1)

    # Calculate the cost & bond dimension
    cost = calculate(Ms)
    costVs = calculate(Vs)
    cost += costVs + conj(costVs)
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
            movecenter!(Ms, site)
            movecenter!(Vs, site)

            # Get the contracted tensors
            A0 = psi[site1]
            for i = 1:nsites-1
                A0 = contract(A0, psi[site1+i], 2+i, 1)
            end

            # Construct the effective hamilonian and solve
            f(x) = project(Ms, x, direction, nsites)
            b = project(Vs, A0, direction, nsites)
            vec, info = linsolve(f, b, A0, maxiter=kryloviter,
                                 krylovdim=krylovdim)

            # Replace
            replacesites!(psi, vec, site1, direction; cutoff=cutoff,
                          maxdim=maxdim, mindim=mindim)
        end
        # Reverse direction, build projector blocks to the end
        movecenter!(Ms, direction ? 1 : length(psi))
        movecenter!(Vs, direction ? 1 : length(psi))
        direction = !direction

        # Check convergence
        sweeps += 1
        D = maxbonddim(psi)
        cost = calculate(Ms)
        costVs = calculate(Vs)
        cost += costVs + conj(costVs)
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


function vmps(psi::MPS; kwargs...)
    # Orthogonalize psi and make a copy
    psi0 = deepcopy(psi)
    movecenter!(psi0, 1)

    # Create the projections
    projPsi = ProjMPS(psi0, psi; kwargs...)
    projNorm = ProjMPS(psi0, psi0; kwargs...)
    Ms = ProjMPSSum([projNorm]; squared=true, kwargs...)
    Vs = ProjMPSSum([projPsi]; kwargs...)
    return vmps(psi0, Ms, Vs; kwargs...)
end


function vmps(psis::Vector{MPS}, Vs::Vector{MPS}; kwargs...)
    # Orthogonalize psi and make a copy
    psi0 = deepcopy(psi)
    movecenter!(psi0, 1)

    # Create the projections
    projPsis = ProjMPSSum([ProjMPS(psi0, psi) for psi in psis]; kwargs...)
    projMs = [ProjMPS(psi0, psi, squared=true; kwargs...) for psi in Vs]
    push!(projMs, ProjMPS(psi0, psi0; kwargs...))
    projMs = ProjMPSSum(projMs; kwargs...)

    return vmps(psi0, projMs, projPsis; kwargs...)
end
