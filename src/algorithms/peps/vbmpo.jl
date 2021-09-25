function vbMPO(O::MPO; kwargs...)
    # Truncation criteria
    cutoff = get(kwargs, :cutoff, 1e-12)
    maxchi = get(kwargs, :chi, 0)

    # Convergence criteria
    maxiter = get(kwargs, :maxiter, 100)
    miniter = get(kwargs, :miniter, 4)
    tol = get(kwargs, :tol, 1e-14)

    # Make a copy of the bMPO and truncate
    P = deepcopy(O)
    P.center = length(P)
    movecenter!(P, 1; maxdim=maxchi)

    # Construct the projectors
    projOP = ProjbMPO(O, P)

    # Calculate the cost
    projOPdiff = calculate(projOP)
    normal = norm(P)^2
    cost =  normal - projOPdiff - conj(projOPdiff)

    # Loop until convergence
    converged = false
    rev = false
    iterations = 0
    #println("-----")
    while !converged
        # Loop through each site and optimize
        for i in length(O)
            # Determine the site
            site = !rev ? i : length(O) + 1 - i

            # Orthogonalize
            movecenter!(P, site)
            movecenter!(projOP, site)

            # Find optimal update
            P[site] = project(projOP)
        end

        # Reverse the direction
        rev = !rev

        # Calculate cost
        oldcost = cost
        projOPdiff = calculate(projOP)
        normal = norm(P)^2
        cost = normal - projOPdiff - conj(projOPdiff)

        # Check convergence
        iterations += 1
        converged = real((oldcost-cost) / normal) <= tol ? true : converged
        converged = (maxiter != 0 && iterations >= maxiter) ? true : converged
        converged = iterations < miniter ? false : converged
        #println((oldcost-cost)/normal)
    end
    #println(normal)

    return P
end
