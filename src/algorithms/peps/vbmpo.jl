function vbMPO(O::MPO, env::Environment, direction::Bool, level::Int; kwargs...)
    # Truncation criteria
    cutoff = get(kwargs, :cutoff, 1e-12)
    maxchi = get(kwargs, :chi, 0)

    # Convergence criteria
    maxiter = get(kwargs, :maxiter, 100)
    miniter = get(kwargs, :miniter, 4)
    tol = get(kwargs, :tol, 1e-10)

    # Make a copy of the bMPO and truncate
    movecenter!(O, 1; maxdim=maxchi)

    # Construct the projectors
    proj = ProjbMPO(O, env, direction, level)
    movecenter!(proj, 1)

    # Calculate the cost
    projdiff = calculate(proj)
    normal = norm(O)^2
    cost =  normal - projdiff - conj(projdiff)

    # Loop until convergence
    converged = false
    rev = false
    iterations = 0
    #println("-------")
    while !converged
        # Loop through each site and optimize
        for i = 1:length(O)
            # Determine the site
            site = !rev ? i : length(O) + 1 - i

            # Orthogonalize
            movecenter!(O, site)
            movecenter!(proj, site)

            # Find optimal update
            O[site] = project(proj)
        end

        # Reverse the direction
        rev = !rev

        # Calculate cost
        oldcost = cost
        projdiff = calculate(proj)
        normal = norm(O)^2
        cost = normal - projdiff - conj(projdiff)

        # Check convergence
        iterations += 1
        converged = real((oldcost-cost) / normal) <= tol ? true : converged
        converged = (maxiter != 0 && iterations >= maxiter) ? true : converged
        converged = iterations < miniter ? false : converged
        #println((oldcost-cost)/normal)
    end

    return O
end
