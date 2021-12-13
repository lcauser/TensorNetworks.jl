function vbMPO(O::MPS, env::SingleEnvironment, level::Int; kwargs...)
    # Truncation criteria
    cutoff = get(kwargs, :cutoff, 0)
    maxchi = get(kwargs, :chi, 0)

    # Convergence criteria
    maxiter = get(kwargs, :maxiter, 1000)
    miniter = get(kwargs, :miniter, 2)
    tol = get(kwargs, :tol, 1e-14)

    # Make a copy of the bMPS and truncate
    movecenter!(O, 1)

    # Construct the projectors
    proj = ProjbMPS(O, env, direction, level)
    movecenter!(proj, 1)

    # Calculate the cost
    projdiff = calculate(proj)
    normal = norm(O)^2
    cost = real(normal - projdiff - conj(projdiff))

    # Loop until convergence
    converged = false
    rev = false
    iterations = 0
    chi = maxbonddim(O)
    #println("------")
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
        cost = real(normal - projdiff - conj(projdiff))
        diff = real((oldcost-cost) / real(normal))
        #println(diff)

        # Check convergence
        iterations += 1
        converged = diff < tol ? true : converged
        converged = (maxiter != 0 && iterations >= maxiter) ? true : converged
        converged = iterations < miniter ? false : converged
    end
    #println(iterations)
    return O
end
