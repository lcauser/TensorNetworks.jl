function vmps(psi0::MPS, Ms::ProjMPSSum, Vs::ProjMPSSum; kwargs...)
    println("------------------")
    println("------------------")
    println(calculate(Vs))
    direction = false
    movecenter!(Ms, 1)
    movecenter!(Vs, 1)
    for sweep = 1:2
        # Orthogonalize
        movecenter!(psi0, direction ? length(psi0) : 1)
        movecenter!(Ms, direction ? length(psi0) : 1)
        movecenter!(Vs, direction ? length(psi0) : 1)

        # Loop through each site
        for i = 1:length(psi0)-1
            println("------------------")
            # Determine the site
            site = direction ? length(psi0) - i + 1 : i
            site1 = direction ? site - 1 : site
            site2 = direction ? site : site + 1
            println(site)
            println(site1)
            println(site2)

            # Determine A0
            A = contract(psi0[site1], psi0[site2], 3, 1)

            #  Optimize
            f(x) = project(Ms, x, direction)
            b = conj(project(Vs, A, direction))
            println(sum(abs.(A-b)))
            A, info = linsolve(f, conj(b), A0; krylovdim=3, maxiter=1)
            replacesites!(psi0, b, site1, direction; kwargs...)
            println(sum(abs.(A-b)))

            # Move centers
            movecenter!(Ms, direction ? site1 : site2)
            movecenter!(Vs, direction ? site1 : site2)
            println(psi0.center)
            println(calculate(Vs))
        end

        direction = !direction
    end

    return psi0
end


function vmps(psi::MPS; kwargs...)
    # Orthogonalize psi and make a copy
    movecenter!(psi, 1)
    psi0 = deepcopy(psi)

    # Create the projections
    projPsi = ProjMPS(psi0, psi; kwargs...)
    projNorm = ProjMPS(psi0, psi0; kwargs...)
    Ms = ProjMPSSum([projNorm]; squared=true, kwargs...)
    Vs = ProjMPSSum([projPsi]; kwargs...)
    return vmps(psi0, Ms, Vs; kwargs...)
end
