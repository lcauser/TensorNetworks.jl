function dmrg(psi::MPS, Hs::ProjMPSSum; kwargs...)
    # DMRG options
    nsites::Int = get(kwargs, :nsites, 2)
    krylovdim::Int = get(kwargs, :krylovdim, 3)
    kryloviter::Int = get(kwargs, :kryloviter, 2)
    ishermitian::Bool = get(kwargs, :ishermitian, true)

    # Convergence criteria
    minsweeps::Int = get(kwargs, :minsweeps, 1)
    maxsweeps::Int = get(kwargs, :maxsweeps, 1000)
    tol::Float64 = get(kwargs, :tol, 1e-10)
    tolgrad::Float64 = get(kwargs, :tolgrad, 1e-5)
    numconverges::Float64 = get(kwargs, :numconverges, 4)
    verbose::Bool = get(kwargs, :verbose, 1)

    # Truncation
    cutoff::Float64 = get(kwargs, :cutoff, 1e-12)
    maxdim::Int = get(kwargs, :maxdim, 1000)
    mindim::Int = get(kwargs, :mindim, 1)

    # Calculate the cost & bond dimension
    cost = calculate(Hs)
    lastcost = copy(cost)
    D = maxbonddim(psi)
    lastD = copy(D)
    grad::Float64 = 0.0

    #  Loop through until convergence
    direction = false
    converged = false
    convergedsweeps = 0
    convergedgrad = 0
    sweeps = 0
    while !converged
        for j = 1:length(psi)+1-nsites
            # Determine the site
            site = direction ? length(psi) + 1 - j : j
            site1 = direction ? site + 1 - nsites : site

            # Build the projector blocks
            movecenter!(Hs, site)

            # Get the contracted tensors
            A0 = psi[site1]
            for i = 1:nsites-1
                A0 = contract(A0, psi[site1+i], 2+i, 1)
            end

            # Construct the effective hamilonian and solve
            Heff(x) = project(Hs, x, direction, nsites)
            eig, vec = eigsolve(Heff, A0, 1, :SR, maxiter=kryloviter,
                                krylovdim=krylovdim, ishermitian=ishermitian,
                                tol=1e-14)
            #eig, vec = eigs(Heff; nev=1, ncv=3, tol=0.1, v0=flatten(A0), which=:SR)
            cost = eig[1]
            # Replace
            replacesites!(psi, vec[1], site1, direction; cutoff=cutoff,
                          maxdim=maxdim, mindim=mindim)

        end
        # Reverse direction, build projector blocks to the end
        movecenter!(Hs, direction ? 1 : length(psi))
        direction = !direction

        # Check convergence
        sweeps += 1
        D = maxbonddim(psi)
        diff(x, y) = abs(x) < 1e-10 ? abs(x-y) : abs((x - y) / x)
        if sweeps >= minsweeps
            if diff(cost, lastcost) < tol && lastD == D
                convergedsweeps += 1
            else
                convergedsweeps = 0
            end
            if abs((diff(cost, lastcost) - grad) / (diff(cost, lastcost) + grad)) < tolgrad && lastD == D
                convergedgrad += 1
            else
                convergedgrad = 0
            end
            if max(convergedsweeps, convergedgrad) >= numconverges
                converged = true
            end
            if sweeps >= maxsweeps && maxsweeps != 0
                converged = true
            end
        end
        grad = abs(diff(cost, lastcost))
        lastcost = copy(cost)
        lastD = copy(D)

        # Output information
        if verbose
            @printf("Sweep=%d, energy=%.12f, maxbonddim=%d \n", sweeps,
                   real(cost), D)
        end
    end

    return psi, cost
end

"""
    dmrg(psi0::MPS, H::MPO; kwargs...)
    dmrg(psi0::MPS, Hs::Vector{MPO}; kwargs...)
    dmrg(psi0::MPS, H::MPO, V::MPS; kwargs...)
    dmrg(psi0::MPS, Hs::Vector{MPO}, Vs::Vector{MPS}; kwargs...)

Perform DMRG calculations with an MPO, or list of MPOs, and project out
vectors.

Key arguments:
    - nsites::Int : Number of sites to optimize over. Default is 2.
    - krylovdim::Int : Number of krylov vectors to create. Default is 3.
    - kryloviter::Int : Number of krylov iterations. Default is 1.
    - minsweeps::Int : Minimum number of DMRG sweeps to perform. Default is 1.
    - maxsweeps::Int : Maximum number of DMRG sweeps to perform. Use 0 for
        unlimited. Default is 1000.
    - tol::Float64 : Change in energy (per unit energy) tolerance before
        converged. Default is 1e-10.
    - numconverges::Int : Number of sweeps for convergence to be satisfied
        before finshing.
    - verbose::Bool : true for information output, false for no output. Default
        is true.
    - cutoff::Float64 : Truncation error for SVD. Default is 1e-12.
    - maxdim::Int : Maximum bond dimension. Default is 1000.
    - mindim::Int : Minimum bond dimension. Default is 1.
"""

function dmrg(psi0::MPS, H::MPO; kwargs...)
    movecenter!(psi0, 1)
    return dmrg(psi0, ProjMPSSum([ProjMPO(psi0, H)]); kwargs...)
end

function dmrg(psi0::MPS, Hs::Vector{MPO}; kwargs...)
    movecenter!(psi0, 1)
    return dmrg(psi0, ProjMPSSum([ProjMPO(psi0, H) for H = Hs]); kwargs...)
end

function dmrg(psi0::MPS, H::MPO, V::MPS; kwargs...)
    movecenter!(psi0, 1)
    Hs = [ProjMPS(psi0, V, 1.0; squared=true), ProjMPO(psi0, H)]
    return dmrg(psi0, ProjMPSSum(Hs); kwargs...)
end


function dmrg(psi0::MPS, Hs::Vector{MPO}, Vs::Vector{MPS}; kwargs...)
    movecenter!(psi0, 1)
    Hs = []
    for H in Hs
        push!(Hs, ProjMPO(psi0, H))
    end
    for V in Vs
        push!(Hs, ProjMPS(psi0, V, 1.0; squared=true))
    end
    return dmrg(psi0, ProjMPSSum(Hs); kwargs...)
end
