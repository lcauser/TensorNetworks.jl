function dmrgx(psi::GMPS, Hs::ProjMPSSum; kwargs...)
    # DMRG options
    nsites::Int = get(kwargs, :nsites, 2)
    krylovdim::Int = get(kwargs, :krylovdim, 100)
    kryloviter::Int = get(kwargs, :kryloviter, 100)
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

    # Create projection of inner product of psi and psi
    projnorm = ProjMPS(psi, psi; squared=false, rank=1)
    movecenter!(projnorm, 1)
    initial = deepcopy(psi)
    projinitial = ProjMPS(initial, psi; squared=false, rank=1)
    movecenter!(projinitial, 1)

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
            movecenter!(projnorm, site)
            movecenter!(projinitial, site)

            # Get the contracted tensors
            A0 = psi[site1]
            for i = 1:nsites-1
                A0 = contract(A0, psi[site1+i], 2+i, 1)
            end

            # Construct the effective hamilonian and solve
            Heff(x) = product(Hs, x, direction, nsites)
            eig, vec = eigsolve(Heff, A0, 1, :SR, krylovdim=prod(size(A0)), ishermitian=ishermitian,
                                tol=1e-14)
            #eig, vec = eigs(Heff; nev=1, ncv=3, tol=0.1, v0=flatten(A0), which=:SR)

            # Determine overlaps
            overlaps = []
            for i = 1:length(vec)
                push!(overlaps, abs(product(projinitial, vec[i], direction, nsites))^2)
            end
            println(maximum(overlaps))

            # Find maximum overlap and update
            idx = argmax(overlaps)
            cost = eig[idx]
            replacesites!(psi, vec[idx], site1, direction; cutoff=cutoff,
                          maxdim=maxdim, mindim=mindim)

        end
        # Reverse direction, build projector blocks to the end
        movecenter!(Hs, direction ? 1 : length(psi))
        movecenter!(projnorm, direction ? 1 : length(psi))
        movecenter!(projinitial, direction ? 1 : length(psi))
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
    dmrgx(psi0::GMPS, H::GMPS; kwargs...)
    dmrgx(psi0::GMPS, Hs::Vector{GMPS}; kwargs...)
    dmrgx(psi0::GMPS, H::GMPS, V::GMPS; kwargs...)
    dmrgx(psi0::GMPS, Hs::Vector{MPO}, Vs::Vector{MPS}; kwargs...)

Perform DMRG-X calculations with an MPO, or list of MPOs, and project out
vectors. It calculates all the eigenvectors and eigenvalues of some effective
Hamiltonian, and picks the one with most overlap with the current state as an
update.

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

function dmrgx(psi::GMPS, Hs::GMPS...; kwargs...)
    # Make sure psi is correct
    d = dim(psi)
    N = length(psi)
    rank(psi) != 1 && error("Psi must be a GMPS of rank 1 (vector).")

    # Construct effective Hamiltonian
    ProjHs = ProjMPS[]
    length(Hs) == 0 && error("You must provide atleast one MPS/MPO for the Hamiltonian.")
    for i = 1:length(Hs)
        length(Hs[i]) != N && error("GMPS must share the same properties.")
        dim(Hs[i]) != d && error("GMPS must share the same properties.")
        if rank(Hs[i]) == 2
            push!(ProjHs, ProjMPS(psi, Hs[i], psi; rank=2))
        elseif rank(Hs[i]) == 1
            push!(ProjHs, ProjMPS(Hs[i], psi; rank=2, squared=true))
        else
            error("Hamiltonian must be composed of MPOs (rank 2) or MPSs (rank 1).")
        end
    end

    H = ProjMPSSum(ProjHs)
    movecenter!(H, 1)
    return dmrgx(psi, H; kwargs...)
end
