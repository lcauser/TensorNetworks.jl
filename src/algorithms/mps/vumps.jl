"""
    vumps(st::Sitetypes, psi::uMPS, H::InfiniteOpList; kwargs...)

The variational uniform matrix product states method minimizes the expectation
of a uMPS with an infinite translationally invariant Hamiltonian, H.
"""
function vumps(st::Sitetypes, psi::uMPS, H::InfiniteOpList; kwargs...)
    # Convergence criteria
    maxiter::Int = get(kwargs, :maxiter, 0)
    miniter::Int = get(kwargs, :miniter, 1)
    tol::Float64 = get(kwargs, :tol, 1e-8)
    verbose::Bool = get(kwargs, :verbose, true)

    # Optimisation parameters 
    krylov_dim::Int = get(kwargs, :krylov_dim, 10)
    krylov_iters::Int = get(kwargs, :krylov_iters, 1000)
    krylov_tol_factor::Float64 = get(kwargs, :krylov_tol_factor, 1e-2)

    # Find MPO of Hamiltonian & energy 
    H = MPO(st, H)
    E = inner(psi, H)

    # Iterate until convergences 
    iter = 0
    converge = false
    δ = 1
    while !converge 
        # Determine relevent tensors 
        C = psi.C
        Ac = contract(psi.Al, C, 3, 1)
        LH = leftEnvironment(psi, H, E)
        RH = rightEnvironment(psi, H, E)

        # Determine the gradient 
        δ = norm(_Ac_eff(Ac, psi.Al, psi.Ar, LH, RH, H, E) -
        contract(psi.Al, _C_eff(C, psi.Al, psi.Ar, LH, RH, H, E), 3, 1))
        
        # Find eigensolutions 
        eigs, vecs = eigsolve(x -> _Ac_eff(x, psi.Al, psi.Ar, LH, RH, H, E),
                              Ac, 1, :SR;tol=krylov_tol_factor*δ,
                              krylovdim=krylov_dim, maxiter=krylov_iters)
        Ac = vecs[argmin(real(eigs))]

        eigs, vecs = eigsolve(x -> _C_eff(x, psi.Al, psi.Ar, LH, RH, H, E),
                              C, 1, :SR; tol=krylov_tol_factor*δ,
                              krylovdim=krylov_dim, maxiter=krylov_iters)
        C = vecs[argmin(real(eigs))]

        # Do Polar decompositions to find the optimal Al and Ar
        Ucl, _ = polar(C; alg=:hybrid)
        Ac2, cmb = combineidxs(Ac, [1, 2])
        Ac2 = moveidx(Ac2, 2, 1)
        Ual, _ = polar(Ac2; alg=:hybrid)
        Ual = uncombineidxs(moveidx(Ual, 1, 2), cmb)
        Al = contract(Ual, conj(Ucl), 3, 2)

        Ucr, _ = polar(moveidx(C, 2, 1); alg=:hybrid)
        Ucr = moveidx(Ucr, 1, 2)
        Ac2, cmb = combineidxs(Ac, [2, 3])
        Ac2 = moveidx(Ac2, 2, 1)
        Uar, _ = polar(Ac2; alg=:hybrid)
        Uar = uncombineidxs(moveidx(Uar, 1, 2), cmb)
        Ar = contract(conj(Ucr), Uar, 1, 1)

        # SVD to make C diagonal of singular values 
        U, C, V = svd(C, 2)
        Al = contract(conj(U), contract(Al, U, 3, 1), 1, 1)
        Ar = contract(V, contract(Ar, conj(V), 3, 2), 2, 1)

        # Update tensors 
        psi.C = C
        psi.Al = Al
        psi.Ar = Ar

        # Find energy & ouput 
        iter += 1
        E = inner(psi, H)
        if verbose
            @printf("Iter=%d, energy=%.8e, bond_dim=%d, gradient=%.8e \n", iter, real(E),
                    maxbonddim(psi), δ)
        end

        # Check for convergence 
        if iter >= miniter
            converge = (iter >= maxiter && maxiter != 0) ? true : converge
            converge = δ < tol ? true : converge
        end
    end

    return psi, E
end



function _Ac_eff(Ac, Al, Ar, LH, RH, H, E)
    ### Function to calculate the action of the effective Hamiltonian on the
    ### Ac term
    # Calculate the first terms 
    left = diagm(ones(ComplexF64, size(Al)[1]))
    left = tensorproduct(left, ones(ComplexF64, 1))
    left = moveidx(left, 3, 2)
    right = deepcopy(left)
    lefts = [left]
    rights = [right]
    for i = 1:length(H)-1
        # Grow the left 
        left = contract(left, conj(Al), 1, 1)
        left = contract(left, H[i], [1, 3], [1, 2])
        left = contract(left, Al, [1, 3], [1, 2])
        push!(lefts, left)

        # Grow the right 
        right = contract(Ar, right, 3, 3)
        right = contract(H[length(H)+1-i], right, [3, 4], [2, 4])
        right = contract(conj(Ar), right, [2, 3], [2, 4])
        pushfirst!(rights, right)
    end

    # Do all the contractions 
    term1 = zeros(ComplexF64, size(Ac))
    for i = 1:length(H)
        prod = contract(lefts[i], conj(Ac), 1, 1)
        prod = contract(prod, H[i], [1, 3], [1, 2])
        prod = contract(prod, rights[i], [2, 4], [1, 2])
        term1 .+= prod .- (E * conj(Ac))
    end

    # Calculate the second terms 
    term2 = contract(LH, conj(Ac), 1, 1)
    term3 = contract(conj(Ac), RH, 3, 1)

    return conj(term1 .+ term2 .+ term3)
end


function _C_eff(C, Al, Ar, LH, RH, H, E)
    ### Calculates the action of the effective Hamiltonian on the C term
    # Calculate the first terms 
    lefts = []
    rights = []
    left = diagm(ones(ComplexF64, size(Al)[1]))
    left = tensorproduct(left, ones(ComplexF64, 1))
    left = moveidx(left, 3, 2)
    right = deepcopy(left)
    for i = 1:length(H)-1
        # Grow the left 
        left = contract(left, conj(Al), 1, 1)
        left = contract(left, H[i], [1, 3], [1, 2])
        left = contract(left, Al, [1, 3], [1, 2])
        push!(lefts, left)

        # Grow the right 
        right = contract(Ar, right, 3, 3)
        right = contract(H[length(H)+1-i], right, [3, 4], [2, 4])
        right = contract(conj(Ar), right, [2, 3], [2, 4])
        pushfirst!(rights, right)
    end

    # Do all the contractions 
    term1 = zeros(ComplexF64, size(C))
    for i = 1:length(H)-1
        prod = contract(lefts[i], conj(C), 1, 1)
        prod = contract(prod, rights[i], [3, 1], [1, 2])
        term1 .+= prod .- (E * conj(C))
    end

    # Second terms 
    term2 = contract(LH, conj(C), 1, 1)
    term3 = contract(conj(C), RH, 2, 1)

    return conj(term1 .+ term2 .+ term3)
end