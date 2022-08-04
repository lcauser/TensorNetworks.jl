function vpeps(st::Sitetypes, psi::GPEPS, H::GPEPS, ops::OpList2d; kwargs...)
    # Key arguments
    chi::Int = get(kwargs, :chi, maxbonddim(psi)^2 * maxbonddim(H))
    chieval::Int = get(kwargs, :chi_eval, 100)
    minsweeps::Int = get(kwargs, :minsweeps, 1)
    maxsweeps::Int = get(kwargs, :maxsweeps, 100)
    tol::Float64 = get(kwargs, :tol, 1e-6)

    # Construct environments
    Henv = Environment(psi, H, psi; chi=chi)
    Nenv = Environment(psi, psi; chi=chi)

    # Evaluation of energy
    function calculateenergy()
        env = Environment(psi, psi; chi=chieval)
        return real(sum(inner(st, env, ops)))
    end

    # Determine size and sweeps
    N = size(psi)[1]
    idxs = [1:N..., N-1:-1:1...]

    # Loop through optimizing each site
    E = calculateenergy()
    Elast = copy(E)
    converge = false
    sweeps = 0
    @printf("iter=%d, energy=%.12f \n", sweeps, E)
    while !converge
        for i = idxs
            println(i)
            for j = idxs
                # Build environments
                build!(Henv, i, j)
                build!(Nenv, i, j)

                # Optimize site
                A = vpeps_krylov(Henv, Nenv, psi[i, j]; kwargs...)
                A /= sqrt(sum(conj(A) .* vpeps_contract_norm(Nenv, A)))
                E = sum(conj(A) .* vpeps_contract_env(Henv, A))
                println(E)
                psi[i, j] = A
            end
        end

        # Check for convergence
        sweeps += 1
        E = calculateenergy()
        if sweeps >= minsweeps
            converge = 2 * (Elast - E) / abs(E + Elast) < tol ? true : converge
        end
        converge = sweeps >= maxsweeps ? true : converge
        Elast = E

        # Output information
        @printf("iter=%d, energy=%.12f \n", sweeps, E)
    end
    return psi, E
end


function vpeps_contract_norm(env::Environment, A)
    left = MPSblock(env, env.centerMPS-1)
    prod = MPSblock(env, env.centerMPS+1)
    #M = env.objects[2][env.center, env.centerMPS]
    M1 = block(env, env.center-1)[env.centerMPS]
    M2 = block(env, env.center+1)[env.centerMPS]
    prod = contract(M2, prod, 4, 4)
    prod = contract(A, prod, [3, 4], [3, 6])
    prod = contract(M1, prod, [3, 4], [2, 6])
    prod = contract(left, prod, [1, 3, 4], [1, 3, 5])
    prod = moveidx(prod, 3, 5)
    return prod
end

function vpeps_contract_env(env::Environment, A)
    left = MPSblock(env, env.centerMPS-1)
    prod = MPSblock(env, env.centerMPS+1)
    M = env.objects[2][env.center, env.centerMPS]
    M1 = block(env, env.center-1)[env.centerMPS]
    M2 = block(env, env.center+1)[env.centerMPS]
    prod = contract(M2, prod, 5, 5)
    prod = contract(A, prod, [3, 4], [4, 8])
    prod = contract(M, prod, [3, 4, 6], [6, 9, 3])
    prod = contract(M1, prod, [3, 4, 5], [2, 5, 8])
    prod = contract(left, prod, [1, 3, 4, 5], [1, 3, 5, 6])
    prod = moveidx(prod, 3, 5)
    return prod
end

function vpeps_krylov(Henv::Environment, Nenv::Environment, A; kwargs...)
    # Fetch key arguments
    kryloviter::Int = get(kwargs, :kryloviter, 100)
    krylovdim::Int = get(kwargs, :krylovdim, 3)
    tol::Float64 = get(kwargs, :krylovtol, 1e-10)

    # Define effective matrices
    Heff(A) = vpeps_contract_env(Henv, A)
    Neff(A) = vpeps_contract_norm(Nenv, A)

    # Apply krylov iterations
    for iter = 1:kryloviter
        # Starting guess
        q = A / sqrt(sum(conj(A) .* A))
        vecs = [q]

        # Find krylov subspace
        for i = 2:krylovdim
            q = Heff(vecs[i-1])
            for j = 1:i-1
                h = sum(conj(vecs[j]) .* q)
                q = q .- h .* vecs[j]
            end
            h = sum(conj(q) .* q)
            real(h) <= tol && break
            q = q / sqrt(h)
            push!(vecs, q)
        end
        length(vecs) == 1 && break

        # Find effective H and N matricies
        Hk = zeros(ComplexF64, length(vecs), length(vecs))
        Nk = zeros(ComplexF64, length(vecs), length(vecs))
        for i = 1:length(vecs)
            for j = 1:length(vecs)
                Hk[i, j] = sum(conj(vecs[i]) .* Heff(vecs[j]))
                Nk[i, j] = sum(conj(vecs[i]) .* Neff(vecs[j]))
            end
        end

        # Invert effective norm to solve
        Hprime = inv(Nk) * Hk
        v = zeros(length(vecs))
        v[1] = 1
        eig, vec = eigsolve(Hprime, v, 1, :SR)
        A = zeros(ComplexF64, size(A))
        for i = 1:length(vecs)
            A .+= vec[1][i].*vecs[i]
        end
        A /= sqrt(sum(conj(A) .* A))
    end

    return A
end
