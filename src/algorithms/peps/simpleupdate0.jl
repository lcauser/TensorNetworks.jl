## TO-DO: Energy calculation via gate, not through list

function simpleupdate(psi::PEPS, dt::Real, st::Sitetypes, ops, coeffs; kwargs...)
    # Get convergence arguments
    maxiter = get(kwargs, :maxiter, 0)
    miniter = get(kwargs, :miniter, 1)
    tol = get(kwargs, :tol, 1e-7)
    saveiter = get(kwargs, :saveiter, 100)

    # Get psi properties
    maxdim = get(kwargs, :maxdim, maxbonddim(psi))
    N = length(psi)

    # Create the gate
    gate = exp(dt*creategate(st, ops, coeffs), [2, 4])

    # Measure energies
    ops = [[op(st, name) for name in op1] for op1 in ops]
    function calculateenergy(psi)
        env = Environment(psi, psi; kwargs...)
        return real(inner(env, ops, coeffs) / inner(env))
    end

    # Create the list of gate applications
    gatelist = []
    for i = 1:N
        j = 1
        while j < N
            push!(gatelist, Any[i, j, false])
            j += 2
        end

        j = 2
        while j < N
            push!(gatelist, Any[i, j, false])
            j += 2
        end
    end
    for j = 1:N
        i = 1
        while i < N
            push!(gatelist, Any[i, j, true])
            i += 2
        end

        i = 2
        while i < N
            push!(gatelist, Any[i, j, true])
            i += 2
        end
    end

    # Determine the (square root) singular values of bonds
    singulars1 = Any[[] for i = 1:N]
    singulars2 = Any[[] for i = 1:N-1]
    for i = 1:N
        for j = 1:N
            if j < N
                # Bring tensors together and apply SVD
                A1 = psi[i, j]
                A2 = psi[i, j+1]
                dims1 = size(A1)
                dims2 = size(A2)
                A = contract(A1, A2, 4, 1)
                A, cmb1 = combineidxs(A, [1, 2, 3, 4])
                A, cmb2 = combineidxs(A, [1, 2, 3, 4])
                U, S, V = svd(A, 2; maxdim=dims1[4])
                S = sqrt.(S)

                # Store singular values, restore tensors
                push!(singulars1[i], diag(S))
                U = contract(U, S, 2, 1)
                V = contract(S, V, 2, 1)
                U = reshape(U, (dims1[1], dims1[2], dims1[3], dims1[5], dims1[4]))
                U = moveidx(U, 5, 4)
                V = reshape(V, (dims2[1], dims2[2], dims2[3], dims2[4], dims2[5]))
                psi[i, j] = U
                psi[i, j+1] = V
            end

            if i < N
                # Bring tensors together and apply SVD
                A1 = psi[i, j]
                A2 = psi[i+1, j]
                dims1 = size(A1)
                dims2 = size(A2)
                A = contract(A1, A2, 3, 2)
                A, cmb1 = combineidxs(A, [1, 2, 3, 4])
                A, cmb2 = combineidxs(A, [1, 2, 3, 4])
                U, S, V = svd(A, 2; maxdim=dims1[3])
                S = sqrt.(S)

                # Store singular values, restore tensors
                push!(singulars2[i], diag(S))
                U = contract(U, S, 2, 1)
                V = contract(S, V, 2, 1)
                U = reshape(U, (dims1[1], dims1[2], dims1[4], dims1[5], dims1[3]))
                U = moveidx(U, 5, 3)
                V = reshape(V, (dims2[2], dims2[1], dims2[3], dims2[4], dims2[5]))
                V = moveidx(V, 2, 1)
                psi[i, j] = U
                psi[i+1, j] = V
            end
        end
    end

    converge = false
    iter = 0
    lastenergy = calculateenergy(psi)
    energy = copy(lastenergy)
    while !converge
        for i in gatelist
            # Bring in the singular values to the relevent sites
            if !i[3]
                A1 = psi[i[1], i[2]]
                A2 = psi[i[1], i[2]+1]
                if i[2] > 1
                    A1 = contract(diagm(singulars1[i[1]][i[2]-1]), A1, 2, 1)
                end
                if i[1] > 1
                    A1 = contract(diagm(singulars2[i[1]-1][i[2]]), A1, 2, 2)
                    A1 = moveidx(A1, 1, 2)
                    A2 = contract(diagm(singulars2[i[1]-1][i[2]+1]), A2, 2, 2)
                    A2 = moveidx(A2, 1, 2)
                end
                if i[1] < N
                    A1 = contract(A1, diagm(singulars2[i[1]][i[2]]), 3, 1)
                    A1 = moveidx(A1, 5, 3)
                    A2 = contract(A2, diagm(singulars2[i[1]][i[2]+1]), 3, 1)
                    A2 = moveidx(A2, 5, 3)
                end
                if i[2]+1 < N
                    A2 = contract(A2, diagm(singulars1[i[1]][i[2]+1]), 4, 1)
                    A2 = moveidx(A2, 5, 4)
                end
            else
                A1 = psi[i[1], i[2]]
                A2 = psi[i[1]+1, i[2]]
                if i[1] > 1
                    A1 = contract(diagm(singulars2[i[1]-1][i[2]]), A1, 2, 2)
                    A1 = moveidx(A1, 1, 2)
                end
                if i[2] > 1
                    A1 = contract(diagm(singulars1[i[1]][i[2]-1]), A1, 2, 1)
                    A2 = contract(diagm(singulars1[i[1]+1][i[2]-1]), A2, 2, 1)
                end
                if i[2] < N
                    A1 = contract(A1, diagm(singulars1[i[1]][i[2]]), 4, 1)
                    A1 = moveidx(A1, 5, 4)
                    A2 = contract(A2, diagm(singulars1[i[1]+1][i[2]]), 4, 1)
                    A2 = moveidx(A2, 5, 4)
                end
                if i[1]+1 < N
                    A2 = contract(A2, diagm(singulars2[i[1]+1][i[2]]), 3, 1)
                    A2 = moveidx(A2, 5, 3)
                end
            end

            # Contract sites with gate and apply SVD
            if !i[3]
                # Contract tensors
                prod = contract(A1, A2, 4, 1)
                prod = contract(prod, gate, 4, 2)
                prod = trace(prod, 7, 10)
                dims1 = size(prod)[[1, 2, 3, 7]]
                dims2 = size(prod)[[4, 5, 6, 8]]
                prod, cmb1 = combineidxs(prod, [1, 2, 3, 7])
                prod, cmb2 = combineidxs(prod, [1, 2, 3, 4])

                # Split with SVD
                U, S, V = svd(prod, -1; maxdim=maxdim, cutoff=1e-16)
                S = sqrt.(S)
                #println(diag(S))
                S = S / maximum(S)
                U = contract(U, S, 2, 1)
                V = contract(S, V, 2, 1)

                # Reshape and restore
                A1 = reshape(U, [dims1..., size(S)[2]]...)
                A1 = moveidx(A1, 5, 4)
                A2 = reshape(V, [size(S)[1], dims2...]...)
                singulars1[i[1]][i[2]] = diag(S)
            else
                # Contract tensors
                prod = contract(A1, A2, 3, 2)
                prod = contract(prod, gate, 4, 2)
                prod = trace(prod, 7, 10)
                dims1 = size(prod)[[1, 2, 3, 7]]
                dims2 = size(prod)[[4, 5, 6, 8]]
                prod, cmb1 = combineidxs(prod, [1, 2, 3, 7])
                prod, cmb2 = combineidxs(prod, [1, 2, 3, 4])

                # Split with SVD
                U, S, V = svd(prod, -1; maxdim=maxdim, cutoff=1e-16)
                S = sqrt.(S)
                #println(diag(S))
                S = S / maximum(S)
                U = contract(U, S, 2, 1)
                V = contract(S, V, 2, 1)

                # Reshape and restore
                A1 = reshape(U, [dims1..., size(S)[2]]...)
                A1 = moveidx(A1, 5, 3)
                A2 = reshape(V, [size(S)[1], dims2...]...)
                A2 = moveidx(A2, 1, 2)
                singulars2[i[1]][i[2]] = diag(S)
            end

            # Multiple by inverse of the singular values
            if !i[3]
                if i[2] > 1
                    A1 = contract(diagm(1 ./ singulars1[i[1]][i[2]-1]), A1, 2, 1)
                end
                if i[1] > 1
                    A1 = contract(diagm(1 ./ singulars2[i[1]-1][i[2]]), A1, 2, 2)
                    A1 = moveidx(A1, 1, 2)
                    A2 = contract(diagm(1 ./ singulars2[i[1]-1][i[2]+1]), A2, 2, 2)
                    A2 = moveidx(A2, 1, 2)
                end
                if i[1] < N
                    A1 = contract(A1, diagm(1 ./ singulars2[i[1]][i[2]]), 3, 1)
                    A1 = moveidx(A1, 5, 3)
                    A2 = contract(A2, diagm(1 ./ singulars2[i[1]][i[2]+1]), 3, 1)
                    A2 = moveidx(A2, 5, 3)
                end
                if i[2]+1 < N
                    A2 = contract(A2, diagm(1 ./ singulars1[i[1]][i[2]+1]), 4, 1)
                    A2 = moveidx(A2, 5, 4)
                end
            else
                if i[1] > 1
                    A1 = contract(diagm(1 ./ singulars2[i[1]-1][i[2]]), A1, 2, 2)
                    A1 = moveidx(A1, 1, 2)
                end
                if i[2] > 1
                    A1 = contract(diagm(1 ./ singulars1[i[1]][i[2]-1]), A1, 2, 1)
                    A2 = contract(diagm(1 ./ singulars1[i[1]+1][i[2]-1]), A2, 2, 1)
                end
                if i[2] < N
                    A1 = contract(A1, diagm(1 ./ singulars1[i[1]][i[2]]), 4, 1)
                    A1 = moveidx(A1, 5, 4)
                    A2 = contract(A2, diagm(1 ./ singulars1[i[1]+1][i[2]]), 4, 1)
                    A2 = moveidx(A2, 5, 4)
                end
                if i[1]+1 < N
                    A2 = contract(A2, diagm(1 ./ singulars2[i[1]+1][i[2]]), 3, 1)
                    A2 = moveidx(A2, 5, 3)
                end
            end

            # Restore tensors
            if !i[3]
                psi[i[1], i[2]] = A1
                psi[i[1], i[2]+1] = A2
            else
                psi[i[1], i[2]] = A1
                psi[i[1]+1, i[2]] = A2
            end
        end

        # Check convergence
        iter += 1
        converge = iter >= maxiter ? true : converge
        if iter % saveiter == 0
            energy = calculateenergy(psi)
            converge = ((energy-lastenergy) / abs(energy) < tol) ? true : converge
            @printf("iter=%d, energy=%.12f \n", iter, energy)
            lastenergy = energy
        end
        converge = iter < miniter ? false : converge
    end

    return psi, energy

end
