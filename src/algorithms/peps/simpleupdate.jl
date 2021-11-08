## TO-DO: Energy calculation via gate, not through list

function simpleupdate(psi::PEPS, dt::Real, st::Sitetypes, ops::OpList2d; kwargs...)
    # Get convergence arguments
    maxiter = get(kwargs, :maxiter, 0)
    miniter = get(kwargs, :miniter, 1)
    tol = get(kwargs, :tol, 1e-6)
    saveiter = get(kwargs, :saveiter, 1000)
    savetime = dt*saveiter
    tol *= savetime

    # Get psi properties
    maxdim = get(kwargs, :maxdim, maxbonddim(psi))
    chi = maxdim != 1 ? get(kwargs, :chi, 300) : 1
    N = length(psi)

    # Create the gates
    gates = trotterize(st, ops, dt)

    # Measure energies
    function calculateenergy(psi)
        env = Environment(psi, psi; chi=chi)
        normal = inner(env)
        return real(sum(inner(st, env, ops) / normal))
    end

    # Determine the (square root) singular values of bonds
    singulars1 = Any[[] for i = 1:N]
    singulars2 = Any[[] for i = 1:N-1]
    for i = 1:N
        for j = 1:N
            if j < N
                # Bring tensors together and apply SVD
                A1, R1 = reducedtensor(psi, i, j, 4)
                A2, R2 = reducedtensor(psi, i, j+1, 1)
                dims1 = size(A1)
                dims2 = size(A2)
                D = size(R1)[2]
                R = contract(R1, R2, 2, 1)
                R, cmb1 = combineidxs(R, [1, 2])
                R, cmb2 = combineidxs(R, [1, 2])
                R1, S, R2 = svd(R, 2; maxdim=D)
                S = sqrt.(S)

                # Store singular values, restore tensors
                push!(singulars1[i], diag(S))
                R1 = contract(R1, S, 2, 1)
                R2 = contract(S, R2, 2, 1)
                R1 = moveidx(R1, 1, 2)
                R1 = reshape(R1, (size(S)[1], dims1[4], dim(psi)))
                R1 = moveidx(R1, 1, 2)
                R2 = reshape(R2, (size(S)[1], dims2[1], dim(psi)))
                A1 = contract(A1, R1, 4, 1)
                A2 = contract(R2, A2, 2, 1)
                A2 = moveidx(A2, 2, 5)
                psi[i, j] = A1
                psi[i, j+1] = A2
            end

            if i < N
                # Bring tensors together and apply SVD
                A1, R1 = reducedtensor(psi, i, j, 3)
                A2, R2 = reducedtensor(psi, i+1, j, 2)
                dims1 = size(A1)
                dims2 = size(A2)
                D = size(R1)[2]
                R = contract(R1, R2, 2, 1)
                R, cmb1 = combineidxs(R, [1, 2])
                R, cmb2 = combineidxs(R, [1, 2])
                R1, S, R2 = svd(R, 2; maxdim=D)
                S = sqrt.(S)

                # Store singular values, restore tensors
                push!(singulars2[i], diag(S))
                R1 = contract(R1, S, 2, 1)
                R2 = contract(S, R2, 2, 1)
                R1 = moveidx(R1, 1, 2)
                R1 = reshape(R1, (size(S)[1], dims1[3], dim(psi)))
                R1 = moveidx(R1, 1, 2)
                R2 = reshape(R2, (size(S)[1], dims2[2], dim(psi)))
                A1 = contract(A1, R1, 3, 1)
                A1 = moveidx(A1, 4, 3)
                A2 = contract(R2, A2, 2, 2)
                A2 = moveidx(A2, 2, 5)
                A2 = moveidx(A2, 1, 2)
                psi[i, j] = A1
                psi[i+1, j] = A2
            end
        end
    end

    converge = false
    iter = 0
    lastenergy = calculateenergy(psi)
    energy = copy(lastenergy)
    while !converge
        for j = 1:length(gates)
            i = [gates.sites[j][1], gates.sites[j][2], gates.directions[j]]
            gate = gates.gates[j]
            if length(size(gate)) == 2
                A = psi[i[1], i[2]]
                A = contract(A, gate, 5, 2)
                psi[i[1], i[2]] = A
            else
                # Bring in the singular values to the relevent sites
                if i[3]==0
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

                # Find the reduced tensors
                axis1 = i[3]==0 ? 4 : 3
                axis2 = i[3]==0 ? 1 : 2
                A1, R1 = reducedtensor(A1, axis1)
                A2, R2 = reducedtensor(A2, axis2)

                # Contract reduced tensors with gate
                R = contract(R1, R2, 2, 1)
                R = contract(R, gate, [2, 4], [2, 4])
                R, cmb1 = combineidxs(R, [1, 3])
                R, cmb1 = combineidxs(R, [1, 2])
                R1, S, R2 = svd(R, -1; maxdim=maxdim, cutoff=1e-10)
                S = S / maximum(S)
                S = sqrt.(S)
                R1 = contract(R1, S, 2, 1)
                R2 = contract(S, R2, 2, 1)
                R1 = moveidx(R1, 1, 2)
                R1 = reshape(R1, (size(S)[1], size(A1)[axis1], dim(psi)))
                R1 = moveidx(R1, 1, 2)
                R2 = reshape(R2, (size(S)[1], size(A2)[axis2], dim(psi)))

                # Restore full tensors
                A1 = contract(A1, R1, axis1, 1)
                A2 = contract(R2, A2, 2, axis2)
                A2 = moveidx(A2, 2, 5)
                if i[3]==1
                    A1 = moveidx(A1, 4, 3)
                    A2 = moveidx(A2, 1, 2)
                end


                # Multiple by inverse of the singular values
                if i[3]==0
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
                if i[3]==0
                    psi[i[1], i[2]] = A1
                    psi[i[1], i[2]+1] = A2
                    singulars1[i[1]][i[2]] = diag(S)
                else
                    psi[i[1], i[2]] = A1
                    psi[i[1]+1, i[2]] = A2
                    singulars2[i[1]][i[2]] = diag(S)
                end
            end
        end

        # Check convergence
        iter += 1
        converge = iter >= maxiter ? true : converge
        if iter % saveiter == 0
            energy = calculateenergy(psi)
            converge = ((energy-lastenergy) / abs(energy) < tol) ? true : converge
            @printf("iter=%d, energy=%.12f, maxdim=%d \n", iter, energy, maxbonddim(psi))
            lastenergy = energy
        end
        converge = iter < miniter ? false : converge
    end

    return psi, energy

end
