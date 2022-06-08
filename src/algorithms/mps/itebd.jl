abstract type iTEBDObserver end

function itebd(st::Sitetypes, psi::iGMPS, H::OpList, dt::Number, tmax::Number,
               observers::Vector=[]; kwargs...)
    # Check properties line up
    st.dim != dim(psi) && error("Sitetypes and MPS physical dimensions do not match.")
    length(psi) != length(H) && error("OpList and MPS have different lengths.")

    # Determine truncation behaviour
    cutoff::Float64 = get(kwargs, :cutoff, 1e-12)
    maxdim::Int = get(kwargs, :maxdim, 0)
    mindim::Int = get(kwargs, :mindim, 1)

    # Trotterize the oplist
    evol::String = get(kwargs, :evol, "imag")
    !(evol == "imag" || evol == "real") && error("evol must be real of imag.")
    length(H) != length(psi) && error("The evolution operator list must be the same length as the iMPS.")
    gate = sitetensor(H, st, 1)
    gate = exp(dt*gate, [2*i for i = 1:length(psi)])

    # Checks on the iGMPS being evolved


    # Iterations
    maxiters = Int(round(tmax / dt, digits=0))
    time = 0.0
    observer_times::Vector{Float64} = get(kwargs, :times, [])
    save_time::Float64 = get(kwargs, :save_time, dt)
    if observer_times == []
        observer_times = 0.0:save_time:tmax
    end


    # Make initial observers
    if 0.0 in observer_times
        build!(psi)
        energy = sum(inner(st, psi, H))
        for observer = observers
            measure!(observer, 0.0, psi, energy)
        end
    end

    # Evolve the iMPS
    for iter = 1:maxiters
        if psi.rank == 1
            itebd_apply_gates_mps!(psi, gate, mindim, maxdim, cutoff)
        else
            itebd_apply_gates_mpo!(psi, gate, mindim, maxdim, cutoff)
        end
        time = round(time + dt, digits=8)
        if time in observer_times
            build!(psi)
            energy = sum(inner(st, psi, H))
            for observer = observers
                measure!(observer, 0.0, psi, energy)
            end
            @printf("time=%.6f, energy=%.8e, maxbonddim=%d \n",
                    time, real(energy), maxbonddim(psi))
        end
    end

    return psi
end


function itebd_apply_gates_mps!(psi::iGMPS, gate::Array{ComplexF64}, mindim::Int,
                               maxdim::Int, cutoff::Float64)
    # Loop through the cell
    for i = 1:2
        # Find the order of tensor
        idxs = [i-1, i] .% 2 .+ 1

        # Do the contraction
        prod = diagm(psi.singulars[idxs[1]])
        prod = contract(prod, psi.tensors[idxs[1]], 2, 1)
        prod = contract(prod, diagm(psi.singulars[idxs[2]]), 3, 1)
        prod = contract(prod, psi.tensors[idxs[2]], 3, 1)
        prod = contract(prod, diagm(psi.singulars[idxs[1]]), 4, 1)

        # Apply gate
        prod = contract(gate, prod, [2, 4], [2, 3])
        prod = moveidx(prod, 1, 3)
        prod = moveidx(prod, 1, 3)

        # SVD
        prod, cmb = combineidxs(prod, [1, 2])
        prod, cmb = combineidxs(prod, [1, 2])
        U, S, V = svd(prod, 2; mindim=mindim, maxdim=maxdim, cutoff=cutoff)

        # Reshape into the correct tensors
        U = moveidx(U, 1, 2)
        U = reshape(U, (size(S)[1], size(psi.tensors[idxs[1]])[1], psi.dim))
        U = moveidx(U, 1, 3)
        V = reshape(V, (size(S)[2], psi.dim, size(psi.tensors[idxs[2]])[3]))

        # Multiply in singular values
        U = contract(diagm(1 ./ psi.singulars[idxs[1]]), U, 2, 1)
        V = contract(V, diagm(1 ./ psi.singulars[idxs[1]]), 3, 1)

        # Stop S growing to large
        S = diag(S)
        psi.norms[idxs[2]] += log(sqrt(sum(S.^2)))
        S ./= sqrt(sum(S.^2))

        # Update tensors
        psi.singulars[idxs[2]] = S
        psi.tensors[idxs[1]] = U
        psi.tensors[idxs[2]] = V
    end
end


function itebd_apply_gates_mpo!(psi::iGMPS, gate::Array{ComplexF64}, mindim::Int,
                               maxdim::Int, cutoff::Float64)
    # Loop through the cell
    for i = 1:2
       # Find the order of tensor
       idxs = [i-1, i] .% 2 .+ 1

       # Do the contraction
       prod = diagm(psi.singulars[idxs[1]])
       prod = contract(prod, psi.tensors[idxs[1]], 2, 1)
       prod = contract(prod, diagm(psi.singulars[idxs[2]]), 4, 1)
       prod = contract(prod, psi.tensors[idxs[2]], 4, 1)
       prod = contract(prod, diagm(psi.singulars[idxs[1]]), 6, 1)

       # Apply gate
       prod = contract(gate, prod, [2, 4], [2, 4])
       prod = moveidx(prod, 1, 3)
       prod = moveidx(prod, 1, 4)

       # SVD
       prod, cmb = combineidxs(prod, [1, 2, 3])
       prod, cmb = combineidxs(prod, [1, 2, 3])
       U, S, V = svd(prod, 2; mindim=mindim, maxdim=maxdim, cutoff=cutoff)

       # Reshape into the correct tensors
       U = moveidx(U, 1, 2)
       U = reshape(U, (size(S)[1], size(psi.tensors[idxs[1]])[1], psi.dim, psi.dim))
       U = moveidx(U, 1, 4)
       V = reshape(V, (size(S)[2], psi.dim, psi.dim, size(psi.tensors[idxs[2]])[4]))

       # Multiply in singular values
       U = contract(diagm(1 ./ psi.singulars[idxs[1]]), U, 2, 1)
       V = contract(V, diagm(1 ./ psi.singulars[idxs[1]]), 4, 1)

       # Stop S growing to large
       S = diag(S)
       psi.norms[idxs[2]] += log(sqrt(sum(S.^2)))
       S ./= sqrt(sum(S.^2))

       # Update tensors
       psi.singulars[idxs[2]] = S
       psi.tensors[idxs[1]] = U
       psi.tensors[idxs[2]] = V
    end
end

### Observers
"""
    checkdone!(observer:<iTEBDObserver)

Check for convergence.
"""
checkdone!(observer::iTEBDObserver) = false


"""
    measure!(observer<:iTEBDObserver, time::Float64, psi::GMPS, norm::Float64, energy::Number)

Take a measurement of an observer.
"""
measure!(observer::iTEBDObserver, time::Number, psi::iGMPS, energy::Number) = true


mutable struct iTEBDNorm
    times::Vector{Float64}
    measurements::Vector{Float64}
    tol::Float64
end

function iTEBDNorm(tol::Float64 = 1e-10)
    return iTEBDNorm([], [], tol)
end

function measure!(observer::iTEBDNorm, time::Number, psi::iGMPS, energy::Number)
    push!(observer.times, time)
    push!(observer.measurements, real(norm(psi)))
    println(observer.measurements[end])
end

function checkdone!(observer::iTEBDNorm)
    return false
end
