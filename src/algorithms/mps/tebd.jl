abstract type TEBDObserver end

function tebd(st::Sitetypes, psi::GMPS, H::OpList, dt::Number, tmax::Number,
              save::Number, observers::Vector=[], projectors=[]; kwargs...)
    # Check properties line up
    st.dim != dim(psi) && error("Sitetypes and MPS physical dimensions do not match.")
    length(psi) != length(H) && error("OpList and MPS have different lengths.")

    # Determine truncation behaviour
    cutoff::Float64 = get(kwargs, :cutoff, 1e-12)
    variational_cutoff::Float64 = get(kwargs, :variational_cutoff, cutoff)
    maxdim::Int = get(kwargs, :maxdim, 0)
    mindim::Int = get(kwargs, :mindim, 1)

    # Trotterize the oplist
    evol::String = get(kwargs, :evol, "imag")
    !(evol == "imag" || evol == "real") && error("evol must be real of imag.")
    order::Int = get(kwargs, :order, 2)
    !(order == 1 || order == 2) && error("Only first or second order trotter supported.")
    gates = trotterize(st, H, dt; evol=evol, order=order)

    # Process the projectors; MPS, MPO and MPSProjector allowed
    projsteps::Int = get(kwargs, :projection_every, 10)
    i = 1
    projs = []
    for proj = projectors
        if rank(proj) == 1
            push!(projs, MPSProjector(proj, proj))
        elseif rank(proj) != 2 && typeof(proj) != MPSProjector
            error("Only MPS, MPO and MPSProjectors are supported as projectors.")
        else
            push!(projs, proj)
        end
    end
    if length(projectors) > 0
        psis = GMPS[psi]
        for i = 1:length(projs)
            push!(psis, -1*(projs[i]*psi))
        end
        psi = vmps(psis...; cutoff=variational_cutoff, maxdim=maxdim)
    end


    # Determine number of steps
    nsteps = Int(round(tmax / dt))
    save = save < dt ? dt : save
    nsave = Int(round(save / dt))

    # Initial norm
    normal::Float64 = get(kwargs, :norm, 0)
    energy = real(sum(inner(st, psi, H, psi)))

    # Make initial observables
    for observer = observers
        measure!(observer, 0.0, psi, normal, energy)
    end

    # Repeatedly evolve in time
    converged = false
    step = 0
    psinorm = log(norm(psi))
    while !converged
        # Apply gates
        applygates!(psi, gates; mindim=mindim, maxdim=maxdim, cutoff=cutoff)

        # Project out
        if length(projectors) > 0 && ((step + 1) % projsteps) == 0
            psis = GMPS[psi]
            for i = 1:length(projs)
                push!(psis, -1*(projs[i]*psi))
            end
            psi = vmps(psis...; cutoff=variational_cutoff, maxdim=maxdim)
        end

        # Renormalize
        psinorm = log(real(norm(psi)))
        normal += psinorm
        normalize!(psi)

        # Increase steps and check convergence
        step += 1
        if step >= nsteps
            converged = true
        end

        # Print information and check for convergence
        if step % nsave == 0
            @printf("time=%.4f, energy=%.12f, maxbonddim=%d \n",
                    step*dt, energy, maxbonddim(psi))
            energy = real(sum(inner(st, psi, H, psi)))
            for observer = observers
                measure!(observer, step*dt, psi, normal, energy)
                converged = converged || checkdone!(observer)
            end
        end
    end

    return psi, energy

end

function tebd(st::Sitetypes, psi::GMPS, H::OpList, dt::Number, tmax::Number,
              save::Number, observers::Vector=[]; kwargs...)
    return tebd(st, psi, H, dt, tmax, save, observers, GMPS[]; kwargs...)
end


### Observers
"""
    checkdone!(observer<:TEBDObserver)

Check for convergence.
"""
checkdone!(observer::TEBDObserver) = false


"""
    measure!(observer<:TEBDObserver, time::Float64, psi::GMPS, norm::Float64, energy::Number)

Take a measurement of an observer.
"""
measure!(observer::TEBDObserver, time::Number, psi::GMPS, norm::Number, energy::Number) = true


"""
    TEBDNorm(tol::Float64 = 1e-10)

Create an observer to measure the log norm of the MPS during TEBD. The tol
defines the convergence criteria for how the difference in the "energy",
change in log norm / change in time.
"""
mutable struct TEBDNorm
    times::Vector{Float64}
    measurements::Vector{Float64}
    tol::Float64
end

function TEBDNorm(tol::Float64 = 1e-10)
    return TEBDNorm([], [], tol)
end

function measure!(observer::TEBDNorm, time::Number, psi::GMPS, norm::Number, energy::Number)
    push!(observer.times, time)
    push!(observer.measurements, norm)
end

function checkdone!(observer::TEBDNorm)
    length(observer.times) < 3 && return false
    E1 = observer.measurements[end] - observer.measurements[end-1]
    E1 /= observer.times[end] - observer.times[end-1]
    E2 = observer.measurements[end-1] - observer.measurements[end-2]
    E2 /= observer.times[end-1] - observer.times[end-2]
    if abs(E2 + E1) < 1e-8
        abs(E2 - E1) < observer.tol && return true
    else
        abs((E2 - E1) / (0.5*(E2 + E1))) < observer.tol && return true
    end
    return false
end


"""
    TEBDEnergy(tol::Float64 = 1e-10)

Create an observer to measure the energy of the MPS during TEBD. The tol
defines the convergence criteria for how the difference in the "energy".
"""
mutable struct TEBDEnergy
    times::Vector{Float64}
    measurements::Vector{Float64}
    tol::Float64
end

function TEBDEnergy(tol::Float64 = 1e-10)
    return TEBDEnergy([], [], tol)
end

function measure!(observer::TEBDEnergy, time::Number, psi::GMPS, norm::Number, energy::Number)
    push!(observer.times, time)
    push!(observer.measurements, energy)
end

function checkdone!(observer::TEBDEnergy)
    length(observer.times) < 3 && return false
    E1 = observer.measurements[end]
    E2 = observer.measurements[end-1]
    if 0.5*abs(E2 + E1) < 1e-8
        abs(E2 - E1) < observer.tol && return true
    else
        abs(2*(E2 - E1)/(E2 + E1)) < observer.tol && return true
    end
    return false
end


"""
    TEBDOperators(oplist::OpList, st::Sitetypes)

Measure multiple operators of a time evolved MPS.
"""
mutable struct TEBDOperators
    times::Vector{Float64}
    measurements::Vector{Vector{Number}}
    oplist::OpList
    st::Sitetypes
end

function TEBDOperators(st::Sitetypes, oplist::OpList)
    return TEBDOperators([], [], oplist, st)
end

function measure!(observer::TEBDOperators, time::Number, psi::GMPS, norm::Number, energy::Number)
    push!(observer.times, time)
    push!(observer.measurements, inner(observer.st, psi, observer.oplist, psi))
end

function checkdone!(observer::TEBDOperators)
    return false
end
