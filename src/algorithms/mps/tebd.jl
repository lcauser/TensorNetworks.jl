abstract type TEBDObserver end

function tebd(psi::MPS, H::OpList, st::Sitetypes, dt::Number, tmax::Number,
              save::Number, observers::Vector=[], projectors=[]; kwargs...)
    # Determine truncation behaviour
    fullerror::Float64 = get(kwargs, :cutoff, 1e-12)
    fulldim::Int = get(kwargs, :maxdim, 0)
    updates::String = get(kwargs, :updates, "fast")
    updates = length(projectors) > 0 ? "full" : "fast"
    if updates == "fast"
        gateserror = fullerror
        gatesdim = fulldim
    elseif updates == "full"
        gateserror = 0
        gatesdim = 0
    else
        error("Only fast or full updates are supported.")
    end
    mindim::Int = get(kwargs, :mindim, 1)

    # Trotterize the oplist
    evol::String = get(kwargs, :evol, "imag")
    !(evol == "imag" || evol == "real") && error("evol must be real of imag.")
    order::Int = get(kwargs, :order, 2)
    !(order == 1 || order == 2) && error("Only first or second order trotter supported.")
    gates = trotterize(H, st, dt; evol=evol, order=order)

    # Process the projectors; MPS, MPO and MPSProjector allowed
    i = 1
    projs = []
    for proj = projectors
        if typeof(proj) == MPS
            push!(projs, MPSProjector(proj, proj))
        elseif typeof(proj) != MPO && typeof(proj) != MPSProjector
            error("Only MPS, MPO and MPSProjectors are supported as projectors.")
        else
            push!(projs, proj)
        end
    end

    # Determine number of steps
    nsteps = Int(round(tmax / dt))
    save = save < dt ? dt : save
    nsave = Int(round(save / dt))

    # Initial norm
    normal::Float64 = get(kwargs, :norm, 0)

    # Make initial observables
    for observer = observers
        measure!(observer, 0.0, psi, normal)
    end

    # Repeatedly evolve in time
    converged = false
    step = 0
    psinorm = log(norm(psi))
    while !converged
        # Apply gates
        applygates!(psi, gates; mindim=mindim, maxdim=gatesdim, cutoff=gateserror)

        # Project out
        if length(projectors) > 0
            psis = MPS[psi]
            for i = 1:length(projs)
                push!(psis, -1*(projs[i]*psi))
            end
            psi = vmps(psis, MPS[]; cutoff=fullerror, maxdim=fulldim)
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
            @printf("time=%d, energy=%.12f, maxbonddim=%d \n",
                    step*dt, real(psinorm) / dt, maxbonddim(psi))

            for observer = observers
                measure!(observer, step*dt, psi, normal)
                converged = converged || checkdone!(observer)
            end
        end
    end

    return psi, psinorm / dt

end

function tebd(psi::MPS, H::OpList, st::Sitetypes, dt::Number, tmax::Number,
              save::Number, observers::Vector=[]; kwargs...)
    return tebd(psi, H, st, dt, tmax, save, observers, MPS[]; kwargs...)
end


### Observers
"""
    checkdone!(observer<:TEBDObserver)

Check for convergence.
"""
checkdone!(observer::TEBDObserver) = false


"""
    measure!(observer<:TEBDObserver, time::Float64, psi::MPS, norm::Float64)

Take a measurement of an observer.
"""
measure!(observer::TEBDObserver, time::Number, psi::MPS, norm::Number) = true


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

function measure!(observer::TEBDNorm, time::Number, psi::MPS, norm::Number)
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
    TEBDOperators(oplist::OpList, st::Sitetypes)

Measure multiple operators of a time evolved MPS.
"""
mutable struct TEBDOperators
    times::Vector{Float64}
    measurements::Vector{Vector{Number}}
    oplist::OpList
    st::Sitetypes
end

function TEBDOperators(oplist::OpList, st::Sitetypes)
    return TEBDOperators([], [], oplist, st)
end

function measure!(observer::TEBDOperators, time::Number, psi::MPS, norm::Number)
    push!(observer.times, time)
    push!(observer.measurements, inner(observer.st, psi, observer.oplist, psi))
end
