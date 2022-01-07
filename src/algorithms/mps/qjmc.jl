abstract type QJMCObserver end

"""
    qjmcgates(st : Sitetypes, H : OpList, jumpops : OpList, dt; kwargs...)

Calculate the escape operators and effective hamiltonian (as trotter gates)
for some hamiltonian and jump operator list.
"""
function qjmc_gates(st::Sitetypes, H::OpList, jumpops::OpList, dt; kwargs...)
    # Create the escape operators
    escapeops = OpList(jumpops.length)
    for i = 1:length(jumpops.ops)
        sites = jumpops.sites[i]
        coeff = jumpops.coeffs[i] ^ 2
        ops = String[]
        for op in jumpops.ops[i]
            push!(ops, opprod(st, [dag(st, op), op]))
        end
        add!(escapeops, ops, sites, coeff)
    end

    # Create the effective hamiltonian
    Heff = add(-1im*H, -0.5*escapeops)
    gates = trotterize(Heff, st, dt; kwargs...)
    return Heff, gates
end

function qjmc_simulation(psi::MPS, Hs::OpList, jumpops::OpList, st::Sitetypes,
                         tmax::Real, dt::Real, observers=[]; kwargs...)
    # Set the save time and the steps
    save::Real = get(kwargs, :save, 0)
    save = save == 0 ? dt : save
    steps = ceil(round(tmax / dt, digits=5))
    savesteps = ceil(round(save / dt))

    # Truncation
    cutoff::Real = get(kwargs, :cutoff, 1e-12)
    mindim::Int = get(kwargs, :mindim, 1)
    maxdim::Int = get(kwargs, :maxdim, 0)

    # Update jump operators?
    update::Bool = get(kwargs, :update, false)

    # Store quantum jump information
    jumps = []
    jumptimes = []

    # Calculate the escape rates and evolution gates
    escapeops, gates = qjmc_gates(st, H, jumpops, dt; kwargs...)

    # Initiate time and observerations
    time = 0
    for observer in observers
        measure!(observer, time, psi, jumps, jumptimes)
    end

    # Loop through each step and simulate
    for i = 1:steps
        # Quantum simulation
        applygates!(psi, gates; mindim=mindim, maxdim=maxdim, cutoff=cutoff)
        normalize!(psi)


        # Classical simulation
        class_t = 0
        while class_t < dt
            # Calculate jump rates
            rates = qjmc_emission_rates(psi, jumpops, st)

            # Determine a jump time
            jumptime = -log(rand()) / sum(rates)
            class_t += jumptime

            # Do the transition
            if class_t < dt
                # Pick the transition
                r = rand()
                idx = findfirst([r < x for x = (cumsum(rates) / sum(rates))])

                # Apply it
                movecenter!(psi, 1)
                applyop!(psi, st, jumpops.ops[idx], jumpops.sites[idx])
                movecenter!(psi, length(psi))
                movecenter!(psi, 1; cutoff=cutoff, maxdim=maxdim, mindim=mindim)
                normalize!(psi)

                # Store the jump
                push!(jumps, idx)
                push!(jumptimes, time+class_t)
            end
        end

        # Increment time
        time += dt

        # Measure observations and output information
        if i % savesteps == 0
            @printf("time=%.5f, jumps=%d, maxbonddim=%d, activity=%.5f \n", time,
                   length(jumps), maxbonddim(psi), length(jumps)/time)

            for observer in observers
                measure!(observer, time, psi, jumps, jumptimes)
            end
        end
    end

    return jumps, jumptimes
end


function qjmc_emission_rates(psi::MPS, jumpops::OpList, st::Sitetypes)
    # Use the projected norm to determine jump rates
    proj = ProjMPS(psi, psi)
    rates = zeros(ComplexF64, length(jumpops.ops))

    # Loop through each site
    for i = 1:length(psi)
        # Find all jumps which start at the site and move the projection
        idxs = siteindexs(jumpops, i)
        movecenter!(proj, i)

        # Calculate emission rate for each idx
        for idx = idxs
            # Fetch the operators and rlevent blocks
            sites = jumpops.sites[idx]
            ops = jumpops.ops[idx]
            coeff = jumpops.coeffs[idx]^2
            rng = sites[end] - sites[1] + 1
            prod = block(proj, sites[1]-1)
            right = block(proj, sites[end]+1)

            # Determine the expectation
            k = 1
            for j = 1:rng
                site = i + j -1
                A = psi.tensors[site]
                if site in sites
                    O = ops[k]
                    k += 1
                else
                    O = "id"
                end
                O = op(st, O)

                # Contract rho
                prod = contract(prod, conj(A), 1, 1)
                prod = contract(prod, conj(O), 2, 2)
                prod = contract(prod, O, 3, 1)
                prod = contract(prod, A, [1, 3], [1, 2])
            end

            # Contract with right block
            prod = contract(prod, right, [1, 2], [1, 2])

            # Set the expectation
            rates[idx] = coeff*prod[1]
        end
    end

    return abs.(rates)
end


### Observers
"""
    measure!(<:TEBDObserver, time::Number, psi::MPS, jumps::Vector{Int},
             jumptimes::Vector{Number})
Take a measurement during QJMC.
"""
measure!(observer::QJMCObserver, time::Number, psi::MPS, jumps::Vector{Int},
         jumptimes::Vector{Number}) = true


"""
    TEBDOperators(oplist::OpList, st::Sitetypes)

Measure multiple operators of a time evolved MPS.
"""
mutable struct QJMCOperators
    times::Vector{Float64}
    measurements::Vector{Vector{Number}}
    oplist::OpList
    st::Sitetypes
    save::Bool
    savefile::String
end

function QJMCOperators(oplist::OpList, st::Sitetypes, save::Bool = false,
                       savefile::String = "")
    return QJMCOperators([], [], oplist, st, save, savefile)
end

function measure!(observer::QJMCOperators, time::Number, psi::MPS, jumps::Vector,
         jumptimes::Vector)
    push!(observer.times, time)
    push!(observer.measurements, inner(observer.st, psi, observer.oplist, psi))
    println(real(sum(observer.measurements) / length(observer.measurements)))
    #println(real(observer.measurements[end]))

    # Add save
end


mutable struct QJMCActivity
    time::Float64
    jumps::Int
    save::Bool
    savefile::String
end

function QJMCActivity(save::Bool=false, savefile::String="")
    return QJMCActivity(0.0, 0, save, savefile)
end

function measure!(observer::QJMCActivity, time::Number, psi::MPS, jumps::Vector,
         jumptimes::Vector)
    observer.time = time
    observer.jumps = length(jumps)
    if observer.save == true
        f = h5open(observer.savefile, "w")
        write(f, "time", observer.time)
        write(f, "jumps", observer.jumps)
        close(f)
    end
end
