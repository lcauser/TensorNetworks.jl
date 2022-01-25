using HDF5
include("src/TensorNetworks.jl")

N = 6
cs = [0.1, 0.2, 0.3, 0.4]
ss = -exp10.(range(-3, stop=0, length=61))
ss2 = exp10.(range(-5, stop=1, length=121))
append!(ss, ss2)
push!(ss, 0.0)
unique!(ss)
sort!(ss)
ss = round.(ss, digits=10)
maxdim = 6
cutoff = 0
home = "D:/Test/FA Data/2d/PEPS/"

# Create a list of parameters
params = []
for s in ss
    for c in cs
        push!(params, [c, s])
    end
end

# Find the s value
#idx = parse(Int, ARGS[1])
idx = 1
c = params[idx][1]
s = params[idx][2]
c = 0.5
s = 0.1
println(idx)
println(c)
println(s)

dtstart = s < 0 ? 0.1 : 0.1

sh = spinhalf()

ops = OpList2d(N)
# Add kinetic terms
global kin = 0
for i = 1:N
    for j = 1:N
        p = (i == 1 && j == 1) ? "n" : "n"
        if i <= N-1
            add!(ops, [p, "x"], [i, j], true, sqrt(c*(1-c))*exp(-s))
            add!(ops, ["x", p], [i, j], true, sqrt(c*(1-c))*exp(-s))
            global kin = kin + 2
        end
        if j <= N-1
            add!(ops, [p, "x"], [i, j], false, sqrt(c*(1-c))*exp(-s))
            add!(ops, ["x", p], [i, j], false, sqrt(c*(1-c))*exp(-s))
            global kin = kin + 2
        end
    end
end

# Add escape rate terms
global es = 0
for i = 1:N
    for j = 1:N
        p = (i == 1 && j == 1) ? "n" : "n"
        if i <= N-1
            add!(ops, [p, "pu"], [i, j], true, -(1-c))
            add!(ops, [p, "pd"], [i, j], true, -c)
            add!(ops, ["pu", p], [i, j], true, -(1-c))
            add!(ops, ["pd", p], [i, j], true, -c)
            global es = es + 4
        end
        if j <= N-1
            add!(ops, [p, "pu"], [i, j], false, -(1-c))
            add!(ops, [p, "pd"], [i, j], false, -c)
            add!(ops, ["pu", p], [i, j], false, -(1-c))
            add!(ops, ["pd", p], [i, j], false, -c)
            global es = es + 4
        end
    end
end


# Attempt to load the savefile
direct = string(home, "c = ", c, "/N = ", N, "/")
if !isdir(direct)
  mkpath(direct)
end
direct = string(direct, "s = ", s, ".h5")

if isfile(direct)
    # Load in the properties
    f = h5open(direct)
    global psi = read(f, "psi", PEPS)
    global energy = read(f, "scgf")
    completed = read(f, "completed")
    global iter = read(f, "iter")
    close(f)
else
    completed = false

    # Find initial guess
    up = zeros(ComplexF64, 1, 1, 1, 1, 2)
    up[1, 1, 1, 1, 1] = 1
    dn = zeros(ComplexF64, 1, 1, 1, 1, 2)
    dn[1, 1, 1, 1, 2] = 1
    eq = zeros(ComplexF64, 1, 1, 1, 1, 2)
    eq[1, 1, 1, 1, 1] = sqrt(c)
    eq[1, 1, 1, 1, 2] = sqrt(1-c)

    psi1 = productPEPS(N, eq)
    psi1[1, 1] = up
    psi1, energy1 = simpleupdate(psi1, dtstart, sh, ops; maxiter=2000, maxdim=1, saveiter=1000, chi=50, cutoff=1e-4)
    psi1, energy1 = simpleupdate(psi1, dtstart / 10, sh, ops; maxiter=5000, maxdim=2, saveiter=1000, chi=50, cutoff=1e-4)

    if s > 0
        psi2 = productPEPS(N, dn)
        psi2[1, 1] = up
        psi2, energy2 = simpleupdate(psi2, dtstart, sh, ops; maxiter=10, maxdim=1, saveiter=10, chi=50, cutoff=1e-4)
        psi2, energy2 = simpleupdate(psi2, dtstart / 10, sh, ops; maxiter=5000, maxdim=2, saveiter=1000, chi=50, cutoff=1e-4)
        global psi = energy1 > energy2 ? psi1 : psi2
        global energy = max(energy1, energy2)
    else
        global psi = psi1
        global energy = energy1
    end
    global iter = 1

    # Save file
    f = h5open(direct, "w")
    write(f, "psi", psi)
    write(f, "scgf", energy)
    write(f, "completed", false)
    write(f, "iter", iter)
    close(f)
end

if !completed
    updates = [[2, dtstart/10, 1e-6], [3, dtstart/10, 1e-8], [4, dtstart/10, 1e-10], [4, dtstart/100, 1e-10], [4, dtstart/100, 1e-10]]
    while iter <= 4
        update = updates[iter]
        D = Int(update[1])
        dt = update[2]
        cutoff = update[3]
        @time psi2, energy2 = simpleupdate(psi, dt, sh, ops; maxiter=100000, miniter=2000, maxdim=D, chi=50, cutoff=cutoff)
        global psi = psi2
        global energy = energy2
        global iter = iter + 1

        # Save the current
        f = h5open(direct, "w")
        write(f, "psi", psi)
        write(f, "scgf", energy)
        write(f, "completed", false)
        write(f, "iter", iter)
        close(f)
    end
    println(energy)

    # Add occupations
    global occ = 0
    for i = 1:N
        for j = 1:N
            add!(ops, ["n"], [i, j], false, 1)
            global occ = occ + 1
        end
    end

    # Add identity
    add!(ops, ["id"], [1, 1], false, 1)

    # Calculate expectations, and then calculate energy from this
    expectations = inner(sh, psi, ops, psi; maxchi=200)

    # Calculate scgf
    scgf = sum(expectations[1:kin+es]) / expectations[end]

    # Calculate activity
    activity = sum(expectations[1:kin]) / expectations[end]

    # Occupations
    occs = expectations[kin+es+1:kin+es+occ] / expectations[end]

    # Save the peps
    f = h5open(direct, "w")
    write(f, "psi", psi)
    write(f, "scgf", scgf)
    write(f, "activity", activity)
    write(f, "iter", iter)
    write(f, "completed", true)
    for i = 1:N
        for j = 1:N
            num = (i-1)*N+j
            write(f, string("occs_", i, "_", j), occs[num])
        end
    end
    close(f)
end
