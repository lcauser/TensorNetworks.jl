using HDF5
include("src/TensorNetworks.jl")

N = 30
c = 0.5
ss = -exp10.(range(-3, stop=0, length=61))
ss2 = exp10.(range(-5, stop=1, length=121))
append!(ss, ss2)
push!(ss, 0.0)
unique!(ss)
sort!(ss)
ss = round.(ss, digits=10)
maxdim = 6
cutoff = 0
home = "D:/East Data/2d/PEPS/"

idx = 97

s = ss[idx]


sh = spinhalf()

ops = OpList2d(N)
# Add kinetic terms
kin = 0
for i = 1:N
    for j = 1:N
        p = (i == 1 && j == 1) ? "id" : "n"
        if i <= N-1
            if i == 1 && j == 1
                add!(ops, ["x"], [2, 1], true, sqrt(c*(1-c))*exp(-s))
            else
                add!(ops, [p, "x"], [i, j], true, sqrt(c*(1-c))*exp(-s))
            end
            kin += 1
        end
        if j <= N-1
            if i == 1 && j == 1
                add!(ops, ["x"], [1, 2], false, sqrt(c*(1-c))*exp(-s))
            else
                add!(ops, [p, "x"], [i, j], false, sqrt(c*(1-c))*exp(-s))
            end
            kin += 1
        end
    end
end

# Add escape rate terms
es = 0
for i = 1:N
    for j = 1:N
        p = (i == 1 && j == 1) ? "id" : "n"
        if i <= N-1
            add!(ops, [p, "pu"], [i, j], true, -(1-c))
            add!(ops, [p, "pd"], [i, j], true, -c)
            es += 2
        end
        if j <= N-1
            add!(ops, [p, "pu"], [i, j], false, -(1-c))
            add!(ops, [p, "pd"], [i, j], false, -c)
            es += 2
        end
    end
end

println("--------")
# Find initial guess
directfile = string(home, "c = ", c, "/N = ", N, "/s = ", ss[idx-1], ".h5")
f = h5open(directfile)
psi = read(f, "psi", PEPS)
close(f)


# Incrase bond dim and converge
energies = []
energy = 0
for dt = [0.01, 0.001]
    psi, energy = simpleupdate(psi, dt, sh, ops; maxiter=100000, miniter=2000, maxdim=4, chi=200)
end

# Add occupations
occ = 0
for i = 1:N
    for j = 1:N
        add!(ops, ["n"], [i, j], false, 1)
        occ += 1
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
direct = string(home, "c = ", c, "/N = ", N, "/")
if !isdir(direct)
  mkpath(direct)
end
direct = string(direct, "s = ", s, ".h5")
f = h5open(direct, "w")
write(f, "psi", psi)
write(f, "scgf", scgf)
write(f, "activity", activity)
for i = 1:N
    for j = 1:N
        num = (i-1)*N+j
        write(f, string("occs_", i, "_", j), occs[num])
    end
end
close(f)
