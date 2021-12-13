using HDF5
include("src/TensorNetworks.jl")

Ns = [6, 10, 14, 18, 22, 26, 30]
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

# Create a list of parameters
params = []
for N in Ns
    for s in ss
        push!(params, [N, s])
    end
end

# Find the s value
#idx = parse(Int, ARGS[1])
idx = 1
N = Int(params[idx][1])
s = params[157][2]
N = 10
s = -1.0

sh = spinhalf()

# Create the gate
gate = sqrt(c*(1-c))*exp(-s)*tensorproduct(op(sh, "n"), op(sh, "x"))
gate += -(1-c)*tensorproduct(op(sh, "n"), op(sh, "pu"))
gate += -c*tensorproduct(op(sh, "n"), op(sh, "pd"))
#gate = exp(dt*gate, [2, 4])

ops = Vector(Vector{String}[])
push!(ops, Vector(["n", "x"]))
push!(ops, Vector(["n", "pu"]))
push!(ops, Vector(["n", "pd"]))
coeffs = Vector(Number[sqrt(c*(1-c))*exp(-s), -(1-c), -c])
ops2 = [[op(sh, name) for name in op1] for op1 in ops]

println("--------")
# Find initial guess
states = [["dn" for i = 1:N] for j = 1:N]
states[1][1] = "up"
states[N][N] = "s"
psi1 = productPEPS(sh, states)
psi1, energy1 = simpleupdate(psi1, 0.1, sh, ops, coeffs; maxiter=1000, maxdim=2, saveiter=1000, chi=100)

states = [["s" for i = 1:N] for j = 1:N]
states[1][1] = "up"
states[N][N] = "s"
psi2 = productPEPS(sh, states)
psi2, energy2 = simpleupdate(psi2, 0.1, sh, ops, coeffs; maxiter=2000, maxdim=2, saveiter=1000, chi=100)
psi = energy1 > energy2 ? psi1 : psi2


# Incrase bond dim and converge
energies = []
energy = 0
for D = 2:4
    for dt = [0.01, 0.001]
        @time psi, energy = simpleupdate(psi, dt, sh, ops, coeffs; maxiter=10000, miniter=2000, maxdim=D, chi=2*D^2)
    end
    println(energy)
    push!(energies, energy)
end


ops = OpList2d(N)
# Add kinetic terms
kin = 0
for i = 1:N
    for j = 1:N
        if i <= N-1
            add!(ops, ["n", "x"], [i, j], true, sqrt(c*(1-c))*exp(-s))
            kin += 1
        end
        if j <= N-1
            add!(ops, ["n", "x"], [i, j], false, sqrt(c*(1-c))*exp(-s))
            kin += 1
        end
    end
end

# Add escape rate terms
es = 0
for i = 1:N
    for j = 1:N
        if i <= N-1
            add!(ops, ["n", "pu"], [i, j], true, -(1-c))
            add!(ops, ["n", "pd"], [i, j], true, -c)
            es += 2
        end
        if j <= N-1
            add!(ops, ["n", "pu"], [i, j], false, -(1-c))
            add!(ops, ["n", "pd"], [i, j], false, -c)
            es += 2
        end
    end
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
