include("src/TensorNetworks.jl")

N = 14
c = 0.5
s = 0.01
dt = 0.01
maxdim = 4
maxchi = 50
cutoff = 0

sh = spinhalf()
states = [["dn" for i = 1:N] for j = 1:N]
states[1][1] = "up"
states[N][N] = "s"
psi = productPEPS(sh, states)

H = OpList2d(N)
for i = 1:N
    for j = 1:N
        if i <= N-1
            add!(H, ["n", "x"], [i, j], true, sqrt(c*(1-c))*exp(-s))
            add!(H, ["n", "pu"], [i, j], true, -(1-c))
            add!(H, ["n", "pd"], [i, j], true, -c)
        end
        if j <= N-1
            add!(H, ["n", "x"], [i, j], false, sqrt(c*(1-c))*exp(-s))
            add!(H, ["n", "pu"], [i, j], false, -(1-c))
            add!(H, ["n", "pd"], [i, j], false, -c)
        end
    end
end
add!(H, ["id"], [1, 1], true, 1)
println("---")
#env = Environment(psi, psi)

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

println("--------")
psi, energy = simpleupdate(psi, 0.1, sh, ops, coeffs; maxiter=100000, miniter=200, maxdim=1, saveiter=100)
psi, energy = simpleupdate(psi, 0.1, sh, ops, coeffs; maxiter=100000, miniter=200, maxdim=maxdim, saveiter=100)
energies = []
converge = false
D = 0
lastenergy = 0
energy = 0
count1 = 0
dt = 1.0
while !converge
    dt = dt / 10
    converge2 = false
    count2 = 0
    lastenergy2 = 0
    while !converge2
        D += maxdim^2
        psi, energy = fullupdate(psi, gate, dt, sh, ops, coeffs; maxdim=maxdim, chi=D, miniter=200)
        count2 += 1
        if count2 > 1
            converge2 = (energy - lastenergy2) / abs(energy) < 1e-4 ? true : false
        end
        lastenergy2 = energy
        converge2 = D >= maxchi ? true : converge
    end
    count1 += 1
    if count1 > 1
        converge = (energy - lastenergy) / abs(energy) < 1e-4 ? true : false
    end
    lastenergy = energy
    push!(energies, energy)
end

ns = OpList2d(N)
for i = 1:N
    for j = 1:N
        add!(ns, ["n"], [i, j], false, 1)
    end
end
add!(ns, ["id"], [1, 1], false, 1)
ns = inner(sh, psi, ns, psi; maxchi=200)
#ns = [ns[i] / ns[end] for i = 1:length(ns)-1]
#ns = reshape(ns, (N, N))
