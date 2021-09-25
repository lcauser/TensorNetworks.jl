include("src/TensorNetworks.jl")

N = 6
c = 0.5
s = -1.0
dt = 0.001
maxdim = 1
cutoff = 0

sh = spinhalf()
states = [["s" for i = 1:N] for j = 1:N]
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
env = Environment(psi, psi)

# Create the gate
gate = sqrt(c*(1-c))*exp(-s)*tensorproduct(op(sh, "n"), op(sh, "x"))
gate += -(1-c)*tensorproduct(op(sh, "n"), op(sh, "pu"))
gate += -c*tensorproduct(op(sh, "n"), op(sh, "pd"))



println("-")
psi = simpleupdate(psi, gate, 0.01, H, sh; maxiter=10000, maxdim=1)
lastenergy = calculateenergy(psi, sh, H)
for D = [1, 2, 3, 4]
    psi = simpleupdate(psi, gate, dt, H, sh; maxiter=10000, maxdim=D)
    energy = calculateenergy(psi, sh, H)
    energy-lastenergy < 1e-7 && break
    lastenergy = energy
    println(energy)
end



ns = OpList2d(N)
for i = 1:N
    for j = 1:N
        add!(ns, ["n"], [i, j], false, 1)
    end
end
add!(ns, ["id"], [1, 1], false, 1)
ns = inner(sh, psi, ns, psi; maxchi=100)
#ns = [ns[i] / ns[end] for i = 1:length(ns)-1]
#ns = reshape(ns, (N, N))
