include("src/TensorNetworks.jl")

N = 10
c = 0.05
s = -0.01
maxdim = 6
cutoff = 0

sh = spinhalf()

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
ops2 = [[op(sh, name) for name in op1] for op1 in ops]

println("--------")
# Find initial guess
states = [["dn" for i = 1:N] for j = 1:N]
states[1][1] = "up"
states[N][N] = "s"
psi1 = productPEPS(sh, states)
#psi1, energy1 = simpleupdate(psi1, 0.1, sh, ops, coeffs; maxiter=100000, maxdim=1, saveiter=100, chi=1)

states = [["s" for i = 1:N] for j = 1:N]
states[1][1] = "up"
states[N][N] = "s"
psi2 = productPEPS(sh, states)
psi2, energy2 = simpleupdate(psi2, 0.1, sh, ops, coeffs; maxiter=100000, maxdim=1, saveiter=100, chi=1)
psi = energy1 > energy2 ? psi1 : psi2


# Incrase bond dim and converge
energies = []
energy = 0
for D = [1, 2, 3, 4]
    for dt = [1.0, 0.1, 0.01]
        psi, energy = simpleupdate(psi, dt, sh, ops, coeffs; maxiter=100000, miniter=2000, maxdim=D, chi=100)
    end
    #env = Environment(psi, psi; chi=500)
    #energy = real(inner(env, ops2, coeffs) / inner(env))
    println(energy)
    push!(energies, energy)
end

psi, energy = simpleupdate(psi, 0.001, sh, ops, coeffs; maxiter=100000, miniter=2000, maxdim=4, chi=100)

ns = OpList2d(N)
for i = 1:N
    for j = 1:N
        add!(ns, ["n"], [i, j], false, 1)
    end
end
add!(ns, ["id"], [1, 1], false, 1)
ns = inner(sh, psi, ns, psi; maxchi=200)
ns = [ns[i] / ns[end] for i = 1:length(ns)-1]
ns = reshape(ns, (N, N))
