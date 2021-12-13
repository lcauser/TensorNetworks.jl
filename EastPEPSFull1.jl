include("src/TensorNetworks.jl")

N = 6
c = 0.5
s = -1.0
dt = 0.01
maxdim = 3
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
@time begin
psi, energy = fullupdate(psi, gate, 0.1, sh, ops, coeffs; maxdim=1, maxiter=10000, miniter=10, chi=1, saveiter=500)
psi, energy = fullupdate(psi, gate, 0.01, sh, ops, coeffs; maxdim=2, maxiter=10000, miniter=10, chi=4*4, dropoff=4, saveiter=200)
psi, energy = fullupdate(psi, gate, 0.01, sh, ops, coeffs; maxdim=3, maxiter=10000, miniter=10, chi=4*9, dropoff=4, saveiter=200)
#psi, energy = fullupdate(psi, gate, 0.01, sh, ops, coeffs; maxdim=4, maxiter=10000, miniter=10, chi=16, dropoff=1, saveiter=200)
psi, energy = fullupdate(psi, gate, 0.001, sh, ops, coeffs; maxdim=3, maxiter=10000, miniter=10, chi=4*9, dropoff=4, saveiter=100)
end

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
