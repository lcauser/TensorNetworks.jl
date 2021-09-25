include("src/TensorNetworks.jl")

N = 4
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
gate = exp(dt*gate, [2, 4])


gatelist = []
for i = 1:N
    j = 1
    while j < N
        push!(gatelist, Any[i, j, false])
        j += 2
    end

    j = 2
    while j < N
        push!(gatelist, Any[i, j, false])
        j += 2
    end
end
for j = 1:N
    i = 1
    while i < N
        push!(gatelist, Any[i, j, true])
        i += 2
    end

    i = 2
    while i < N
        push!(gatelist, Any[i, j, true])
        i += 2
    end
end

function calculateenergy(psi, sh, H)
    energies = inner(sh, psi, H, psi; maxchi=100)
    energy = sum(energies[1:end-1]) / energies[end]
    return real(energy)
end

energy = calculateenergy(psi, sh, H)
println(energy)
lastenergy = -100
energies = []
Ds = []
for D = [2]
    lastenergy = -10
    energy = -9
    while (energy-lastenergy)/abs(energy) > 1e-6
        for iter = 1:100
            for i in gatelist
                applygate!(psi, [i[1], i[2]], gate, i[3]; maxdim=D, cutoff=cutoff)
            end
            rescale!(psi, 1)
            #println(calculateenergy(psi, sh, H))
        end
        lastenergy = energy
        energy = calculateenergy(psi, sh, H)
        println(energy)
        push!(energies, energy)
        push!(Ds, D)
    end
end


ns = OpList2d(N)
for i = 1:N
    for j = 1:N
        add!(ns, ["n"], [i, j], false, 1)
    end
end
add!(ns, ["id"], [1, 1], false, 1)
ns = inner(sh, psi, ns, psi; maxchi=100)
ns = [ns[i] / ns[end] for i = 1:length(ns)-1]
ns = reshape(ns, (N, N))
