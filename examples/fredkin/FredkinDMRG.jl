#using .TensorNetworks

# Model parameters
N = 40
s = 1.0
c = 0.7

function FredkinHamiltonian(sh, N, c, s)
    H = OpList(N)

    # Add hamiltoiam
    add!(H, ["s+", "s-", "pd"], [1, 2, 3], -exp(-s)*sqrt(c*(1-c)))
    add!(H, ["pd", "pu", "pd"], [1, 2, 3], c)
    add!(H, ["s-", "s+", "pd"], [1, 2, 3], -exp(-s)*sqrt(c*(1-c)))
    add!(H, ["pu", "pd", "pd"], [1, 2, 3], 1-c)
    for i = 2:N-2
        add!(H, ["pu", "s+", "s-"], [i-1, i, i+1], -exp(-s)*sqrt(c*(1-c)))
        add!(H, ["pu", "pd", "pu"], [i-1, i, i+1], c)
        add!(H, ["pu", "s-", "s+"], [i-1, i, i+1], -exp(-s)*sqrt(c*(1-c)))
        add!(H, ["pu", "pu", "pd"], [i-1, i, i+1], 1-c)

        add!(H, ["s+", "s-", "pd"], [i, i+1, i+2], -exp(-s)*sqrt(c*(1-c)))
        add!(H, ["pd", "pu", "pd"], [i, i+1, i+2], c)
        add!(H, ["s-", "s+", "pd"], [i, i+1, i+2], -exp(-s)*sqrt(c*(1-c)))
        add!(H, ["pu", "pd", "pd"], [i, i+1, i+2], 1-c)
    end
    add!(H, ["pu", "s+", "s-"], [N-2, N-1, N], -exp(-s)*sqrt(c*(1-c)))
    add!(H, ["pu", "pd", "pu"], [N-2, N-1, N], c)
    add!(H, ["pu", "s-", "s+"], [N-2, N-1, N], -exp(-s)*sqrt(c*(1-c)))
    add!(H, ["pu", "pu", "pd"], [N-2, N-1, N], 1-c)

    # Add boundaries
    add!(H, "pd", 1, -10)
    add!(H, "pu", N, -10)
    return MPO(sh, H)
end

function OccsConstraint(sh, occs)
    H = OpList(N)
    for i = 1:N
        add!(H, "pu", i, (1-2*occs[i]))
        add!(H, "id", i, occs[i]^2)
    end
    return MPO(sh, H)
end


# Spin half
sh = spinhalf()
Hfull = FredkinHamiltonian(sh, N, c, s)
excited = []
excited_energies = []
excited_variances = []
excited_entropies = []
for Np = 6:2:N
    # Prepare initial MPS
    psiGS = productMPS(sh, [isodd(i) ? "up" : "dn" for i = 1:Np])
    psiES = productMPS(sh, ["up", [isodd(i) ? "up" : "dn" for i = 1:Np-2]..., "dn"])

    # Create the Hamiltonian
    H = FredkinHamiltonian(sh, Np, c, s)
    psiGS, energy = dmrg(psiGS, -1*H)
    psiES, energy = dmrg(psiES, -1*H, psiGS)
    push!(excited, psiES)

    # Expand
    if Np == N
        psi = psiES
    else
        psi = MPS(2, N)
        for i = 1:N
            if i <= (N-Np)/2
                psi[i] = reshape(state(sh, "up"), (1, 2, 1))
            elseif i > (N + Np) / 2
                psi[i] = reshape(state(sh, "dn"), (1, 2, 1))
            else
                psi[i] = psiES[i - Int((N-Np)/2)]
            end
        end
    end
    E = inner(psi, Hfull, psi)
    var = inner(psi, Hfull, Hfull, psi) - E^2
    if real(var / E^2) < 1e-1
        push!(excited_energies, E)
        push!(excited_variances, var)
        push!(excited_entropies, entropy(psi, Int(N/2)))
    end
    println("----")
    println(E)
    psi, E = dmrgx(psi, -1*Hfull)
end

# GS
psiAGS = productMPS(sh, [isodd(i) ? "up" : "dn" for i = 1:N])
psiAGS, energyAGS = dmrg(psiAGS, -1*H)
psiGS = productMPS(sh, [i <= N/2 ? "up" : "dn" for i = 1:N])
psiGS, energyGS = dmrg(psiGS, 1*H)

# Merging GSs
gs = []
for Np = 10:2:N-2
    psiGS = productMPS(sh, [i <= N/2 ? "up" : "dn" for i = 1:Np])
    H = FredkinHamiltonian(sh, Np, c, s)
    psiGS, energyGS = dmrg(psiGS, 1*H)
    push!(gs, psiGS)
end

gs_energies = []
gs_variances = []
gs_entropies = []
for i = 2:Int(N/2)-1
    psi = MPS(2, N)
    psi1 = gs[i-1]
    psi2 = gs[length(gs)+1-i]
    for j = 1:N
        if j <= 2*i
            psi[j] = psi1[j]
        else
            psi[j] = psi2[j-2*i]
        end
    end
    E = inner(psi, Hfull, psi)
    var = inner(psi, Hfull, Hfull, psi) - E^2
    if real(var / E^2) < 1e-1
        push!(excited_energies, E)
        push!(excited_variances, var)
        push!(excited_entropies, entropy(psi, Int(N/2)))
    end
end


plot(real(excited_energies), real(excited_entropies), seriestype = :scatter, ylim=[0, 1.0], color="blue")
plot!([real(energyGS), -real(energyAGS)], [real(entropy(psiGS, Int(N/2))), real(entropy(psiAGS, Int(N/2)))], seriestype = :scatter, color="black")
