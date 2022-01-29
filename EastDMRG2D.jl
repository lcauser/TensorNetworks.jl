include("src/TensorNetworks.jl")

# Model parameters
N = 6
s = -0.0
c = 0.5

# Create lattice type
sh = spinhalf()

# Create hamiltonian
H = OpList(N^2)
add!(H, "pd", 1, 1)
for i = 1:N
    for j = 1:N
        site1 = (i-1)*N + j
        if j < N
            site1 = isodd(i) ? (i-1)*N + j : i*N - j + 1
            site2 = isodd(i) ? (i-1)*N + j + 1 : i*N - j
            add!(H, ["pu", "x"], [site1, site2], -exp(-s)*sqrt(c*(1-c)))
            add!(H, ["pu", "pu"], [site1, site2], (1-c))
            add!(H, ["pu", "pd"], [site1, site2], c)
        end

        if i < N
            site1 = isodd(i) ? (i-1)*N + j : i*N - j + 1
            site2 = isodd(i) ? (i+1)*N + 1 - j : i*N + j
            println("----")
            println(site1)
            println(site2)
            add!(H, ["pu", "x"], [site1, site2], -exp(-s)*sqrt(c*(1-c)))
            add!(H, ["pu", "pu"], [site1, site2], (1-c))
            add!(H, ["pu", "pd"], [site1, site2], c)
        end
    end
end

println("----")
H = MPO(H, sh)

# Create initial guess
psi = productMPS(sh, ["dn" for i = 1:N])
psi = randomMPS(2, N, 1)
movecenter!(psi, 1)

# Do DMRG
@time psi1, energy1 = dmrg(psi, H; maxsweeps=100, cutoff=1e-16, maxdim=4)

# Find excited state
psi2 = randomMPS(2, N, 1)
@time psi1, energy2 = dmrg(psi2, H, psi1; maxsweeps=100, cutoff=1e-16)

# Measure Occupations and Correlations
oplist = OpList(N)
for i = 1:N
    add!(oplist, ["pu"], [i])
end
for i = 1:N-1
    add!(oplist, ["pu", "pu"], [i, i+1])
end
expectations = inner(sh, psi1, oplist, psi1)
occupations = expectations[1:N]
correlations = expectations[N+1:end]
