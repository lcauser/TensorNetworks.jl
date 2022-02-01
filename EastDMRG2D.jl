include("src/TensorNetworks.jl")

# Model parameters
N = 14
s = 0.1
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
            println("----")
            println(site1)
            println(site2)
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
H = MPO(sh, H)

# Create initial guess
psi = randomMPS(2, N^2, 1)
movecenter!(psi, 1)

# Do DMRG; interactions are "long ranged" so not cutoff, work up to large D
@time psi, energy = dmrg(psi, H; maxsweeps=100, cutoff=0, maxdim=10)
@time psi, energy = dmrg(psi, H; maxsweeps=100, cutoff=0, maxdim=100)
@time psi, energy = dmrg(psi, H; maxsweeps=100, cutoff=0, maxdim=256)
@time psi, energy = dmrg(psi, H; maxsweeps=100, cutoff=0, maxdim=512)
@time psi, energy = dmrg(psi, H; maxsweeps=100, cutoff=0, maxdim=1024)

# Measure Occupations
oplist = OpList(N^2)
for i = 1:N^2
    add!(oplist, ["pu"], [i])
end

expectations = inner(sh, psi, oplist, psi)
occupations = expectations[1:N^2]
occs = zeros(Float64, (N, N))
for i = 1:N
    for j = 1:N
        site = isodd(i) ? (i-1)*N + j : i*N - j + 1
        occs[i, j] = occupations[site]
    end
end
