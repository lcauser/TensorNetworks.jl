include("src/TensorNetworks.jl")

# Model parameters
N = 8
s = -0.1
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
for D = [2, 4, 8, 16, 32, 64, 128, 256, 512, 1024]
    @time psi, energy = dmrg(psi, H; maxsweeps=1, cutoff=0, maxdim=D)
    @time psi, energy = dmrg(psi, H; maxsweeps=1000, cutoff=0, maxdim=D, minsweeps=1, nsites=1)
end

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
