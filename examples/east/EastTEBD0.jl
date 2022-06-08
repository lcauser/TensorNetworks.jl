include("src/TensorNetworks.jl")

# Model parameters
N = 10
s = 1.0
c = 0.5
tmax = 10000.0
save = 1.0

sh = spinhalf()

oplist = OpList(N)
add!(oplist, "s+", 1, exp(-s)*c)
add!(oplist, "s-", 1, exp(-s)*(1-c))
add!(oplist, "pu", 1, -(1-c))
add!(oplist, "pd", 1, -c)
for i = 1:N-1
    add!(oplist, ["pu", "s+"], [i, i+1], exp(-s)*c)
    add!(oplist, ["pu", "s-"], [i, i+1], exp(-s)*(1-c))
    add!(oplist, ["pu", "pu"], [i, i+1], -(1-c))
    add!(oplist, ["pu", "pd"], [i, i+1], -c)
end
gates = trotterize(sh, oplist, 0.01)
flat = productMPS(N, [1, 1])

function sample(psi::GMPS)
    flat = productMPS(N, [1, 1])
    proj = ProjMPS(flat, psi)
    d = dim(psi)
    config = zeros(length(psi))
    for j = 1:length(psi)
        movecenter!(psi, j)
        movecenter!(proj, j)
        probs = zeros(d)
        for i = 1:d
            flat[j] = zeros(1, d, 1)
            flat[j][1, i, 1] = 1
            probs[i] = calculate(proj)
        end
        probs = probs / sum(probs)
        r = rand()
        m = findfirst([prob > r for prob in cumsum(probs)])
        flat[j] = zeros(1, d, 1)
        flat[j][1, m, 1] = 1
        config[j] = m
    end

    return config
end

function sample2(psi::GMPS)
    movecenter!(psi, 1)
    d = dim(psi)
    config = zeros(length(psi))

end

# DMRG
psi = productMPS(sh, ["dn" for _ = 1:N])
psi[N][1, :, 1] = [sqrt(c), sqrt(1-c)]
psi, energy = dmrg(psi, MPO(sh, -1*oplist); cutoff=1e-16)

occsList = OpList(N)
for i = 1:N
    add!(occsList, "pu", i, 1)
end
occs = inner(sh, flat, occsList, psi) / inner(flat, psi)
println(real(occs))

# Find the GS
config = [2 - i for i in sample(psi)]
println(config)
psi2 = deepcopy(psi)
psi = productMPS(sh, [j == 1 ? "up" : "dn" for j in config])
z = log(real(inner(flat, psi2)))

for i = 1:10000
    println("--------")
    for j = 1:10
        applygates!(psi, gates; cutoff=1e-12)
    end
    config = [2 - i for i in sample(psi)]
    psi2 = deepcopy(psi)
    occs = inner(sh, flat, occsList, psi) / inner(flat, psi)
    println(real(occs))
    psi = productMPS(sh, [j == 1 ? "up" : "dn" for j in config])
    z += log(real(inner(flat, psi2)))
    println(i)
    println(config)
    println(z / i)
end
