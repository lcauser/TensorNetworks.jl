include("src/TensorNetworks.jl")

# Model parameters
N = 20
s = -1.0
c = 0.5
tmax = 10000.0
save = 1.0

sh = spinhalf()

oplist = OpList(N)

# First Site
add!(oplist, ["x"], [1], 0.5*exp(-s)*sqrt(c*(1-c)))
#add!(oplist, ["x", "pd"], [1, 2], exp(-s)*sqrt(c*(1-c)))
add!(oplist, ["pu"], [1], -0.5*c)
add!(oplist, ["pd"], [1], -0.5*(1-c))
#add!(oplist, ["pu", "pd"], [1, 2], -c)
#add!(oplist, ["pd", "pd"], [1, 2], -(1-c))

# First pair
add!(oplist, ["s+", "s-", "pd"], [1, 2, 3], exp(-s)*sqrt(c*(1-c)))
add!(oplist, ["s-", "s+", "pd"], [1, 2, 3], exp(-s)*sqrt(c*(1-c)))
add!(oplist, ["pd", "pu", "pd"], [1, 2, 3], -c)
add!(oplist, ["pu", "pd", "pd"], [1, 2, 3], -(1-c))
#add!(oplist, ["s+", "s-"], [1, 2], 0.5*exp(-s)*sqrt(c*(1-c)))
#add!(oplist, ["s-", "s+"], [1, 2], 0.5*exp(-s)*sqrt(c*(1-c)))
#add!(oplist, ["pd", "pu"], [1, 2], -0.5*c)
#add!(oplist, ["pu", "pd"], [1, 2], -0.5*(1-c))

for i = 1:N-3
    add!(oplist, ["pu", "s+", "s-"], [i, i+1, i+2], exp(-s)*sqrt(c*(1-c)))
    add!(oplist, ["pu", "s-", "s+"], [i, i+1, i+2], exp(-s)*sqrt(c*(1-c)))
    add!(oplist, ["s+", "s-", "pd"], [i+1, i+2, i+3], exp(-s)*sqrt(c*(1-c)))
    add!(oplist, ["s-", "s+", "pd"], [i+1, i+2, i+3], exp(-s)*sqrt(c*(1-c)))
    add!(oplist, ["pu", "pd", "pu"], [i, i+1, i+2], -c)
    add!(oplist, ["pu", "pu", "pd"], [i, i+1, i+2], -(1-c))
    add!(oplist, ["pd", "pu", "pd"], [i+1, i+2, i+3], -c)
    add!(oplist, ["pu", "pd", "pd"], [i+1, i+2, i+3], -(1-c))
end


# Last pair
add!(oplist, ["pu", "s+", "s-"], [N-2, N-1, N], exp(-s)*sqrt(c*(1-c)))
add!(oplist, ["pu", "s-", "s+"], [N-2, N-1, N], exp(-s)*sqrt(c*(1-c)))
add!(oplist, ["pu", "pd", "pu"], [N-2, N-1, N], -c)
add!(oplist, ["pu", "pu", "pd"], [N-2, N-1, N], -(1-c))
#add!(oplist, ["s+", "s-"], [N-2, N-1],-0.5*exp(-s)*sqrt(c*(1-c)))
#add!(oplist, ["s-", "s+"], [N-2, N-1], 0.5*exp(-s)*sqrt(c*(1-c)))
#add!(oplist, ["pd", "pu"], [N-2, N-1], -0.5*c)
#add!(oplist, ["pu", "pd"], [N-2, N-1], -0.5*(1-c))

# Last site
add!(oplist, ["x"], [N], -0.5*exp(-s)*sqrt(c*(1-c)))
#add!(oplist, ["pu", "x"], [N-1, N], -exp(-s)*sqrt(c*(1-c)))
add!(oplist, ["pd"], [N], 0.5*c)
add!(oplist, ["pu"], [N], 0.5*(1-c))
#add!(oplist, ["pu", "pd"], [N-1, N], c)
#add!(oplist, ["pu", "pu"], [N-1, N], (1-c))

psi = randomMPS(2, N, 2)
movecenter!(psi, 1)

for dt in [0.1, 0.01, 0.001]
    psi, energy = tebd(psi, oplist, sh, dt, tmax, save, [TEBDNorm(1e-10)]; order=1)
end

# Measure Occupations and Correlations
occs = OpList(N)
for i = 1:N
    add!(occs, ["pu"], [i])
end
expectations = inner(sh, psi, occs, psi)
occupations = expectations[1:N]

energy = sum(inner(sh, psi, oplist, psi))
