include("src/TensorNetworks.jl")

# Model parameters
N = 10
s = 0
c = 0.5
tmax = 10000.0
save = 1.0

sh = spinhalf()

oplist = OpList(N)
add!(oplist, "x", 1, exp(-s)*sqrt(c*(1-c)))
add!(oplist, "pu", 1, -(1-c))
add!(oplist, "pd", 1, -c)
for i = 1:N-1
    add!(oplist, ["pu", "x"], [i, i+1], exp(-s)*sqrt(c*(1-c)))
    add!(oplist, ["pu", "pu"], [i, i+1], -(1-c))
    add!(oplist, ["pu", "pd"], [i, i+1], -c)
end

# Find the GS
psi = productMPS(sh, ["s" for i = 1:N])
energy = 1
movecenter!(psi, 1)
for dt in [0.1, 0.01, 0.001]
    @time psi, energy = tebd(psi, oplist, sh, dt, tmax, save, [TEBDNorm()])
end

# Find the excited state
psi2 = randomMPS(2, N, 1)
energy2 = 1
movecenter!(psi2, 1)
for dt in [0.1, 0.01, 0.001]
    @time psi2, energy2 = tebd(psi2, oplist, sh, dt, tmax, save, [TEBDNorm()], [psi])
end
