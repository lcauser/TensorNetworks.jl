include("src/TensorNetworks.jl")

# Model parameters
N = 100
s = 0.01
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
psi = productMPS(sh, ["s" for i = 1:N])
movecenter!(psi, 1)

for dt in [1.0, 0.1, 0.01, 0.001]
    @time psi, energy = tebd(psi, oplist, sh, dt, tmax, save, [TEBDNorm()])
end
