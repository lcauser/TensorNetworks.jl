include("src/TensorNetworks.jl")

# Model parameters
N = 40
omega = 0.0
gamma = 0.4
kappa = 1.0
s = 1e-3
dt = 0.01
tmax = 1000.0

sh = qKCMSDM(omega, gamma, kappa)

# Calculate steady state activity
if gamma == kappa
    light = [1, 0]
    dark = [0, 1]
    plight = 0.5
    pdark = 0.5
else
    ss = [4*omega^2+gamma*(kappa+gamma) -2im*omega*(kappa-gamma);
          2im*omega*(kappa-gamma) 4*omega^2+kappa*(kappa+gamma)]
    ss *= (8*omega^2 + 2*(kappa+gamma)^2)^(-1)
    F = eigen(ss)
    eigs = F.values
    vecs = F.vectors
    light = vecs[:, argmin(real(eigs))]
    dark = vecs[:, argmax(real(eigs))]
    plight = eigs[argmin(real(eigs))]
    pdark = eigs[argmax(real(eigs))]
    normalization = plight + pdark
    plight = plight / normalization
    pdark = pdark / normalization
end
ss =[1.0, 0.0, 0.0, 1.0]

function lindblad(N, omega, gamma, kappa, s)
    # Create the MPO
    A1 = 1im*omega*op(sh, "xid") - 0.5*kappa*op(sh, "puid") - 0.5*gamma*op(sh, "pdid")
    A2 = -1im*omega*op(sh, "idx") - 0.5*kappa*op(sh, "idpu") - 0.5*gamma*op(sh, "idpd")
    A3 = kappa*exp(-s)*op(sh, "s+s+") + gamma*exp(-s)*op(sh, "s-s-")
    M = zeros(ComplexF64, 5, 4, 4, 5)
    M[1, :, :, 1] = op(sh, "idid")
    M[2, :, :, 1] = -A1
    M[3, :, :, 1] = -A2
    M[4, :, :, 1] = -A3
    M[5, :, :, 2] = op(sh, "plid")
    M[5, :, :, 3] = op(sh, "idpl")
    M[5, :, :, 4] = op(sh, "plpl")
    M[5, :, :, 5] = op(sh, "idid")

    M1 = copy(M[5:5, :, :, :])
    M1[1, :, :, 1] = -A1-A2-A3

    H = productMPO(N, M)
    H[1] = M1

    return H
end

oplist = OpList(N)
add!(oplist, "xid", 1, 1im*omega)
add!(oplist, "idx", 1, -1im*omega)
add!(oplist, "s-s-", 1, exp(-s)*gamma)
add!(oplist, "s+s+", 1, exp(-s)*kappa)
add!(oplist, "puid", 1, -0.5*kappa)
add!(oplist, "idpu", 1, -0.5*kappa)
add!(oplist, "pdid", 1, -0.5*gamma)
add!(oplist, "idpd", 1, -0.5*gamma)
for i = 1:N-1
    add!(oplist, ["plid", "xid"], [i, i+1], 1im*omega)
    add!(oplist, ["idpl", "idx"], [i, i+1], -1im*omega)
    add!(oplist, ["plpl", "s-s-"], [i, i+1],  exp(-s)*gamma)
    add!(oplist, ["plpl", "s+s+"], [i, i+1], exp(-s)*kappa)
    add!(oplist, ["plid", "puid"], [i, i+1], -0.5*kappa)
    add!(oplist, ["idpl", "idpu"], [i, i+1], -0.5*kappa)
    add!(oplist, ["plid", "pdid"], [i, i+1], -0.5*gamma)
    add!(oplist, ["idpl", "idpd"], [i, i+1], -0.5*gamma)
end

psi = productMPS(N, ss)
#psi = productMPS(sh, ["dn" for i = 1:N])
movecenter!(psi, 1)

for dt in [0.001]
    psi, energy = tebd(psi, oplist, sh, dt, tmax, 1.0, [TEBDNorm(1e-12)]; cutoff=1e-12)
end
H = lindblad(N, omega, gamma, kappa, s)
E = inner(psi, H, psi)
var = inner(psi, H, applyMPO(H, psi)) - inner(psi, H, psi)^2
