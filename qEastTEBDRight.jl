include("src/TensorNetworks.jl")

# Model parameters
N = 10
gamma = 0.0
kappa = 1.0
omegas = [0.0, 0.2, 0.4]
tmax = 10000.0
direct = "D:/qEast Data/"

ss = -exp10.(range(-4, stop=0, length=81))
ss2 = exp10.(range(-4, stop=0, length=81))
append!(ss, ss2)
push!(ss, 0.0)
unique!(ss)
sort!(ss)
ss = round.(ss, digits=10)

params = []
for omega = omegas
    for s = ss
        push!(params, [omega, s])
    end
end

#idx = parse(Int, ARGS[1])
#omega = params[idx][1]
#s = params[idx][2]
omega = 0.4
s = -0.1

# Time step
if s < -1.0
    dt = 0.0001
    save = 0.01
else
    dt = 0.01
    save = 1.0
end

# Calculate steady state
light = [1, 0]
dark = [0, 1]
plight = gamma / (gamma + kappa)
pdark = kappa / (gamma + kappa)
ss = plight*light'.*light + pdark * dark'.*dark
ss = conj(reshape(ss, 4))

function lindblad(N, omega, gamma, kappa, s, sh)
    # Create the MPO
    A1 = -1im*omega*op(sh, "xid") - 0.5*kappa*op(sh, "puid") - 0.5*gamma*op(sh, "pdid")
    A2 = 1im*omega*op(sh, "idx") - 0.5*kappa*op(sh, "idpu") - 0.5*gamma*op(sh, "idpd")
    A3 = kappa*exp(-s)*op(sh, "s-s-") + gamma*exp(-s)*op(sh, "s+s+")
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

function lindblad_list(N, omega, gamma, kappa, s)
    oplist = OpList(N)
    add!(oplist, "xid", 1, -1im*omega)
    add!(oplist, "idx", 1, 1im*omega)
    add!(oplist, "s+s+", 1, exp(-s)*gamma)
    add!(oplist, "s-s-", 1, exp(-s)*kappa)
    add!(oplist, "puid", 1, -0.5*kappa)
    add!(oplist, "idpu", 1, -0.5*kappa)
    add!(oplist, "pdid", 1, -0.5*gamma)
    add!(oplist, "idpd", 1, -0.5*gamma)
    for i = 1:N-1
        add!(oplist, ["plid", "xid"], [i, i+1], -1im*omega)
        add!(oplist, ["idpl", "idx"], [i, i+1], 1im*omega)
        add!(oplist, ["plpl", "s+s+"], [i, i+1],  exp(-s)*gamma)
        add!(oplist, ["plpl", "s-s-"], [i, i+1], exp(-s)*kappa)
        add!(oplist, ["plid", "puid"], [i, i+1], -0.5*kappa)
        add!(oplist, ["idpl", "idpu"], [i, i+1], -0.5*kappa)
        add!(oplist, ["plid", "pdid"], [i, i+1], -0.5*gamma)
        add!(oplist, ["idpl", "idpd"], [i, i+1], -0.5*gamma)
    end
    return oplist
end


# Create initial guess
sh = qKCMSDM(omega, gamma, kappa)
psi1 = productMPS(N, ss)
psi2 = productMPS(sh, ["da" for i = 1:N])
oplist = lindblad_list(N, omega, gamma, kappa, s)
if s > 0
    for dt in [0.01]
        psi1, energy1 = tebd(psi1, oplist, sh, dt, 100.0, save, [TEBDNorm(1e-5)]; maxdim=4)
        global psi1 = psi1
        global energy1 = energy1
        psi2, energy2 = tebd(psi2, oplist, sh, dt, 100.0, save, [TEBDNorm(1e-5)]; maxdim=4)
        global psi2 = psi2
        global energy2 = energy2
    end
    psi = energy1 > energy2 ? psi1 : psi2
    energy = energy1 > energy2 ? energy1 : energy2
else
    psi = psi1
    energy = 0
end

# Do a final evolution
converge = false
D = 2
energy = 0
while !converge
    D *= 2
    lastenergy = energy
    psi, energy = tebd(psi, oplist, sh, 0.01, tmax, 1.0, [TEBDNorm(1e-6)]; maxdim=D, cutoff=1e-20)
    if abs((lastenergy - energy) / (lastenergy + energy)) < 1e-5
        converge = true
    end
    if D >= 32
        converge = true
    end
end
#psi, energy = tebd(psi, oplist, sh, 0.001, tmax, 1, [TEBDNorm(1e-8)]; maxdim=D, cutoff=1e-16)
H = lindblad(N, omega, gamma, kappa, s, sh)
global E = inner(psi, H, psi)
global var = inner(psi, H, applyMPO(H, psi)) - inner(psi, H, psi)^2
global psi = psi

saveDirect = string(direct, "TEBD/right/gamma = ", gamma, "/omega = ",
                    omega, "/N = ", N, "/")
if !isdir(saveDirect)
    mkpath(saveDirect)
end
saveDirect = string(saveDirect, "s = ", s, ".h5")
f = h5open(saveDirect, "w")
write(f, "psi", psi)
write(f, "energy", E)
write(f, "variance", var)
close(f)
println(E)
println(var)

H = lindblad(N, omega, gamma, kappa, s, sh)
psi, energy = dmrg(psi, H; cutoff=1e-16, ishermitian=false, maxsweeps=100, nsites=2)
global E = inner(psi, H, psi)
global var = inner(psi, H, applyMPO(H, psi)) - inner(psi, H, psi)^2
println(E)
println(var)
