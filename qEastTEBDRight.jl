include("src/TensorNetworks.jl")

# Model parameters
N = 6
gamma = 1.0
kappa = 1.0
tmax = 10000.0
direct = "D:/qEast Data/"

ss = -exp10.(range(-4, stop=0, length=81))
ss2 = exp10.(range(-4, stop=0, length=81))
append!(ss, ss2)
push!(ss, 0.0)
unique!(ss)
sort!(ss)
ss = round.(ss, digits=10)

#idx = parse(Int, ARGS[1])
idx = 1
s = ss[idx]
s = 0.9

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

mutable struct SaveMPS <: TEBDObserver
    omega::Float64
    psi::MPS
    completed::Bool
    direct::String
end

function measure!(observer::SaveMPS, time::Number, psi::MPS, norm::Number)
    observer.psi = psi
    f = h5open(observer.direct, "w")
    write(f, "psi", psi)
    write(f, "completed", observer.completed)
    write(f, "omega", observer.omega)
    close(f)
end

function checkdone!(observer::TEBDNorm)
    length(observer.times) < 3 && return false
    E1 = observer.measurements[end] - observer.measurements[end-1]
    E1 /= observer.times[end] - observer.times[end-1]
    E2 = observer.measurements[end-1] - observer.measurements[end-2]
    E2 /= observer.times[end-1] - observer.times[end-2]
    if abs(E2+E1) > 1e-12
        diff = abs(E2 - E1) / abs(E2 + E1)
    else
        diff = abs(E2 - E1)
    end
    diff < observer.tol && return true
    return false
end


# See if the simulation has already begun
tempDirect = string(direct, "TEBD Temp/right/gamma = ", gamma, "/N = ", N, "/")
if !isdir(tempDirect)
    mkpath(tempDirect)
end
tempDirect = string(tempDirect, "s = ", s, ".h5")
if isfile(tempDirect)
    # Load in
    f = h5open(tempDirect, "r")
    omega = read(f, "omega")
    completed = read(f, "completed")
    psi = read(f, "psi", MPS)
    close(f)
    saveObs = SaveMPS(omega, psi, completed, tempDirect)
else
    # Create initial guess
    omega = 0
    sh = qKCMSDM(omega, gamma, kappa)
    if s <= 1e-4
        psi = productMPS(N, ss)
    else
        psi = productMPS(sh, ["da" for i = 1:N])
    end
    movecenter!(psi, 1)
    saveObs = SaveMPS(omega, psi, false, tempDirect)
    for dt in [0.1]
        oplist = lindblad_list(N, omega, gamma, kappa, s)
        psi, energy = tebd(psi, oplist, sh, dt, 10000.0, save, [TEBDNorm(1e-5), saveObs]; cutoff=1e-10)
        global psi = psi
    end
end

if !saveObs.completed
    # Loop through omegas
    while true
        sh = qKCMSDM(saveObs.omega, gamma, kappa)
        oplist = lindblad_list(N, saveObs.omega, gamma, kappa, s)
        psi, energy = tebd(psi, oplist, sh, 0.01, tmax, 1.0, [TEBDNorm(1e-7), saveObs]; cutoff=1e-12)
        H = lindblad(N, saveObs.omega, gamma, kappa, s, sh)
        global E = inner(psi, H, psi)
        global var = inner(psi, H, applyMPO(H, psi)) - inner(psi, H, psi)^2
        global psi = psi

        saveDirect = string(direct, "TEBD/right/gamma = ", gamma, "/omega = ",
                            saveObs.omega, "/N = ", N, "/")
        if !isdir(saveDirect)
            mkpath(saveDirect)
        end
        saveDirect = string(saveDirect, "s = ", s, ".h5")
        f = h5open(saveDirect, "w")
        write(f, "psi", psi)
        write(f, "energy", E)
        write(f, "variance", var)
        close(f)

        saveObs.omega >= 1.0 && break
        saveObs.omega += 0.1
        saveObs.omega = round(saveObs.omega, digits=1)
    end
end

# Save the final state
f = h5open(saveObs.direct, "w")
write(f, "psi", saveObs.psi)
write(f, "completed", true)
write(f, "omega", saveObs.omega)
close(f)
