using HDF5
include("src/TensorNetworks.jl")

# Model parameters
kappa = 1.0
omega = 0.2
gamma = 0.4
Ds = [2, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100, 200, 300, 400, 500]
numBins = 200
directHome = "D:/qEast Data/DMRG/"

Ns = [6, 10, 20, 40, 60, 80, 100]
scs = [0.07511, 0.03355, 0.01190, 0.004739, 0.002990, 0.002117, 0.001682]
ss = -exp10.(range(-5, stop=1, length=121))
ss2 = exp10.(range(-5, stop=0, length=101))
append!(ss, ss2)
push!(ss, 0.0)
unique!(ss)
sort!(ss)
ss = round.(ss, digits=10)

bins = [[] for i = 1:numBins]
i = 0
for j = 1:length(Ns)
    for s in ss
        push!(bins[i+1], [j, s])
        i += 1
        i = i % numBins
    end
end


#idx = parse(Int, ARGS[1])
idx = 1
params = bins[idx]
params = [[5, 1e-3]]


# Create lattice type
sh = qKCMSDM(omega, gamma, kappa)

function lindblad(N, omega, gamma, kappa, s)
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


function lindbladdag(N, omega, gamma, kappa, s)
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

function activityop(N, gamma, kappa, s)
    # Create the MPO
    A3 = kappa*exp(-s)*op(sh, "s-s-") + gamma*exp(-s)*op(sh, "s+s+")
    M = zeros(ComplexF64, 3, 4, 4, 3)
    M[1, :, :, 1] = op(sh, "idid")
    M[2, :, :, 1] = -A3
    M[3, :, :, 2] = op(sh, "plpl")
    M[3, :, :, 3] = op(sh, "idid")

    M1 = copy(M[3:3, :, :, :])
    M1[1, :, :, 1] = -A3

    H = productMPO(N, M)
    H[1] = M1

    return H
end

# Calculate steady state activity
if gamma == kappa
    light = [1, 0]
    dark = [0, 1]
    plight = 0.5
    pdark = 0.5
    ss = [1 0; 0 1]
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
ss = conj(reshape(ss, 4))

observations = []
for param in params
    # Load parameters
    N = Ns[Int(param[1])]
    s = param[2]
    sc = scs[Int(param[1])]

    # Right eigenvector
    H = lindblad(N, omega, gamma, kappa, s)
    if s > sc
        psi = productMPS(sh, ["da" for i = 1:N])
    else
        psi = productMPS(N, ss)
    end
    energy = 0
    var = 0
    for D in Ds
        psi, energy = dmrg(psi, H, ishermitian=false, maxdim=D, maxsweeps=100, tol=1e-10; cutoff=0)
        var = inner(psi, H, applyMPO(H, psi)) - inner(psi, H, psi)^2
        println("D=", maxbonddim(psi), ", energy=", energy, ", variance=", var)
        D > maxbonddim(psi) && break
    end

    psiright = copy(psi)
    energyright = copy(energy)
    varright = copy(var)

    # Left eigenvector
    H = lindbladdag(N, omega, gamma, kappa, s)
    if s > sc
        psi = productMPS(sh, ["da" for i = 1:N])
    else
        psi = productMPS(N, [1, 0, 0, 1])
    end
    for D in Ds
        psi, energy = dmrg(psi, H, ishermitian=false, maxdim=D, maxsweeps=50, tol=1e-10; cutoff=1e-20)
        var = inner(psi, H, applyMPO(H, psi)) - inner(psi, H, psi)^2
        println("D=", maxbonddim(psi), ", energy=", energy, ", variance=", var)
        D > maxbonddim(psi) && break
    end

    psileft = copy(psi)
    energyleft = copy(energy)
    varleft = copy(var)

    direct = string(directHome, "omega=", omega, " gamma=", gamma, "/N = ", N, "/")
    if !isdir(direct)
      mkpath(direct)
    end
    direct = string(direct, "s = ", s, ".h5")
    directObserver = string(directHome, "observables/omega=", omega, " gamma=", gamma, "/N = ", N, "/")
    if !isdir(directObserver)
      mkpath(directObserver)
    end
    directObserver = string(directObserver, "s = ", s, ".h5")

    f = h5open(direct, "w")
    write(f, "psileft", psileft)
    write(f, "psiright", psiright)
    close(f)

    f = h5open(directObserver, "w")
    write(f, "energyleft", energyleft)
    write(f, "energyright", energyright)
    write(f, "varleft", varleft)
    write(f, "varright", varright)

    Z = inner(psileft, psiright)
    activity = inner(psileft, activityop(N, gamma, kappa, s), psiright) / Z
    write(f, "activity", activity)

    oplist = OpList(N)
    for i = 1:N
        add!(oplist, "puid", i)
        add!(oplist, "plid", i)
    end
    observations = real(inner(sh, psileft, oplist, psiright) / Z)
    for i = 1:N
        write(f, string("pu_", i), observations[2*i-1])
        write(f, string("pl_", i), observations[2*i])
        eeleft = entropy(psileft, i)
        eeright = entropy(psiright, i)
        write(f, string("eeleft_", i), eeleft)
        write(f, string("eeright_", i), eeright)
    end
    close(f)
end
