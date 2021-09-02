include("src/TensorNetworks.jl")

kappa = 1
omega = 0.2
gamma = 1
directHome = "D:/qEast Data/"

Ns = [6, 10, 20, 40, 60, 80, 100]
scs = [0.07511, 0.03355, 0.01190, 0.004739, 0.002990, 0.002117, 0.001682]
ss = -exp10.(range(-5, stop=0, length=26))
ss2 = exp10.(range(-5, stop=0, length=26))
append!(ss, ss2)
push!(ss, 0.0)
unique!(ss)
sort!(ss)
ss = round.(ss, digits=10)

params = []
for N in Ns
    for s in ss
        push!(params, [N, s])
    end
end

idx = 1
param = params[idx]
N = Int(param[1])
s = param[2]
N = 10
s = 1.0

function activityop(N, gamma, kappa, s)
    # Create the MPO
    A3 = kappa*exp(-s)*op(sh2, "s-s-") + gamma*exp(-s)*op(sh2, "s+s+")
    M = zeros(ComplexF64, 3, 4, 4, 3)
    M[1, :, :, 1] = op(sh2, "idid")
    M[2, :, :, 1] = -A3
    M[3, :, :, 2] = op(sh2, "plpl")
    M[3, :, :, 3] = op(sh2, "idid")

    M1 = copy(M[3:3, :, :, :])
    M1[1, :, :, 1] = -A3

    H = productMPO(N, M)
    H[1] = M1

    return H
end


function qjmc_emission_rates(psi::MPS, rho::MPO, s::Number, jumpops::OpList, st::Sitetypes)
    # Use the projected norm to determine jump rates
    projrho = ProjMPO(psi, rho)
    rates = zeros(ComplexF64, length(jumpops.ops))

    # Loop through each site
    for i = 1:length(psi)
        # Find all jumps which start at the site and move the projection
        idxs = siteindexs(jumpops, i)
        movecenter!(projrho, i)

        # Calculate emission rate for each idx
        for idx = idxs
            # Fetch the operators and rlevent blocks
            sites = jumpops.sites[idx]
            ops = jumpops.ops[idx]
            coeff = jumpops.coeffs[idx]^2
            rng = sites[end] - sites[1] + 1
            prod = block(projrho, sites[1]-1)
            right = block(projrho, sites[end]+1)

            # Determine the expectation
            k = 1
            for j = 1:rng
                site = i + j -1
                A = psi.tensors[site]
                P = rho[site]
                if site in sites
                    O = ops[k]
                    k += 1
                else
                    O = "id"
                end
                O = op(st, O)

                # Contract rho
                prod = contract(prod, conj(A), 1, 1)
                prod = contract(prod, O, 3, 2)
                prod = contract(prod, P, 1, 1)
                prod = trace(prod, 3, 4)
                prod = contract(prod, O, 3, 1)
                prod = contract(prod, A, 1, 1)
                prod = trace(prod, 3, 4)
            end

            # Contract with right block
            prod = contract(prod, right, 1, 1)
            prod = trace(prod, 2, 4)
            prod = trace(prod, 1, 2)

            # Set the expectation
            rates[idx] = coeff*prod[1]
        end
    end

    # Find the left component of current state
    leftPsi = contract(block(projrho, length(psi)-1), block(projrho, length(psi)), 1, 1)
    leftPsi = trace(leftPsi, 2, 4)
    leftPsi = trace(leftPsi, 1, 2)[1]
    rates = [exp(-s)*abs.(rate / leftPsi) for rate in rates]

    return rates
end


# Get the state space
sh = qKCMS(omega, gamma, kappa)
sh2 = qKCMSDM(omega, gamma, kappa)

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
k1 = gamma*(plight*abs(light[2])^2 + pdark*abs(dark[2])^2)
k1 += kappa*(plight*abs(light[1])^2 + pdark*abs(dark[1])^2)
kstat = (k1 + (N-1)*plight*k1) / N

# Load in rho, find activity and set the transition rates
rhoDirect = string(directHome, "DMRG/omega=", omega, " gamma=", gamma, "/N = ",
                   N, "/s = ", s, ".h5")
f = h5open(rhoDirect, "r")
leftrho = read(f, "psileft", MPS)
rightrho = read(f, "psiright", MPS)
close(f)
activity = inner(leftrho, activityop(N, gamma, kappa, s), rightrho)
activity = -real(activity / inner(leftrho, rightrho))
flat = productMPS(sh2, ["s" for i = 1:N])
leftrho *= 1/(inner(leftrho, flat))
movecenter!(leftrho, N)
movecenter!(leftrho, 1)
rho = vectodm(leftrho)
qjmc_update_rates(psi::MPS, jumpops::OpList, st::Sitetypes) = qjmc_update_rates(psi::MPS, rho, s, jumpops::OpList, st::Sitetypes)
qjmc_emission_rates(psi::MPS, jumpops::OpList, st::Sitetypes) = qjmc_emission_rates(psi::MPS, rho, s, jumpops::OpList, st::Sitetypes)

# Calculate times
dt = 0.01/omega
dt = min(0.01/activity, dt)
save = 1000*dt
tmax = 100000 / activity

# Create initial state
if activity / N < real(kstat / 2)
    psi = productMPS(sh, ["da" for i = 1:N])
else
    states = [rand() < plight ? "l" : "da" for i = 1:N]
    psi = productMPS(sh, states)
end

# Create the Hamiltonian
H = OpList(N)
add!(H, "x", 1, omega)
for i = 1:N-1
    add!(H, ["pl", "x"], [i, i+1], omega)
end

# Create jump operators
jumpops = OpList(N)
add!(jumpops, "s-", 1, sqrt(kappa))
add!(jumpops, "s+", 1, sqrt(gamma))
for i = 1:N-1
    add!(jumpops, ["pl", "s-"], [i, i+1], sqrt(kappa))
    add!(jumpops, ["pl", "s+"], [i, i+1], sqrt(gamma))
end

# Observers
actDirect = string(directHome, "QJMC/activity/omega=", omega, " gamma=", gamma, "/N = ",
                   N, "/")
if !isdir(actDirect)
    mkpath(actDirect)
end
actDirect = string(actDirect, "s = ", s, ".h5")
actobs = QJMCActivity(true, actDirect)

# QJMC
jumps, times = qjmc_simulation(psi, H, jumpops, sh, tmax, dt, [actobs]; save=save, update=true)
