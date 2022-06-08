include("src/TensorNetworks.jl")
using HDF5

N = 6
c = 0.5
s = -0.01
chi = 4
home = "D:/ORFA Data/2d/PEPS/"

sh = spinhalf()
A = -sqrt(c * (1-c)) * exp(-s) * op(sh, "x") + c * op(sh, "pd") + (1-c)*op(sh, "pu")
M = zeros(ComplexF64, 5, 5, 5, 5, 2, 2)
M[1, 1, 1, 1, :, :] = op(sh, "id")
M[1, 1, 4, 2, :, :] = op(sh, "pu")
M[2, 1, 4, 3, :, :] = A
M[1, 1, 4, 3, :, :] = A
M[3, 1, 4, 3, :, :] = op(sh, "pu")
M[1, 1, 2, 4, :, :] = op(sh, "pu")
M[1, 2, 4, 4, :, :] = A
M[1, 1, 3, 4, :, :] = A
M[1, 3, 4, 4, :, :] = op(sh, "pu")
M[5, 5, 5, 5, :, :] = op(sh, "id")
M[1, 4, 4, 5, :, :] = op(sh, "id")
M[4, 1, 5, 4, :, :] = op(sh, "id")
M[5, 4, 4, 5, :, :] = op(sh, "id")
M[4, 5, 5, 4, :, :] = op(sh, "id")

H = productPEPO(N, M)
H[N, 1] = M[1:1, :, 1:1, :, :, :] + M[1:1, :, 4:4, :, :, :] + M[1:1, :, 5:5, :, :, :]
H[1, N] = M[:, 1:1, :, 1:1, :, :] + M[:, 1:1, :, 4:4, :, :] + M[:, 1:1, :, 5:5, :, :]
for j = 2:N-1
    H[N, j] = M[:, :, 1:1, :, :, :] + M[:, :, 4:4, :, :, :] + M[:, :, 5:5, :, :, :]
    H[j, N] = M[:, :, :, 1:1, :, :] + M[:, :, :, 4:4, :, :] + M[:, :, :, 5:5, :, :]
end
H[N, N] = M[:, :, 4:4, 4:4, :, :] + M[:, :, 4:4, 5:5, :, :] + M[:, :, 5:5, 4:4, :, :] + M[:, :, 5:5, 5:5, :, :]

for i = 1:N
    for j = 1:N
        A = H[i, j]
        if j != 1
            A, S, V = svd(A, 1; cutoff=1e-12)
            V = contract(H[i, j-1], V, 4, 2)
            V = moveidx(V, 6, 4)
            A = contract(A, S, 1, 1)
            A = moveidx(A, 6, 1)
            H[i, j-1] = V
        end

        if j != N
            A, S, V = svd(A, 4; cutoff=1e-12)
            V = contract(H[i, j+1], V, 1, 2)
            V = moveidx(V, 6, 1)
            A = contract(A, S, 4, 1)
            A = moveidx(A, 6, 4)
            H[i, j+1] = V
        end

        if i != 1
            A, S, V = svd(A, 2; cutoff=1e-12)
            V = contract(H[i-1, j], V, 3, 2)
            V = moveidx(V, 6, 3)
            A = contract(A, S, 2, 1)
            A = moveidx(A, 6, 2)
            H[i-1, j] = V
        end

        if i != N
            A, S, V = svd(A, 3; cutoff=1e-12)
            V = contract(H[i+1, j], V, 2, 2)
            V = moveidx(V, 6, 2)
            A = contract(A, S, 3, 1)
            A = moveidx(A, 6, 3)
            H[i+1, j] = V
        end
        H[i, j] = A
    end
end


ops = OpList2d(N)
# Add kinetic terms
global kin = 0
for i = 1:N
    for j = 1:N
        p = (i == 1 && j == 1) ? "id" : "n"
        if i <= N-1
            if i == 1 && j == 1
                add!(ops, ["x"], [2, 1], true, sqrt(c*(1-c))*exp(-s))
                global kin = kin + 1
            else
                add!(ops, [p, "x"], [i, j], true, sqrt(c*(1-c))*exp(-s))
                add!(ops, ["x", p], [i, j], true, sqrt(c*(1-c))*exp(-s))
                global kin = kin + 2
            end
        end
        if j <= N-1
            if i == 1 && j == 1
                add!(ops, ["x"], [1, 2], false, sqrt(c*(1-c))*exp(-s))
                global kin = kin + 1
            else
                add!(ops, [p, "x"], [i, j], false, sqrt(c*(1-c))*exp(-s))
                add!(ops, ["x", p], [i, j], false, sqrt(c*(1-c))*exp(-s))
                global kin = kin + 2
            end
        end
    end
end

# Add escape rate terms
global es = 0
for i = 1:N
    for j = 1:N
        p = (i == 1 && j == 1) ? "id" : "n"
        if i <= N-1
            if i == 1 && j == 1
                add!(ops, [p, "pu"], [i, j], true, -(1-c))
                add!(ops, [p, "pd"], [i, j], true, -c)
                global es = es + 2
            else
                add!(ops, [p, "pu"], [i, j], true, -(1-c))
                add!(ops, [p, "pd"], [i, j], true, -c)
                add!(ops, ["pu", p], [i, j], true, -(1-c))
                add!(ops, ["pd", p], [i, j], true, -c)
                global es = es + 4
            end

        end
        if j <= N-1
            if i == 1 && j == 1
                add!(ops, [p, "pu"], [i, j], false, -(1-c))
                add!(ops, [p, "pd"], [i, j], false, -c)
                global es = es + 2
            else
                add!(ops, [p, "pu"], [i, j], false, -(1-c))
                add!(ops, [p, "pd"], [i, j], false, -c)
                add!(ops, ["pu", p], [i, j], false, -(1-c))
                add!(ops, ["pd", p], [i, j], false, -c)
                global es = es + 4
            end
        end
    end
end


up = zeros(ComplexF64, 1, 1, 1, 1, 2)
up[1, 1, 1, 1, 1] = 1
dn = zeros(ComplexF64, 1, 1, 1, 1, 2)
dn[1, 1, 1, 1, 2] = 1
eq = zeros(ComplexF64, 1, 1, 1, 1, 2)
eq[1, 1, 1, 1, 1] = sqrt(c)
eq[1, 1, 1, 1, 2] = sqrt(1-c)
psi = productPEPS(N, eq)

energySU = []
energyVU = []

D = 1
chi = 4*D^2
psi, energy = simpleupdate(psi, 0.1, sh, ops; maxiter=100000, miniter=2000, maxdim=D, chi=50, cutoff=1e-8)
ops.coeffs .*= - 1
psi, E = vpeps(sh, psi, H, ops; chi=chi)

D = 2
chi = 4*D^2
ops.coeffs .*= - 1
psi, energy = simpleupdate(psi, 0.1, sh, ops; maxiter=100000, miniter=2000, maxdim=D, chi=50, cutoff=1e-8)
ops.coeffs .*= - 1
psi, E = vpeps(sh, psi, H, ops; chi=chi)
