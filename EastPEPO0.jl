include("src/TensorNetworks.jl")
using HDF5

N = 10
c = 0.5
s = -1.0
home = "D:/East Data/2d/PEPS/"

sh = spinhalf()
A = -sqrt(c * (1-c)) * exp(-s) * op(sh, "x") + c * op(sh, "pd") + (1-c)*op(sh, "pu")
M = zeros(ComplexF64, 5, 5, 5, 5, 2, 2)
M[1, 1, 1, 1, :, :] = op(sh, "id")
M[1, 1, 3, 2, :, :] = op(sh, "pu")
M[2, 1, 3, 4, :, :] = A
M[1, 1, 2, 4, :, :] = op(sh, "pu")
M[1, 2, 3, 4, :, :] = A
M[5, 5, 5, 5, :, :] = op(sh, "id")
M[1, 3, 3, 5, :, :] = op(sh, "id")
M[4, 1, 5, 4, :, :] = op(sh, "id")
M[5, 3, 3, 5, :, :] = op(sh, "id")
M[4, 5, 5, 4, :, :] = op(sh, "id")

H = productPEPO(N, M)
H[N, 1] = M[1:1, :, 1:1, :, :, :] + M[1:1, :, 3:3, :, :, :] + M[1:1, :, 5:5, :, :, :]
H[1, N] = M[:, 1:1, :, 1:1, :, :] + M[:, 1:1, :, 4:4, :, :] + M[:, 1:1, :, 5:5, :, :]
for j = 2:N-1
    H[N, j] = M[:, :, 1:1, :, :, :] + M[:, :, 3:3, :, :, :] + M[:, :, 5:5, :, :, :]
    H[j, N] = M[:, :, :, 1:1, :, :] + M[:, :, :, 4:4, :, :] + M[:, :, :, 5:5, :, :]
end
H[N, N] = M[:, :, 3:3, 4:4, :, :] + M[:, :, 3:3, 5:5, :, :] + M[:, :, 5:5, 4:4, :, :] + M[:, :, 5:5, 5:5, :, :]


direct = string(home, "c = ", c, "/N = ", N, "/")
direct = string(direct, "s = ", s, ".h5")

# Load in the properties
f = h5open(direct)
global psi = read(f, "psi", GPEPS)
global energy = read(f, "scgf")
completed = read(f, "completed")
global iter = read(f, "iter")
close(f)


env = Environment(psi, H, psi; chi=32)
E = inner(env) / inner(psi, psi; chi=32)
