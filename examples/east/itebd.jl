#include("src/TensorNetworks.jl")
using .TensorNetworks

# Parameters
c = 0.5
s = 0.0
dt = 0.1
maxdim = 8
cutoff = 1e-12
d = 2

# Create east gates
sh = spinhalf()
gates = OpList(2)
add!(gates, ["pu", "s+"], [1, 2], exp(-s)*c)
add!(gates, ["pu", "s-"], [1, 2], exp(-s)*(1-c))
add!(gates, ["pu", "pd"], [1, 2], -c)
add!(gates, ["pu", "pu"], [1, 2], -(1-c))

# Calculate exponential
gate = sitetensor(gates, sh, 1)
gate = exp(dt*gate, [2, 4])

# Create initial MPO
A = zeros(ComplexF64, 1, 2, 2, 1)
A[1, 1, 1, 1] = 1
A[1, 2, 2, 1] = 1
B = zeros(ComplexF64, 1, 2, 2, 1)
B[1, 1, 1, 1] = 1
B[1, 2, 2, 1] = 1
S1 = ones(ComplexF64, 1, 1)
S2 = ones(ComplexF64, 1, 1)

normalS1 = 0
normalS2 = 0
for i = 1:10000
    #println("---")
    #println(i)
    ### First application
    # Contract in singular values
    A = contract(S1, A, 2, 1)
    B = contract(S2, B, 2, 1)
    B = contract(B, S1, 4, 1)

    # Contract sites and apply gates
    prod1 = contract(A, B, 4, 1)
    prod1 = contract(gate, prod1, [2, 4], [2, 4])
    prod1 = moveidx(prod1, 1, 3)
    prod1 = moveidx(prod1, 1, 4)

    # Do svd
    prod1, cmb = combineidxs(prod1, [1, 2, 3])
    prod1, cmb = combineidxs(prod1, [1, 2, 3])
    A, S2, B = svd(prod1, 2; maxdim=maxdim, cutoff=cutoff)

    # Restore indices
    A = moveidx(A, 1, 2)
    A = reshape(A, size(S2)[1], size(S1)[2], d, d)
    A = moveidx(A, 1, 4)
    B = reshape(B, size(S2)[2], d, d, size(S1)[1])

    # Multiply by inverse
    S = diagm(1 ./ diag(S1))
    A = contract(S, A, 2, 1)
    B = contract(B, S, 4, 1)

    # Renormalize tensors
    normalS2 += log(sqrt(sum(S2.^2)))
    S2 /= sqrt(sum(S2.^2))


    ### Second application
    # Contract in singular values
    B = contract(S2, B, 2, 1)
    A = contract(S1, A, 2, 1)
    A = contract(A, S2, 4, 1)

    # Contract sites and apply gates
    prod1 = contract(B, A, 4, 1)
    prod1 = contract(gate, prod1, [2, 4], [2, 4])
    prod1 = moveidx(prod1, 1, 3)
    prod1 = moveidx(prod1, 1, 4)

    # Do svd
    prod1, cmb = combineidxs(prod1, [1, 2, 3])
    prod1, cmb = combineidxs(prod1, [1, 2, 3])
    B, S1, A = svd(prod1, 2; maxdim=maxdim, cutoff=cutoff)

    # Restore indices
    B = moveidx(B, 1, 2)
    B = reshape(B, size(S1)[1], size(S2)[2], d, d)
    B = moveidx(B, 1, 4)
    A = reshape(A, size(S1)[2], d, d, size(S2)[1])

    # Multiply by inverse
    S = diagm(1 ./ diag(S2))
    B = contract(S, B, 2, 1)
    A = contract(A, S, 4, 1)

    # Renormalize tensors
    normalS1 += log(sqrt(sum(S1.^2)))
    S1 /= sqrt(sum(S1.^2))

    #println(S1)
    #println(S2)
end

# Contract a cell
M1 = contract(trace(A, 2, 3), S2, 2, 1)
M1 = contract(M1, trace(B, 2, 3), 2, 1)
M1 = contract(M1, S1, 2, 1)
M1 *= exp(normalS1 + normalS2)
M2 = contract(S1, trace(A, 2, 3), 2, 1)
M2 = contract(M2, S2, 2, 1)
M2 = contract(M2, trace(B, 2, 3), 2, 1)
M2 *= exp(normalS1 + normalS2)

# Create boundaries
B1 = zeros(ComplexF64, 2, 2, size(M1)[1])
B1[1, 1, :] = ones(ComplexF64, size(M1)[1])
B1 = trace(B1, 1, 2)
B2 = zeros(ComplexF64, 2, 2, size(M1)[1])
B2[2, 2, :] = ones(ComplexF64, size(M1)[1])
B2 = trace(B2, 1, 2)

B1norm = 0
B2norm = 0
normals = []
for i = 1:10000
    B1 = contract(B1, M1, 1, 1)
    B2 = contract(M2, B2, 2, 1)
    B1norm += log(norm(B1))
    B2norm += log(norm(B2))
    B1 /= norm(B1)
    B2 /= norm(B2)

    prod1 = contract(B1, trace(A, 2, 3), 1, 1)
    prod1 = contract(prod1, S2, 1, 1)
    prod1 = contract(prod1, trace(B, 2, 3), 1, 1)
    prod1 = contract(prod1, B2, 1, 1)

    normal = B1norm + B2norm + normalS2+ log(prod1[1])
    push!(normals, normal / (2*i+2))
end

plot(1:10000, real(normals), xaxis=:log)
