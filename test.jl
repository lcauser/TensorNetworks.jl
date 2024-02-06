include("src/TensorNetworks.jl")
using .TensorNetworks
using LinearAlgebra
using KrylovKit
using MatrixFactorizations


function Heff(Ac, Al, Ar, HL, HR, H, E)
    # Calculate the first term
    term1 = diagm(ones(ComplexF64, maxbonddim(psi)))
    term1 = tensorproduct(term1, ones(ComplexF64, 1))
    term1 = moveidx(term1, 3, 2)
    for i in 1:length(H)
        term1 = contract(term1, i == length(H) ? conj(Ac) : conj(Al), 1, 1)
        term1 = contract(term1, H[i], [1, 3], [1, 2])
        if i != length(H)
            term1 = contract(term1, Al, [1, 3], [1, 2])
        end
    end
    term1 = moveidx(term1[:, :, :, 1], 3, 2) .- (E .* conj(Ac))

    # Calculate the second term 
    term2 = diagm(ones(ComplexF64, maxbonddim(psi)))
    term2 = tensorproduct(term2, ones(ComplexF64, 1))
    term2 = moveidx(term2, 3, 2)
    for i in length(H):-1:1
        term2 = contract(term2, i == 1 ? conj(Ac) : conj(Ar), 1, 3)
        term2 = contract(term2, H[i], [1, 4], [4, 2])
        if i != 1
            term2 = contract(term2, Ar, [1, 4], [3, 2])
        end
    end
    term2 = moveidx(term2[:, :, 1, :], 1, 3) .- (E .* conj(Ac))

    # Calculate the third term 
    term3 = contract(HL, conj(Ac), 1, 1)

    # Calculate the fourth term 
    term4 = contract(conj(Ac), HR, 3, 1)

    return conj(term1 .+ term2 .+ term3 .+ term4)
end

function Ceff(C, Al, Ar, HL, HR, H, E)
    # Calculate the first term 
    #term1 = diagm(ones(ComplexF64, maxbonddim(psi)))
    #term1 = tensorproduct(term1, ones(ComplexF64, 1))
    #term1 = moveidx(term1, 3, 2)
    #for i in 1:length(H)
    #    term1 = contract(term1, conj(Al), 1, 1)
    #    term1 = contract(term1, H[i], [1, 3], [1, 2])
    #    term1 = contract(term1, Al, [1, 3], [1, 2])
    #end
    #term1 = term1[:, 1, :]
    #term1 = contract(term1, conj(C), 1, 1) .- (E .* conj(C))

    left = diagm(ones(ComplexF64, maxbonddim(psi)))
    left = tensorproduct(left, ones(ComplexF64, 1))
    left = moveidx(left, 3, 2)
    left = contract(left, conj(Al), 1, 1)
    left = contract(left, H[1], [1, 3], [1, 2])
    left = contract(left, Al, [1, 3], [1, 2])

    right = diagm(ones(ComplexF64, maxbonddim(psi)))
    right = tensorproduct(right, ones(ComplexF64, 1))
    right = moveidx(right, 3, 2)
    right = contract(Ar, right, 3, 3)
    right = contract(H[2], right, [3, 4], [2, 4])
    right = contract(conj(Ar), right, [2, 3], [2, 4])

    term1 = contract(left, conj(C), 1, 1)
    term1 = contract(term1, right, [3, 1], [1, 2])
    term1 = term1 .- (E .* conj(C))

    # Second terms 
    term2 = contract(HL, conj(C), 1, 1)
    term3 = contract(conj(C), HR, 2, 1)


    return conj(term1 .+ term2 .+ term3)
end


h = 1.0
J = 5.0
D = 4

psi = randomUMPS(2, D)

sh = spinhalf()
H = InfiniteOpList()
add!(H, "x", 1, -h)
add!(H, ["z", "z"], [1, 2], -J)
H = MPO(sh, H)

E = inner(psi, H)
println("Iter=0, energy=$(E)")
δ = 1
iter = 0
while δ > 1e-8
    Al = psi.Al
    Ar = psi.Ar
    C = psi.C
    Ac = contract(Al, C, 3, 1)
    L = contract(conj(C), C, 1, 1)
    R = contract(C, conj(C), 2, 2)


    HL = leftEnvironment(psi, H, E)
    HR = rightEnvironment(psi, H, E)


    f(x) = Heff(x, Al, Ar, HL, HR, H, E)
    g(x) = Ceff(x, Al, Ar, HL, HR, H, E)

    δ = norm(f(Ac) - contract(Al, g(C), 3, 1))
    println(δ)

    eigs, vecs = eigsolve(f, Ac, 1, :SR; tol = δ/10)
    #println(eigs)
    Ac = vecs[argmin(real(eigs))]

    g(x) = Ceff(x, Al, Ar, HL, HR, H, E)
    eigs2, vecs2 = eigsolve(g, C, 1, :SR; tol = δ/10)
    #println(eigs2)
    C = vecs2[argmin(real(eigs2))]
    #println(C .- psi.C)

    """
    ten = contract(Ac, conj(C), 3, 2)
    U, S, V = svd(ten, 3)
    Al = contract(U, V, 3, 1)

    ten = contract(conj(C), Ac, 1, 1)
    U, S, V = svd(ten, 1)
    Ar = contract(V, U, 1, 1)
    """

    Ucl, Pcl = polar(C; alg=:hybrid)
    Ac2, cmb = combineidxs(Ac, [1, 2])
    Ac2 = moveidx(Ac2, 2, 1)
    Ual, Pal = polar(Ac2; alg=:hybrid)
    Ual = uncombineidxs(moveidx(Ual, 1, 2), cmb)
    Al = contract(Ual, conj(Ucl), 3, 2)

    Ucr, Pcr = polar(moveidx(C, 2, 1); alg=:hybrid)
    Ucr = moveidx(Ucr, 1, 2)
    Ac2, cmb = combineidxs(Ac, [2, 3])
    Ac2 = moveidx(Ac2, 2, 1)
    Uar, Par = polar(Ac2; alg=:hybrid)
    Uar = uncombineidxs(moveidx(Uar, 1, 2), cmb)
    Ar = contract(conj(Ucr), Uar, 1, 1)


    U, C, V = svd(C, 2)
    Al = contract(conj(U), contract(Al, U, 3, 1), 1, 1)
    Ar = contract(V, contract(Ar, conj(V), 3, 2), 2, 1)

    #println(norm(Ac .- contract(Al, C, 3, 1)))
    #println(norm(Ac .- contract(C, Ar, 2, 1)))
    #println(norm(contract(Al, C, 3, 1) .- contract(C, Ar, 2, 1)))

    psi.C = C
    psi.Al = Al
    psi.Ar = Ar

    E = inner(psi, H)
    iter += 1
    println("Iter=$(iter), energy=$(E)")
    #println(delta)
end

measurements = InfiniteOpList()
add!(measurements, "x", 1)
add!(measurements, "pu", 1)
add!(measurements, ["pu", "pd", "pd"], [1, 2, 3])
measurements = inner(sh, psi, measurements)