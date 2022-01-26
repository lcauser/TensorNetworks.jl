"""
    MPO(dim::Int, tensors::Vector{Array{Complex{Float64}, 4}}, center::Int)

Create a MPO with physical dimension dim.
"""
mutable struct MPO <: AbstractMPS
    dim::Int
    tensors::Vector{Array{ComplexF64, 4}}
    center::Int
end

function MPO(dim::Int, length::Int)
    tensors = []
    for i = 1:length
        push!(tensors, zeros(ComplexF64, (1, dim, dim, 1)))
    end
    return MPO(dim, tensors, 0)
end
### Deal with gauge moving

"""
    moveleft!(O::MPO, idx::Int)

Move the gauge from a tensor within the MPO to the left.
"""
function moveleft!(O::MPO, idx::Int; kwargs...)
    if 1 < idx && idx <= length(O)
        U, S, V = svd(O[idx], 1; kwargs...)
        V = contract(S, V, 2, 1)
        O[idx] = U
        O[idx-1] = contract(O[idx-1], V, 4, 2)
    end
end

"""
    moveright!(psi::MPO, idx::Int)

Move the gauge from a tensor within the MPO to the right.
"""
function moveright!(O::MPO, idx; kwargs...)
    if 0 < idx && idx < length(O)
        U, S, V = svd(O[idx], 4; kwargs...)
        V = contract(S, V, 2, 1)
        O[idx] = U
        O[idx+1] = contract(V, O[idx+1], 2, 1)
    end
end


"""
    norm(O::MPO)

Calculate the norm of an MPO.
"""
function norm(O::MPO)
    if center(O) == 0
        movecenter!(O, 1)
    end
    A = O[center(O)]
    prod = contract(conj(A), A, 1, 1)
    prod = trace(prod, 3, 6)
    prod = trace(prod, 1, 3)
    prod = trace(prod, 1, 2)
    return prod[1]^0.5
end

function bonddim(psi::MPO, site::Int)
    (site < 1 || site >= length(psi)) && return nothing
    return size(psi[site])[4]
end

### Create MPOs
"""
    productMPO(sites::Int, A::Array{Complex{Float64}, 4})
    productMPO(sites::Int, A)

Create a product MPO of some fixed tensor.
A can be a vector for product state entries, or larger dimensional tensor which
is truncated at the edge sites.
"""
function productMPO(sites::Int, A::Array{Complex{Float64}, 4})
    tensors = Array{Complex{Float64}, 4}[]
    push!(tensors, A[end:end, :, :, :])
    for i = 2:sites-1
        push!(tensors, A)
    end
    push!(tensors, A[:, :, :, 1:1])
    return MPO(size(A)[2], tensors, 0)
end

function productMPO(sites::Int, A)
    if length(size(A)) == 1
        A = reshape(A, (1, size(A)[1], size(A)[1], 1))
    end
    A = convert(Array{Complex{Float64}, 4}, A)
    return productMPO(sites, A)
end


"""
    productMO(st::Sitetypes, names::Vector{String})

Create a product operator from the names of local operators on a sitetype.
"""
function productMPO(st::Sitetypes, names::Vector{String})
    tensors = []
    for i = 1:length(names)
        A = convert(Array{Complex{Float64}, 4},
                    reshape(op(st, names[i]), (1, st.dim, st.dim, 1)))
        push!(tensors, A)
    end
    return MPO(st.dim, tensors, 0)
end


"""
    randomMPO(dim::Int, length::Int, bonddim::Int)

Create a MPO with random entries.
"""
function randomMPO(dim::Int, length::Int, bonddim::Int)
    tensors = []
    for i = 1:length
        D1 = i == 1 ? 1 : bonddim
        D2 = i == length ? 1 : bonddim
        push!(tensors, randn(Float64, (D1, dim, dim, D2)))
    end
    psi = MPO(dim, tensors, 0)
    movecenter!(psi, 1)
    psi[1] = randn(Float64, (1, dim, dim, min(dim^2, bonddim)))
    normalize!(psi)
    return psi
end

### Products
"""
    applyMPO(O::MPO, psi::MPS, hermitian=false; kwargs...)

Apply an MPO to an MPS. Specify whether to apply the hermitian conjugate.
Define truncation parameters using key arguments.
"""
function applyMPO(O::MPO, psi::MPS, hermitian=false; kwargs...)
    phi = MPS(dim(psi), length(psi))
    # Loop through applying the MPO, and move the gauge across
    for i = 1:length(psi)
        A = psi[i]
        M = hermitian ? conj(O[i]) : O[i]
        idx = hermitian ? 2 : 3
        B = contract(M, A, idx, 2)
        B, cmb1 = combineidxs(B, [1, 4])
        B, cmb2 = combineidxs(B, [2, 3])
        B = moveidx(B, 1, 2)
        phi[i] = B
        if i > 1
            moveright!(phi, i-1)
        end
    end

    # Orthogonalize to first site with truncation
    movecenter!(phi, 1; kwargs...)

    return phi
end
*(O::MPO, psi::MPS) = applyMPO(O, psi)
*(psi::MPS, O::MPO) = applyMPO(O, psi, true)


function applyMPO(O1::MPO, O2::MPO; kwargs...)
    O = MPO(dim(O1), length(O1))
    # Loop through applying the MPO, and move the gauge across
    for i = 1:length(O1)
        M1 = O1[i]
        M2 = O2[i]
        M = contract(M1, M2, 3, 2)
        M, cmb1 = combineidxs(M, [1, 4])
        M, cmb2 = combineidxs(M, [2, 4])
        M = moveidx(M, 3, 1)
        O[i] = M
        if i > 1
            moveright!(O, i-1)
        end
    end

    # Orthogonalize to first site with truncation
    movecenter!(O, 1; kwargs...)

    return O
end


"""
    inner(psi::MPS, O::MPO, phi::MPO)

Calculate the inner product of some operator with respect to a bra and ket.
"""
function inner(psi::MPS, O::MPO, phi::MPS)
    # Loop through each site contracting
    prod = ones((1, 1, 1))
    for i = 1:length(psi)
        prod = contract(prod, conj(psi[i]), 1, 1)
        prod = contract(prod, O[i], [1, 3], [1, 2])
        prod = contract(prod, phi[i], [1, 3], [1, 2])
    end

    return prod[1, 1, 1]
end

"""
    inner(O1::MPO, psi::MPS, O2::MPO, phi::MPO)

Apply O1 to psi (bra), O2 to phi (ket) and then take the inner product.
"""
function inner(O1::MPO, psi::MPS, O2::MPO, phi::MPS)
    # Loop through each site contracting
    prod = ones((1, 1, 1, 1))
    for i = 1:length(psi)
        prod = contract(prod, conj(psi[i]), 1, 1)
        prod = contract(prod, conj(O1[i]), 1, 1)
        prod = trace(prod, 3, 6)
        prod = contract(prod, O2[i], 1, 1)
        prod = trace(prod, 3, 5)
        prod = contract(prod, phi[i], 1, 1)
        prod = trace(prod, 3, 5)
    end

    return prod[1, 1, 1, 1]
end


"""
    trace(O::MPO)

Determine the trace of an MPO.
"""
function trace(O::MPO)
    prod = ones(1)
    for i = 1:length(O)
        prod = contract(prod, O[i], 1, 1)
        prod = trace(prod, 1, 2)
    end
    return prod[1]
end

"""
    trace(O1::MPO, O2::MPO)

Determine the trace of MPO products.
"""
function trace(O1::MPO, O2::MPO)
    prod = ones((1, 1))
    for i = 1:length(O1)
        prod = contract(prod, O1[i], 1, 1)
        prod = contract(prod, O2[i], [1, 3, 2], [1, 2, 3])
    end
    return prod[1, 1]
end

function trace(O1::MPO, O2::MPO, O3::MPO)
    prod = ones((1, 1, 1))
    for i = 1:length(O1)
        prod = contract(prod, O1[i], 1, 1)
        prod = contract(prod, O2[i], [1, 4], [1, 2])
        prod = contract(prod, O3[i], [1, 4, 2], [1, 2, 3])
    end
    return prod[1, 1, 1]
end

### Boundary MPOs
"""
    bMPO(length::Int)

Creates a boundary MPO.
"""
function bMPO(length::Int)
    M = MPO(1, length)
    for i = 1:length
        M[i] = ones(ComplexF64, 1, 1, 1, 1)
    end
    return M
end

"""
    randombMPO(length::int, bonddims1::Vector{Int}, bonddims2::Vector{Int})

Create a random bMPO with choosen ``physical'' dimensions.
"""
function randombMPO(length::Int, chi::Int, bonddims1, bonddims2)
    M = MPO(1, length)
    for i = 1:length
        D1 = i == 1 ? 1 : chi
        D2 = i == length ? 1 : chi
        M[i] = abs.(randn(Float64, D1, bonddims1[i], bonddims2[i], D2))
    end
    movecenter!(M, length)
    movecenter!(M, 1; maxdim=chi, cutoff=1e-30)

    M[1] = abs.(randn(Float64, 1, bonddims1[1], bonddims2[1], 1*size(M[2])[1]))
    return M
end


function expand!(O::MPO, dim::Int, noise=1e-3)
    len = Int(iseven(length(O)) ? length(O) / 2 - 1 : (length(O) - 1) / 2)
    D1 = D2 = D3 = D4 = 1
    for i = 1:len
        dims1 = size(O[i])
        dims2 = size(O[length(O) + 1 - i])
        D1 = i == 1 ? 1 : D2
        D2 = min(D1*dims1[2]*dims1[3], dim)
        D4 = i == 1 ? 1 : D3
        D3 = min(D4*dims2[2]*dims2[3], dim)
        M1 = noise*randn(Float64, D1, dims1[2], dims1[3], D2)
        M1[1:dims1[1], :, :, 1:dims1[4]] = O[i]
        M2 = noise*randn(Float64, D3, dims2[2], dims2[3], D4)
        M2[1:dims2[1], :, :, 1:dims2[4]] = O[length(O) + 1 - i]
        O[i] = M1
        O[length(O) + 1 - i] = M2
    end

    if iseven(length(O))
        dims1 = size(O[len+1])
        dims2 = size(O[length(O) - len])
        D1 = D2
        D4 = D3
        D2 = D3 = min(D1*dims1[2]*dims1[3], D4*dims2[2]*dims2[3], dim)
        M1 = noise*randn(Float64, D1, dims1[2], dims1[3], D2)
        M1[1:dims1[1], :, :, 1:dims1[4]] = O[len+1]
        M2 = noise*randn(Float64, D3, dims2[2], dims2[3], D4)
        M2[1:dims2[1], :, :, 1:dims2[4]] = O[length(O) - len]
        O[len+1] = M1
        O[length(O) - len] = M2
    else
        D1 = D2
        D4 = D3
        dims = size(O[len+1])
        M = noise*randn(Float64, D1, dims[2], dims[3], D4)
        M1[1:dims1[1], :, :, 1:dims1[4]] = O[len+1]
        O[len+1] = M1
    end
    center = O.center
    O.center = 0
    movecenter!(O, center)
end


### Add MPOs
"""
    addMPOs(O1::MPO, O2::MPO; kwargs...)

Add two MPOs and apply SVD to truncate.
"""
function addMPOs(O1::MPO, O2::MPO; kwargs...)

end


### Automatically construct MPOs from operator lists.
"""
    MPO(H::OpList, st::Sitetypes; kwargs...)

Constructs an MPO from an operator list by sequentially adding terms and
applying SVD.
"""
function MPO(H::OpList, st::Sitetypes; kwargs...)

end


### Save and write
function HDF5.write(parent::Union{HDF5.File, HDF5.Group}, name::AbstractString,
                    M::MPO)
    g = create_group(parent, name)
    attributes(g)["type"] = "MPO"
    attributes(g)["version"] = 1
    write(g, "length", length(M))
    write(g, "center", center(M))
    for i = 1:length(M)
        write(g, "MPO[$(i)]", M[i])
    end
end

function HDF5.read(parent::Union{HDF5.File, HDF5.Group}, name::AbstractString,
                    ::Type{MPO})
    g = open_group(parent, name)
    if read(attributes(g)["type"]) != "MPO"
        error("HDF5 group of file does not contain MPO data.")
    end
    N = read(g, "length")
    center = read(g, "center")
    tensors = [read(g, "MPO[$(i)]") for i=1:N]
    return MPO(size(tensors[1])[2], tensors, center)
end
