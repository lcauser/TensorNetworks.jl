"""
    MPO(dim::Int, tensors::Vector{Array{Complex{Float64}, 4}}, center::Int)

Create a MPO with physical dimension dim.
"""
mutable struct MPO <: AbstractMPS
    dim::Int
    tensors::Vector{Array{Complex{Float64}, 4}}
    center::Int
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
function moveright!(O::MPS, idx; kwargs...)
    if 0 < idx && idx < length(psi)
        U, S, V = svd(O[idx], 4; kwargs...)
        V = contract(S, V, 2, 1)
        O[idx] = U
        O[idx+1] = contract(V, O[idx+1], 2, 1)
    end
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
    return MPS(size(A)[2], tensors, 0)
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

### Products
"""
    applyMPO(O::MPO, psi::MPS, hermitian=false; kwargs...)

Apply an MPO to an MPS. Specify whether to apply the hermitian conjugate.
Define truncation parameters using key arguments.
"""
function applyMPO(O::MPO, psi::MPS, hermitian=false; kwargs...)
    phi = MPS(dim(psi), length(psi), length(psi))
    # Loop through applying the MPO, and move the gauge across
    for i = 1:length(psi)
        A = psi[i]
        M = hermitian ? conj(O[i]) : O[i]
        idx = hermitian ? 2 : 3
        B = contract(O, A, idx, 2)
        B = moveidx(B, 4, 3)
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
*(psi::MPS, O::MPO) = applyMPO(O, psi)

"""
    inner(psi::MPS, O::MPO, phi::MPO)

Calculate the inner product of some operator with respect to a bra and ket.
"""
function inner(psi::MPS, O::MPO, phi::MPO)
    # Loop through each site contracting
    prod = ones((1, 1, 1))
    for i = 1:length(psi)
        prod = contract(prod, conj(psi[i]), 1, 1)
        prod = contract(prod, O[i], 1, 1)
        prod = trace(prod, 2, 4)
        prod = contract(phi[i], 1, 1)
        prod = trace(prod, 2, 4)
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
        prod = contract(phi[i], 1, 1)
        prod = trace(prod, 3, 5)
    end

    return prod[1, 1, 1, 1]
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
