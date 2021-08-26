"""
    MPS(dim::Int, tensors::Vector{Array{Complex{Float64}, 3}}, center::Int)

Create a MPS with physical dimension dim.
"""
mutable struct MPS <: AbstractMPS
    dim::Int
    tensors::Vector{Array{ComplexF64, 3}}
    center::Int
end


### Deal with gauge moving

"""
    moveleft!(psi::MPS, idx::Int; kwargs...)

Move the gauge from a tensor within the MPS to the left.
"""
function moveleft!(psi::MPS, idx::Int; kwargs...)
    if 1 < idx && idx <= length(psi)
        U, S, V = svd(psi[idx], 1; kwargs...)
        V = contract(S, V, 2, 1)
        psi[idx] = U
        psi[idx-1] = contract(psi[idx-1], V, 3, 2)
    end
end

"""
    moveright!(psi::MPS, idx::Int; kwargs...)

Move the gauge from a tensor within the MPS to the right.
"""
function moveright!(psi::MPS, idx; kwargs...)
    if 0 < idx && idx < length(psi)
        U, S, V = svd(psi[idx], 3; kwargs...)
        V = contract(S, V, 2, 1)
        psi[idx] = U
        psi[idx+1] = contract(V, psi[idx+1], 2, 1)
    end
end


"""
    norm(psi::MPS)

Calculate the norm of an MPS.
"""
function norm(psi::MPS)
    if center(psi) == 0
        movecenter!(psi, 1)
    end
    A = psi[center(psi)]
    prod = contract(conj(A), A, 1, 1)
    prod = trace(prod, 2, 4)
    prod = trace(prod, 1, 2)
    return prod[1]^0.5
end


"""
    normalize!(psi::MPS)

Normalize an MPS.
"""
function normalize!(psi::MPS)
    if center(psi) == 0
        movecenter!(psi, 1)
    end
    psi[center(psi)] *= norm(psi)^-1
end

### Products
# Inner product of two MPS
"""
    dot(psi::MPS, phi::MPS)
    inner(psi::MPS, phi::MPS)

Determine the inner product of a bra psi and ket phi.
"""
function dot(psi::MPS, phi::MPS)
    prod = ones((1, 1))
    for i = 1:length(psi)
        A1 = conj(psi[i])
        A2 = phi[i]
        prod = contract(prod, A1, 1, 1)
        prod = contract(prod, A2, 1, 1)
        prod = trace(prod, 1, 3)
    end

    return prod[1, 1]
end
inner(psi::MPS, phi::MPS) = dot(psi, phi)
*(psi::MPS, phi::MPS) = dot(psi, phi)


### Replace the sites within a MPS with a contraction of multiple sites
"""
    replacesites!(psi::MPS, A, site::Int, direction::Bool = false; kwargs...)

Replace the sites from a site onwards, with a contraction of tensors. Specify
a direction to move the gauge.
"""
function replacesites!(psi::MPS, A, site::Int, direction::Bool = false; kwargs...)
    # Determine the number of sites
    nsites = length(size(A)) - 2

    # Deal with case of just one site
    if nsites == 1
        psi[site] = A
        if 0 < (site + 1 - 2*direction) && (site + 1 - 2*direction) < length(psi)
            movecenter!(psi, site + 1 - 2*direction)
        end
        return nothing
    end

    # Repeatidly apply SVD to split the tensors
    U = A
    for i = 1:nsites-1
        if direction
            ### Sweeping Left
            # Group together the last indices
            U, cmb = combineidxs(U, [length(size(U))-1, length(size(U))])

            # Find the next site to update
            site1 = site+nsites-i

            # Apply SVD and determine the tensors
            U, S, V = svd(U, -1; kwargs...)
            U = contract(U, S, length(size(U)), 1)
            D = site1 == length(psi) ? 1 : size(psi[site1+1])[1]
            V = reshape(V, (size(S)[2], dim(psi), D))

            # Update tensors
            psi[site1] = V
        else
            ### Sweeping right
            # Combine first two indexs together
            U, cmb = combineidxs(U, [1, 2])

            # Find the next site to update
            site1 = site + i - 1

            # Apply SVD and determine the tensors
            U, S, V = svd(U, -1; kwargs...)
            U = contract(U, S, length(size(U)), 1)
            U = moveidx(U, length(size(U)), 1)
            D = site1 == 1 ? 1 : size(psi[site1-1])[3]
            V = reshape(V, (size(S)[2], D, dim(psi)))
            V = moveidx(V, 1, -1)

            # Update tensors
            psi[site1] = V
        end
    end

    # Update the final site
    site1 = direction ? site : site + nsites - 1
    psi[site1] = U
    psi.center = site1
    return nothing
end


### Create MPS
"""
    randomMPS(dim::Int, length::Int, bonddim::Int)

Create a MPS with random entries.
"""
function randomMPS(dim::Int, length::Int, bonddim::Int)
    tensors = []
    for i = 1:length
        D1 = i == 1 ? 1 : bonddim
        D2 = i == length ? 1 : bonddim
        push!(tensors, randn(Float64, (D1, dim, D2)))
    end
    psi =  MPS(dim, tensors, 0)
    movecenter!(psi, 1)
    psi[1] = randn(Float64, (1, dim, min(2, bonddim)))
    normalize!(psi)
    return psi
end


"""
    productMPS(sites::Int, A::Array{Complex{Float64}, 3})
    productMPS(sites::Int, A)

Create a product MPS of some fixed tensor.
A can be a vector for product state entries, or larger dimensional tensor which
is truncated at the edge sites.
"""
function productMPS(sites::Int, A::Array{Complex{Float64}, 3})
    tensors = Array{Complex{Float64}, 3}[]
    push!(tensors, A[end:end, :, :])
    for i = 2:sites-1
        push!(tensors, A)
    end
    push!(tensors, A[:, :, 1:1])
    return MPS(size(A)[2], tensors, 0)
end

function productMPS(sites::Int, A)
    if length(size(A)) == 1
        A = reshape(A, (1, size(A)[1], 1))
    end
    A = convert(Array{Complex{Float64}, 3}, A)
    return productMPS(sites, A)
end

"""
    productMPS(st::Sitetypes, names::Vector{String})

Create a product state from the names of local states on a sitetype.
"""
function productMPS(st::Sitetypes, names::Vector{String})
    tensors = []
    for i = 1:length(names)
        A = convert(Array{Complex{Float64}, 3},
                    reshape(state(st, names[i]), (1, st.dim, 1)))
        push!(tensors, A)
    end
    return MPS(st.dim, tensors, 0)
end


### Save and write
function HDF5.write(parent::Union{HDF5.File, HDF5.Group}, name::AbstractString,
                    M::MPS)
    g = create_group(parent, name)
    attributes(g)["type"] = "MPS"
    attributes(g)["version"] = 1
    write(g, "length", length(M))
    write(g, "center", center(M))
    for i = 1:length(M)
        write(g, "MPS[$(i)]", M[i])
    end
end

function HDF5.read(parent::Union{HDF5.File, HDF5.Group}, name::AbstractString,
                    ::Type{MPS})
    g = open_group(parent, name)
    if read(attributes(g)["type"]) != "MPS"
        error("HDF5 group of file does not contain MPS data.")
    end
    N = read(g, "length")
    center = read(g, "center")
    tensors = [read(g, "MPS[$(i)]") for i=1:N]
    return MPS(size(tensors[1])[2], tensors, center)
end
