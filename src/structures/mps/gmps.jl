"""
    GMPS(rank::Int, dim::Int, tensors::Vector{Array{Complex{Float64}}},
         center::Int)
    GMPS(rank::Int, dim::Int, length::Int)

Create a generalised matrix product state with physical dimension dim.
"""
mutable struct GMPS <: AbstractMPS
    rank::Int
    dim::Int
    tensors::Vector{Array{ComplexF64}}
    center::Int
end

function GMPS(rank::Int, dim::Int, length::Int)
    tensors = []
    for i = 1:length
        push!(tensors, zeros(ComplexF64, (1, [dim for i=1:rank]..., 1)))
    end
    return GMPS(rank, dim, tensors, 0)
end


"""
    norm(psi::GMPS)

Calculate the norm of an GMPS.
"""
function norm(psi::GMPS)
    if center(psi) == 0
        movecenter!(psi, 1)
    end
    A = psi[center(psi)]
    dims = [i for i=1:2+rank(psi)]
    prod = contract(conj(A), A, dims, dims)
    return prod[1]^0.5
end


### Normalize
"""
    normalize!(psi::GMPS)

Normalize a GMPS.
"""
function normalize!(psi::GMPS)
    if center(psi) == 0
        movecenter!(psi, 1)
    end
    psi[center(psi)] *= norm(psi)^-1
end


### Move orthogonal center
"""
    moveleft!(psi::GMPS, idx::Int; kwargs...)

Move the gauge from a tensor within the GMPS to the left.
"""
function moveleft!(psi::GMPS, idx::Int; kwargs...)
    if 1 < idx && idx <= length(psi)
        U, S, V = svd(psi[idx], 1; kwargs...)
        V = contract(S, V, 2, 1)
        psi[idx] = U
        psi[idx-1] = contract(psi[idx-1], V, 2+rank(psi), 2)
    end
end


"""
    moveright!(psi::GMPS, idx::Int; kwargs...)

Move the gauge from a tensor within the GMPS to the right.
"""
function moveright!(psi::GMPS, idx; kwargs...)
    if 0 < idx && idx < length(psi)
        U, S, V = svd(psi[idx], 2+rank(psi); kwargs...)
        V = contract(S, V, 2, 1)
        psi[idx] = U
        psi[idx+1] = contract(V, psi[idx+1], 2, 1)
    end
end


"""
    movecenter!(psi::GMPS, idx::Int; kwargs...)

Move the orthogonal center of an GMPS.
"""
function movecenter!(psi::GMPS, idx::Int; kwargs...)
    (idx < 1 || idx > length(psi)) && error("The index is out of range.")
    if center(psi) == 0
        for i = 1:idx-1
            moveright!(psi, i; kwargs...)
        end
        N = length(psi)
        for i = 1:N-idx
            moveleft!(psi, N+1-i; kwargs...)
        end
    else
        if idx > center(psi)
            for i = center(psi):idx-1
                moveright!(psi, i; kwargs...)
            end
        elseif idx < center(psi)
            for i = 1:center(psi)-idx
                moveleft!(psi, center(psi)+1-i; kwargs...)
            end
        end
    end
    psi.center = idx
end


### Adjust bond dimensions
"""
    truncate!(psi::GMPS; kwargs...)

Truncate across the GMPS.
"""
function truncate!(psi::GMPS; kwargs...)
    if psi.center != 1 && psi.center != length(psi)
        movecenter!(psi, 0)
    end
    ctr = center(psi) == 1 ? length(psi) : 1
    movecenter!(psi, ctr; kwargs...)
end

"""
    expand!(psi::GMPS, bonddim::Int, noise::Float64)

Increase the bond dim of a GMPS. Use key arguments:
    - mindim: minimum bond dimension (default = 1)
    - maxdim: maximum bond dimension (default = 0, no limit)
    - cutoff: truncation cutoff error (default = 0)
function expand!(psi::GMPS, bonddim::Int, noise::Float64 = 1e-5)
    movecenter!(psi, 1)
    for i = 1:length(psi)
        dims = size(psi[i])
        dims[1] = i == 1 ? 1 : bonddim 
        dims[2] = i == length(psi) ? 1 : bonddim
        tensor = noise * randn(ComplexF64, dims...)
        tensor[1:size(psi[i])[1], 1:size(psi[i])[2], [: for _ = 1:rank(psi)]...] = psi[i]
        psi[i] = tensor
        
        if i > 1
            movecenter!(psi, i)
        end
    end
end
"""


"""
    conj(psi::GMPS)

Return the conjugate of an GMPS.
"""
function conj(psi::GMPS)
    phi = deepcopy(psi)
    for i = 1:length(phi)
        phi[i] = conj(phi[i])
    end
    return phi
end


### Equality
function ==(psi1::GMPS, psi2::GMPS)
    converge = true
    converge = rank(psi1) != rank(psi2) ? false : converge
    converge = length(psi1) != length(psi2) ? false : converge
    converge = dim(psi1) != dim(psi2) ? false : converge
    return converge
end


### Entanglement entropy
"""
    entropy(psi::GMPS, site::Int)

Calculate the entanglement entropy between site i and i+1.
"""
function entropy(psi::GMPS, site::Int)
    movecenter!(psi, site)
    U, S, V = svd(psi[site], -1)
    S2 = diag(S).^2
    return -sum(S2.*log.(S2))
end


### Replace the sites within a MPS with a contraction of multiple sites
"""
    replacesites!(psi::GMPS, A, site::Int, direction::Bool = false, normalize::Bool = false; kwargs...)

Replace the sites from a site onwards, with a contraction of tensors. Specify
a direction to move the gauge.
"""
function replacesites!(psi::GMPS, A, site::Int, direction::Bool = false, normalize::Bool = false; kwargs...)
    # Determine the number of sites
    nsites::Int = (length(size(A)) - 2) / rank(psi)

    # Deal with case of just one site
    if nsites == 1
        psi[site] = A
        if 0 < (site + 1 - 2*direction) && (site + 1 - 2*direction) <= length(psi)
            movecenter!(psi, site + 1 - 2*direction)
        end
        if normalize
            normalize!(psi)
        end
        return nothing
    end

    # Repeatidly apply SVD to split the tensors
    U = A
    for i = 1:nsites-1
        if direction
            ### Sweeping Left
            # Group together the last indices
            idxs = collect(length(size(U))-rank(psi):length(size(U)))
            U, cmb = combineidxs(U, idxs)

            # Find the next site to update
            site1 = site+nsites-i

            # Apply SVD and determine the tensors
            U, S, V = svd(U, -1; kwargs...)
            U = contract(U, S, length(size(U)), 1)
            D = site1 == length(psi) ? 1 : size(psi[site1+1])[1]
            dims = (size(S)[2], [dim(psi) for i=1:rank(psi)]..., D)
            V = reshape(V, dims)

            # Update tensors
            psi[site1] = V
        else
            ### Sweeping right
            # Combine first two indexs together
            idxs = collect(1:1+rank(psi))
            U, cmb = combineidxs(U, idxs)

            # Find the next site to update
            site1 = site + i - 1

            # Apply SVD and determine the tensors
            U, S, V = svd(U, -1; kwargs...)
            U = contract(U, S, length(size(U)), 1)
            U = moveidx(U, length(size(U)), 1)
            D = site1 == 1 ? 1 : size(psi[site1-1])[2+rank(psi)]
            dims = (size(S)[2], D, [dim(psi) for i=1:rank(psi)]...)
            V = reshape(V, dims)
            V = moveidx(V, 1, -1)

            # Update tensors
            psi[site1] = V
        end
    end

    # Update the final site
    site1 = direction ? site : site + nsites - 1
    psi[site1] = U
    psi.center = site1
    if normalize
        normalize!(psi)
    end
    return true
end


### Random GMPS
"""
    randomGMPS(rank::Int, dim::Int, length::Int, bonddim::Int)

Create a GMPS with random entries.
"""
function randomGMPS(rank::Int, dim::Int, length::Int, bonddim::Int)
    tensors = []
    for i = 1:length
        D1 = i == 1 ? 1 : bonddim
        D2 = i == length ? 1 : bonddim
        idxs = (D1, [dim for i=1:rank]..., D2)
        push!(tensors, randn(Float64, idxs))
    end
    psi = GMPS(rank, dim, tensors, 0)
    movecenter!(psi, length)
    movecenter!(psi, 1)
    idxs = (1, [dim for i=1:rank]..., min(dim^rank, bonddim))
    psi[1] = randn(Float64, idxs)
    normalize!(psi)
    return psi
end


### Converting from quantum state to MPS 
function GMPS(psi::GQS; kwargs...)
    # Store tensors 
    tensors = []

    # SVD 
    tensor = reshape(psi.tensor, (1, size(psi.tensor)..., 1)) 
    for _ = 1:length(psi)-1
        # Reshape & SVD
        tensor, cmb = combineidxs(tensor, [collect(2+rank(psi):length(size(tensor)))...])
        U, S, tensor = svd(tensor, -1; kwargs...)
        tensor = contract(S, tensor, 2, 1)
        
        # Store & put into the correct shape
        push!(tensors, U)
        tensor = reshape(tensor, (size(tensor)[1], cmb[2]...))
    end
    push!(tensors, tensor)

    return GMPS(rank(psi), dim(psi), tensors, length(psi))
end


### Boundary MPS; used for PEPS
"""
    bGMPS(chi::Int, bonddims::Vector{Int}...)

Create a random bMPO with choosen ``physical'' dimensions.
"""
function bGMPS(chi::Int, bonddims::Vector{Int}...)
    # Get GMPS properties
    N = length(bonddims[1])
    r = length(bonddims)

    # Construct MPO
    M = GMPS(r, 1, N)
    for i = 1:N
        D1 = i == 1 ? 1 : chi
        D2 = i == N ? 1 : chi
        M[i] = abs.(randn(Float64, D1, [bonddims[j][i] for j=1:r]..., D2))
    end
    movecenter!(M, N)
    movecenter!(M, 1; maxdim=chi, cutoff=1e-30)
    M[1] = abs.(randn(Float64, 1, [bonddims[j][1] for j=1:r]..., size(M[2])[1]))
    M[1] /= norm(M)
    return M
end

"""
    bGMPSOnes(r::Int, N::Int)

Create a bMPO filled with ones of size N and rank r.
"""
function bGMPSOnes(r::Int, N::Int)
    # Construct MPO
    M = GMPS(r, 1, N)
    for i = 1:N
        M[i] = ones(ComplexF64, [1 for i = 1:2+r]...)
    end
    return M
end



### Save and write
function HDF5.write(parent::Union{HDF5.File, HDF5.Group}, name::AbstractString,
                    M::GMPS)
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
                    ::Type{GMPS})
    g = open_group(parent, name)
    if read(attributes(g)["type"]) != "MPS"
        error("HDF5 group of file does not contain MPS data.")
    end
    N = read(g, "length")
    center = read(g, "center")
    tensors = [read(g, "MPS[$(i)]") for i=1:N]
    rank = length(size(tensors[1])) - 2
    return GMPS(rank, size(tensors[1])[2], tensors, center)
end
