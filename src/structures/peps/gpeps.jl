"""
    GPEPS(rank::Int, dim::Int, tensors::Array{Array{ComplexF64}})
    GPEPS(rank::Int, dim::Int, height::Int, length::Int)
    GPEPS(rank::Int, dim::Int, size::Int)

Initiate a GPEPS. Rank referes to the rank of tensor it is approximating -
PEPS rank 1, PEPO rank 2.
"""
mutable struct GPEPS <: AbstractPEPS
    rank::Int
    dim::Int
    tensors::Array{Array{ComplexF64, 5}, 2}
end

function GPEPS(rank::Int, dim::Int, height::Int, length::Int)
    tensors = Array{Array{ComplexF64, 4+rank}, 2}(undef, (height, length))
    for i = 1:height
        for j = 1:length
            tensors[i, j] = zeros(ComplexF64, 1, 1, 1, 1, [dim for i=1:rank]...)
        end
    end

    return GPEPS(rank, dim, tensors)
end

function GPEPS(rank::Int, dim::Int, size::Int)
    return GPEPS(rank, dim, size, size)
end

"""
    randomGPEPS(rank::Int, dim::Int, length::Int, height::Int, bonddim::Int)
    randomGPEPS(rank::Int, dim::Int, size::Int, bonddim::Int)

Create a GPEPS with random tensor entries.
"""
function randomGPEPS(rank::Int, dim::Int, length::Int, height::Int, bonddim::Int)
    psi = GPEPS(rank, dim, length, height)
    for i = 1:length
        for j = 1:length
            A = randn(ComplexF64, bonddim, bonddim, bonddim, bonddim, [dim for i=1:rank]...)
            if j == 1
                A = A[1:1, :, :, :, [1:dim for i=1:rank]...]
            end
            if j == length
                A = A[:, :, :, 1:1, [1:dim for i=1:rank]...]
            end
            if i == 1
                A = A[:, 1:1, :, :, [1:dim for i=1:rank]...]
            end
            if i == length
                A = A[:, :, 1:1, :, [1:dim for i=1:rank]...]
            end
            A /= norm(A)
            psi[i, j] = A
        end
    end
    return psi
end


"""
    rescale!(psi::GPEPS, maxabs::Real=0)

Rescale the tensors of a GPEPS to have the same maximum absolute value.
"""
function rescale!(psi::GPEPS, maxabs::Real=0)
    if maxabs==0
        for i = 1:size(psi)[1]
            for j = 1:size(psi)[2]
                maxabs += log(maximum(abs.(psi[i, j])))
            end
        end
        maxabs = exp(maxabs/(length(psi)^2))
    end

    for i = 1:size(psi)[1]
        for j = 1:size(psi)[2]
            psi[i, j] *= maxabs / (maximum(abs.(psi[i, j])))
        end
    end
end

### Equality
function ==(psi1::GPEPS, psi2::GPEPS)
    converge = true
    converge = rank(psi1) != rank(psi2) ? false : converge
    converge = length(psi1) != length(psi2) ? false : converge
    converge = dim(psi1) != dim(psi2) ? false : converge
    return converge
end

### Reduced tensors
"""
    reducedtensor(psi::GPEPS, i::Int, j::Int, axis::Int)
    reducedtensor(A, axis::Int)

Split a tensor into a its reduced tensor.
"""
function reducedtensor(psi::GPEPS, i::Int, j::Int, axis::Int)
    # Check it's a virutal axis
    (axis < 1 || axis > 4) && error("Axis must be between 1 and 4.")

    # Group relevent idx with physical axis
    A = psi[i, j]
    d = dim(psi)
    D = size(A)[axis]
    A, cmb = combineidxs(A, [axis, [4+i for i = 1:rank(psi)]...])

    # Apply SVD to split to reduced tensor
    U, S, V = svd(A, 4)
    V = contract(S, V, 2, 1)

    # Organize the axes
    U = moveidx(U, 4, axis)
    V = reshape(V, (size(V)[1], D, [d for i = 1:rank(psi)]...))
    if axis == 1 || axis == 2
        V = moveidx(V, 1, 2)
    end
    return U, V
end

function reducedtensor(A, axis::Int)
    # Check it's a virutal axis
    (axis < 1 || axis > 4) && error("Axis must be between 1 and 4.")

    # Group relevent idx with physical axis
    d = size(A)[5]
    r = length(size(A))-4
    D = size(A)[axis]
    A, cmb = combineidxs(A, [axis, [4+i for i = 1:r]...])

    # Apply QR to split to reduced tensor
    #U, S, V = svd(A, 4)
    #V = contract(S, V, 2, 1)
    U, V = qr(A, 4)

    # Organize the axes
    U = moveidx(U, 4, axis)
    V = reshape(V, (size(V)[1], D, [d for i = 1:r]...))
    if axis == 1 || axis == 2
        V = moveidx(V, 1, 2)
    end
    return U, V
end

### Save and write
function HDF5.write(parent::Union{HDF5.File, HDF5.Group}, name::AbstractString,
                    psi::GPEPS)
    g = create_group(parent, name)
    attributes(g)["type"] = "PEPS"
    attributes(g)["version"] = 1
    write(g, "height", size(psi)[1])
    write(g, "length", size(psi)[2])
    write(g, "dim", psi.dim)
    for i = 1:size(psi)[1]
        for j = 1:size(psi)[2]
            write(g, "PEPS[$(i), $(j)]", psi[i, j])
        end
    end
end

function HDF5.read(parent::Union{HDF5.File, HDF5.Group}, name::AbstractString,
                    ::Type{GPEPS})
    g = open_group(parent, name)
    if read(attributes(g)["type"]) != "PEPS"
        error("HDF5 group of file does not contain PEPS data.")
    end
    height = read(g, "height")
    len = read(g, "length")
    dim = read(g, "dim")
    rank = length(size(read(g, "PEPS[1, 1]")))-4
    psi = GPEPS(rank, dim, height, len)
    for i = 1:height
        for j = 1:len
            psi[i, j] = read(g, "PEPS[$(i), $(j)]")
        end
    end
    return psi
end
