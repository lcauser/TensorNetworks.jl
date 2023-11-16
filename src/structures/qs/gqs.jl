"""
    GQS(rank::Int, dim::Int, tensor::Array{Complex{Float64}})
    GQS(rank::Int, dim::Int, length::Int)

Create a generalised matrix product state with physical dimension dim.
"""
mutable struct GQS{Q<:AbstractArray} <: AbstractQS
    rank::Int
    dim::Int
    length::Int
    tensor::Q
end

function GQS(rank::Int, dim::Int, length::Int)
    return GQS(rank, dim, zeros(ComplexF64, [dim for _ = 1:rank*length]))
end


"""
    norm(psi::GQS)

Calculate the norm of an GQS.
"""
function norm(psi::GQS)
    return norm(psi.tensor)
end


### Normalize
"""
    normalize!(psi::GQS)

Normalize a GQS.
"""
function normalize!(psi::GQS)
    psi.tensor *= norm(psi)^-1
end


### Conjugate
"""
    conj(psi::GQS)

Return the conjugate of an GQS.
"""
function conj(psi::GQS)
    phi = deepcopy(psi)
    phi.tensor = conj(phi.tensor)
    return phi
end


### Equality
function ==(psi1::GQS, psi2::GQS)
    converge = true
    converge = rank(psi1) != rank(psi2) ? false : converge
    converge = length(psi1) != length(psi2) ? false : converge
    converge = dim(psi1) != dim(psi2) ? false : converge
    return converge
end


### Entanglement entropy
"""
    entropy(psi::GQS, site::Int)

Calculate the entanglement entropy between site i and i+1.
"""
function entropy(psi::GQS, site::Int)
    # Check
    (site < 0 || site > length(psi)) && error("The site must be between 1 and length(psi)-1")

    # Reshape
    dims = size(psi.tensor)
    M = reshape(psi.tensor, (dims[1:site*rank(psi)]..., prod(dims[site*rank(psi)+1:end])))
    U, S, V = svd(M, -1)
    S2 = diag(S).^2
    return -sum(S2.*log.(S2))
end


### Random GQS
"""
    randomGQS(rank::Int, dim::Int, length::Int)

Create a GQS with random entries.
"""
function randomGQS(rank::Int, dim::Int, length::Int)
    psi = GQS(rank, dim, length, randn(ComplexF64, [dim for _ = 1:rank*length]...))
    normalize!(psi)
    return psi
end


### Save and write
function HDF5.write(parent::Union{HDF5.File, HDF5.Group}, name::AbstractString,
                    M::GQS)
    g = create_group(parent, name)
    attributes(g)["type"] = "QS"
    attributes(g)["version"] = 1
    write(g, "length", length(M))
    for i = 1:length(M)
        write(g, "tensor", M.tensor)
    end
end


function HDF5.read(parent::Union{HDF5.File, HDF5.Group}, name::AbstractString,
                    ::Type{GQS})
    g = open_group(parent, name)
    if read(attributes(g)["type"]) != "QS"
        error("HDF5 group of file does not contain QS data.")
    end
    N = read(g, "length")
    tensor = read(g, "tensor")
    rank = Int(length(size(tensor)) / N)
    return GQS(rank, size(tensor)[1], length, tensor)
end
