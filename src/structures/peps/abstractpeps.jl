abstract type AbstractPEPS end

### Indexing a PEPS
getindex(psi::AbstractPEPS, i, j) = tensors(psi)[i, j]
function setindex!(psi::AbstractPEPS, x, i::Int, j::Int)
    psi.tensors[i, j] = x
    return psi
end

### Properties of a MPS/MPO
"""
    eltype(::AbstractPEPS)

Return the element type of a PEPS.
"""
eltype(psi::AbstractPEPS) = typeof(psi[1, 1])


"""
    size(::AbstractPEPS)

The sizes of a PEPS.
"""
size(psi::AbstractPEPS) = Base.size(psi.tensors)


"""
    length(::AbstractPEPS)

The length (x direction) of a PEPS.
"""
length(psi::AbstractPEPS) = size(psi)[2]


"""
    dim(::AbstractPEPS)

The size of the physical dimensions in a PEPS.
"""
dim(psi::AbstractPEPS) = psi.dim


"""
    tensors(::PEPS)

Return the tensor within a PEPS.
"""
tensors(psi::AbstractPEPS) = psi.tensors


"""
    bonddim(psi::AbstractPEPS, i::Int, j::Int, vertical::Bool=false)

Return the bond dimension size at site i, j in the horizontal/vertical
direction (to the right or below).
"""
function bonddim(psi::AbstractPEPS, i::Int, j::Int, vertical::Bool=false)
    (i < 1 || i > size(psi)[1]) && return nothing
    (j < 1 || j > size(psi)[2]) && return nothing
    (i == size(psi)[1] && vertical==false) && return nothing
    (j == size(psi)[2] && vertical==true) && return nothing
    vertical == true && return size(psi[i, j])[3]
    vertical == false && return size(psi[i, j])[4]
end


"""
    maxbonddim(::AbstractPEPS)

Calculate the maximum bond dimension within a PEPS
"""
function maxbonddim(psi::AbstractPEPS)
    D = 0
    sz = size(psi)
    for i = 1:sz[1]
        for j = 1:sz[2]
            D = i < sz[1] ? max(D, bonddim(psi, i, j, false)) : D
            D = j < sz[2] ? max(D, bonddim(psi, i, j, true)) : D
        end
    end
    return D
end


function Base.show(io::IO, M::AbstractPEPS)
    println(io, "$(typeof(M))")
    sz = size(M)
    for i = 1:sz[1]
        for j = 1:sz[2]
            println(io, "[$(i)] $(size(M[i, j]))")
        end
    end
end


### Creating copies
copy(psi::AbstractPEPS) = typeof(psi)(dim(psi), tensors(psi))
deepcopy(psi::AbstractPEPS) = typeof(psi)(copy(dim(psi)), copy(tensors(psi)))


### Products with numbers
function *(psi::AbstractPEPS, a::Number)
    phi = deepcopy(psi)
    phi.tensors[1] *= a
    return phi
end
*(a::Number, psi::AbstractPEPS) = *(psi, a)
/(psi::AbstractPEPS, a::Number) = *(psi, 1/a)
