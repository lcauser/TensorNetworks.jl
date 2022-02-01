abstract type AbstractMPS end

### Indexing a MPS/MPO
getindex(psi::AbstractMPS, i) = tensors(psi)[i]
function setindex!(psi::AbstractMPS, x, i::Int)
    psi.tensors[i] = x
    return psi
end

### Properties of a MPS/MPO
"""
    eltype(::AbstractMPS)

Return the element type of an MPS.
"""
eltype(psi::AbstractMPS) = typeof(psi[1])


"""
    length(::AbstractMPS)

The length of an MPS or MPO.
"""
length(psi::AbstractMPS) = length(psi.tensors)


"""
    dim(::AbstractMPS)

The size of the physical dimensions in an MPS or MPO.
"""
dim(psi::AbstractMPS) = psi.dim


"""
    rank(::AbstractMPS)

Return the rank of a GMPS.
"""
rank(psi::AbstractMPS) = psi.rank


"""
    center(::AbstractMPS)

The orthogonal center of an MPS or MPO. Returns 0 if not set.
"""
center(psi::AbstractMPS) = psi.center


"""
    tensors(::AbstractMPS)

Return the tensor within an MPS or MPO
"""
tensors(psi::AbstractMPS) = psi.tensors

"""
    bonddim(::AbstractMPS, idx::Int)

Return the bond dimension size between idx and idx + 1. Returns nothing if
out of range.
"""
function bonddim(psi::AbstractMPS, site::Int)
    (site < 1 || site > length(psi)) && return nothing
    return size(psi[site+1])[1]
end


"""
    maxbonddim(::AbstractMPS)

Calculate the maximum bond dimension within an GMPS.
"""
function maxbonddim(psi::AbstractMPS)
    D = 0
    for i = 1:length(psi)-1
        D = max(D, bonddim(psi, i))
    end
    return D
end


function Base.show(io::IO, M::AbstractMPS)
    println(io, "$(typeof(M))")
    for i = 1:length(M)
        println(io, "[$(i)] $(size(M[i]))")
    end
end



### Creating copies
copy(psi::AbstractMPS) = typeof(psi)(rank(psi), dim(psi), tensors(psi), center(psi))
deepcopy(psi::AbstractMPS) = typeof(psi)(copy(rank(psi)), copy(dim(psi)), copy(tensors(psi)),
                                        copy(center(psi)))


### Products with numbers
function *(psi::AbstractMPS, a::Number)
    phi = deepcopy(psi)
    if center(psi) != 0
        phi.tensors[center(phi)] *= a
    else
        phi.tensors[1] *= a
    end
    return phi
end
*(a::Number, psi::AbstractMPS) = *(psi, a)
/(psi::AbstractMPS, a::Number) = *(psi, 1/a)
