abstract type AbstractQS end

### Indexing a quantum state
getindex(psi::AbstractQS, idxs::Q...) where Q<:Int = psi.tensor[idxs...]
function setindex!(psi::AbstractQS, x::Q, idxs::T...) where {Q<:AbstractArray, T<:Int}
    if len(idxs) == 0
        psi.tensor = x
    else
        psi.tensors[T...] = x
    end
    return psi
end

### Properties of a quantum state
"""
    eltype(::AbstractQW)

Return the element type of an MPS.
"""
eltype(psi::AbstractQS) = typeof(psi.tensor)


"""
    length(::AbstractQS)

The length of a quantum state.
"""
length(psi::AbstractQS) = psi.length


"""
    dim(::AbstractQS)

The size of the physical dimensions in a quantum state
"""
dim(psi::AbstractQS) = psi.dim


"""
    rank(::AbstractQS)

Return the rank of a QS.
"""
rank(psi::AbstractQS) = psi.rank



function Base.show(io::IO, M::AbstractQS)
    println(io, M.tensor)
end



### Creating copies
copy(psi::AbstractQS) = typeof(psi)(rank(psi), dim(psi), length(psi), psi.tensor)
deepcopy(psi::AbstractQS) = typeof(psi)(copy(rank(psi)), copy(dim(psi)), length(psi), psi.tensor)


### Products with numbers
function *(psi::AbstractQS, a::Q) where Q<:Number
    phi = deepcopy(psi)
    phi.tensor *= a
    return phi
end
*(a::Number, psi::AbstractQS) = *(psi, a)
/(psi::AbstractQS, a::Number) = *(psi, 1/a)
