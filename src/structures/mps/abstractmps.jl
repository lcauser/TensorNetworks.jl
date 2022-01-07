abstract type AbstractMPS end

### Indexing a MPS/MPO
getindex(psi::AbstractMPS, i) = tensors(psi)[i]
function setindex!(psi::AbstractMPS, x, i::Int)
    psi.tensors[i] = x
    return psi
end

### Properties of a MPS/MPO
"""
    eltype(::MPS)

Return the element type of an MPS.
"""
eltype(psi::AbstractMPS) = typeof(psi[1])


"""
    length(::MPS/MPO)

The length of an MPS or MPO.
"""
length(psi::AbstractMPS) = length(psi.tensors)


"""
    dim(::MPS/MPO)

The size of the physical dimensions in an MPS or MPO.
"""
dim(psi::AbstractMPS) = psi.dim


"""
    center(::MPS/MPO)

The orthogonal center of an MPS or MPO. Returns 0 if not set.
"""
center(psi::AbstractMPS) = psi.center


"""
    tensors(::MPS/MPO)

Return the tensor within an MPS or MPO
"""
tensors(psi::AbstractMPS) = psi.tensors

"""
    bonddim(m::MPS, idx::Int)
    bonddim(m::MPO, idx::Int)

Return the bond dimension size between idx and idx + 1. Returns nothing if
out of range.
"""
function bonddim(psi::AbstractMPS, site::Int)
    (site < 1 || site >= length(psi)) && return nothing
    return size(psi[site])[3]
end


"""
    maxbonddim(::MPS/::MPO)

Calculate the maximum bond dimension within an MPS or MPO.
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


### Move orthogonal center
"""
    movecenter!(psi::MPS, idx::Int; kwargs...)

Move the orthogonal center of an MPS.
"""
function movecenter!(psi::AbstractMPS, idx::Int; kwargs...)
    (idx < 1 || idx > length(psi)) && error("The idx is out of range.")
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


### Truncate
"""
    truncate!(psi::AbstractMPS; kwargs...)

Truncate across the MPS. Use key arguments:
    - mindim: minimum bond dimension (default = 1)
    - maxdim: maximum bond dimension (default = 0, no limit)
    - cutoff: truncation cutoff error (default = 0)
"""
function truncate!(psi::AbstractMPS; kwargs...)
    if psi.center != 1 && psi.center != length(psi)
        movecenter!(psi, 0)
    end
    ctr = center(psi) == 1 ? length(psi) : 1
    movecenter!(psi, ctr; kwargs...)
end


function conj(psi::AbstractMPS)
    phi = deepcopy(psi)
    for i = 1:length(phi)
        phi[i] = conj(phi[i])
    end
    return phi
end


### Creating copies
copy(psi::AbstractMPS) = typeof(psi)(dim(psi), tensors(psi), center(psi))
deepcopy(psi::AbstractMPS) = typeof(psi)(copy(dim(psi)), copy(tensors(psi)),
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


### Entanglement entropy
function entropy(psi::AbstractMPS, site::Int)
    movecenter!(psi, site)
    U, S, V = svd(psi[site], -1)
    S2 = diag(S).^2
    return -sum(S2.*log.(S2))
end
