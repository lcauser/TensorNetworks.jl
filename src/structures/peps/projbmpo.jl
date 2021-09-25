mutable struct ProjbMPO <: AbstractProjMPS
    O::MPO
    P::MPO
    blocks::Vector{Array{Complex{Float64}, 2}}
    center::Int
    squared::Bool
end

"""
    ProjbMPO(O::MPO, P::MPO)

Construct a projection of the dot of two boundary MPOs.
"""
function ProjbMPO(O::MPO, P::MPO; kwargs...)
    # Get key arguments
    squared::Bool = get(kwargs, :squared, false)
    center::Int = get(kwargs, :center, 1)

    # Create Projector
    length(O) != length(P) && error("The bMPOs must have matching lengths.")
    dim(O) != dim(P) && error("The bMPOs must have matching physical dims.")
    blocks = [edgeblock(ProjbMPO) for i=1:length(O)]
    projV = ProjbMPO(O, P, blocks, 0, squared)
    movecenter!(projV, center)
    return projV
end

length(projV::ProjbMPO) = length(projV.O)


"""
    edgeblock(::Type{ProjbMPO})

Return an edge block.
"""
function edgeblock(::Type{ProjbMPO})
    return convert(Array{Complex{Float64}, 2}, ones((1, 1)))
end

"""
    buildleft!(projV::ProjbMPO, idx::Int)

Expand the left block using the previous.
"""
function buildleft!(projV::ProjbMPO, idx::Int)
    # Fetch the block to the left and the tensors
    left = block(projV, idx-1)
    A1 = conj(projV.O[idx])
    A2 = projV.P[idx]

    # Contract the block with the tensors
    prod = contract(left, A1, 1, 1)
    prod = contract(prod, A2, [1, 2, 3], [1, 2, 3])

    # Save the block
    projV[idx] = prod
end


"""
    buildright!(projV::ProjbMPO, idx::Int)

Expand the right block using the previous.
"""
function buildright!(projV::ProjbMPO, idx::Int)
    # Fetch the block to the right and the tensors
    right = block(projV, idx+1)
    A1 = conj(projV.O[idx])
    A2 = projV.P[idx]

    # Contract the block with the tensors
    prod = contract(A2, right, 4, 2)
    prod = contract(A1, prod, [2, 3, 4], [2, 3, 4])

    # Save the block
    projV[idx] = prod
end

"""
    project(projV::ProjbMPO)

Determine the projection onto the sites at the center in the given direciton.
"""
function project(projV::ProjbMPO)
    # Determine the site
    site = projV.center

    # Get the blocks
    left = block(projV, site - 1)
    right = block(projV, site + 1)

    # Do the contraction
    O = conj(projV.O[site])
    prod = contract(left, O, 1, 1)
    prod = contract(prod, right, 4, 1)

    return prod
end


function calculate(projV::ProjbMPO)
    # Determine the site
    site = projV.center

    # Get the blocks
    left = block(projV, site - 1)
    right = block(projV, site + 1)

    # Do the contraction
    O = conj(projV.O[site])
    P = projV.P[site]
    prod = contract(left, O, 1, 1)
    prod = contract(prod, P, [1, 2, 3], [1, 2, 3])
    prod = contract(prod, right, [1, 2], [1, 2])

    return prod[1]
end
