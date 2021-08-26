mutable struct ProjMPO <: AbstractProjMPS
    psi::MPS
    O::MPO
    blocks::Vector{Array{Complex{Float64}, 3}}
    center::Int
end

"""
    projMPO(psi::MPS, O::MPO)

Construct a projection of an expectation of an MPS on an MPO.
"""
function ProjMPO(psi::MPS, O::MPO; kwargs...)
    # Get key arguments
    center::Int = get(kwargs, :center, 1)

    # Create Projector
    length(psi) != length(O) && error("The MPS and MPO must have matching lengths.")
    dim(psi) != dim(O) && error("The MPS and MPO must have matching physical dims.")
    blocks = [edgeblock(ProjMPO) for i=1:length(psi)]
    projO = ProjMPO(psi, O, blocks, 0)
    movecenter!(projO, center)
    return projO
end


"""
    edgeblock(::Type{ProjMPO})

Return an edge block.
"""
function edgeblock(::Type{ProjMPO})
    return convert(Array{Complex{Float64}, 3}, ones((1, 1, 1)))
end

"""
    buildleft!(projO::ProjMPO, idx::Int)

Expand the left block using the previous.
"""
function buildleft!(projO::ProjMPO, idx::Int)
    # Fetch the block to the left and the tensors
    left = block(projO, idx-1)
    A = projO.psi[idx]
    O = projO.O[idx]

    # Contract the block with the tensors
    prod = contract(left, conj(A), 1, 1)
    prod = contract(prod, O, 1, 1)
    prod = trace(prod, 2, 4)
    prod = contract(prod, A, 1, 1)
    prod = trace(prod, 2, 4)

    # Save the block
    projO[idx] = prod
end


"""
    buildright!(projO::ProjMPO, idx::Int)

Expand the right block using the previous.
"""
function buildright!(projO::ProjMPO, idx::Int)
    # Fetch the block to the right and the tensors
    right = block(projO, idx+1)
    A = projO.psi[idx]
    O = projO.O[idx]

    # Contract the block with the tensors
    prod = contract(A, right, 3, 3)
    prod = contract(O, prod, 4, 4)
    prod = trace(prod, 3, 5)
    prod = contract(conj(A), prod, 3, 4)
    prod = trace(prod, 2, 4)

    # Save the block
    projO[idx] = prod
end

"""
    project(projO::ProjMPO, A, direction::Bool = 0, nsites::Int = 2)

Determine the projection onto the sites at the center in the given direciton.
"""
function project(projO::ProjMPO, A, direction::Bool = 0, nsites::Int = 2)
    # Determine the site
    site = direction ? projO.center - nsites + 1 : projO.center

    # Get the blocks
    left = block(projO, site - 1)
    right = block(projO, site + nsites)

    # Loop through taking the product
    prod = contract(left, A, 3, 1)
    prod = moveidx(prod, 2, -1)
    for i = 1:nsites
        # Fetch tensor
        O = projO.O[site-1+i]

        # Contract
        prod = contract(prod, O, length(size(prod)), 1)
        prod = trace(prod, 2, length(size(prod))-1)
    end

    # Contract with right block
    prod = contract(prod, right, 2, 3)
    prod = trace(prod, length(size(prod))-2, length(size(prod)))
    return prod
end


function calculate(projO::ProjMPO)
    # Determine the site
    site = projO.center

    # Get the blocks
    left = block(projO, site - 1)
    right = block(projO, site + 1)
    A = projO.psi[site]
    O = projO.O[site]

    # Contract the block with the tensors
    prod = contract(left, conj(A), 1, 1)
    prod = contract(prod, O, 1, 1)
    prod = trace(prod, 2, 4)
    prod = contract(prod, A, 1, 1)
    prod = trace(prod, 2, 4)

    # Contract with right
    prod = contract(prod, right, 1, 1)
    prod = trace(prod, 1, 3)
    prod = trace(prod, 1, 2)
    return prod[1]
end
