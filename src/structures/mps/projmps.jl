mutable struct ProjMPS <: AbstractProjMPS
    psi::MPS
    phi::MPS
    blocks::Vector{Array{Complex{Float64}, 2}}
    center::Int
    squared::Bool
end

"""
    projMPS(psi::MPS, phi::MPS)

Construct a projection on the dot product of psi and phi.
"""
function ProjMPS(psi::MPS, phi::MPS; kwargs...)
    # Get key arguments
    squared::Bool = get(kwargs, :squared, false)
    center::Int = get(kwargs, :center, 1)

    # Create Projector
    length(psi) != length(phi) && error("The MPS must have matching lengths.")
    dim(psi) != dim(phi) && error("The MPS must have matching physical dims.")
    blocks = [edgeblock(ProjMPS) for i=1:length(psi)]
    projV = ProjMPS(psi, phi, blocks, 0, squared)
    movecenter!(projV, center)
    return projV
end


"""
    edgeblock(::Type{ProjMPS})

Return an edge block.
"""
function edgeblock(::Type{ProjMPS})
    return convert(Array{Complex{Float64}, 2}, ones((1, 1)))
end

"""
    buildleft!(projV::ProjMPS, idx::Int)

Expand the left block using the previous.
"""
function buildleft!(projV::ProjMPS, idx::Int)
    # Fetch the block to the left and the tensors
    left = block(projV, idx-1)
    A1 = conj(projV.phi[idx])
    A2 = projV.psi[idx]

    # Contract the block with the tensors
    prod = contract(left, A1, 1, 1)
    prod = contract(prod, A2, 1, 1)
    prod = trace(prod, 1, 3)

    # Save the block
    projV[idx] = prod
end


"""
    buildright!(projV::ProjMPS, idx::Int)

Expand the right block using the previous.
"""
function buildright!(projV::ProjMPS, idx::Int)
    # Fetch the block to the right and the tensors
    right = block(projV, idx+1)
    A1 = conj(projV.phi[idx])
    A2 = projV.psi[idx]

    # Contract the block with the tensors
    prod = contract(A2, right, 3, 2)
    prod = contract(A1, prod, 3, 3)
    prod = trace(prod, 2, 4)

    # Save the block
    projV[idx] = prod
end

"""
    project(projV::ProjMPS, A, direction::Bool = 0, nsites::Int = 2)

Determine the projection onto the sites at the center in the given direciton.
"""
function project(projV::ProjMPS, A, direction::Bool = 0, nsites::Int = 2)
    # Determine the site
    site = direction ? projV.center - nsites + 1 : projV.center

    # Get the blocks
    left = block(projV, site - 1)
    right = block(projV, site + nsites)

    # Loop through taking the product
    prod = moveidx(left, 1, -1)
    for i = 1:nsites
        B = conj(projV.phi[site - 1 + i])
        prod = contract(prod, B, length(size(prod)), 1)
    end
    prod = contract(prod, right, length(size(prod)), 1)


    # Get the square
    if projV.squared == true
        prod2 = contract(left, A, 2, 1)
        prod2 = moveidx(prod2, 1, -1)
        for i = 1:nsites
            B = conj(projV.phi[site - 1 + i])
            prod2 = contract(prod2, B, length(size(prod)), 1)
            prod2 = trace(prod2, 1, length(size(prod2))-1)
        end
        prod2 = contract(prod2, right, 1, 2)
        prod2 = trace(prod2, 1, 2)
        prod *= conj(prod2[1])
    end

    return conj(prod)
end


function calculate(projV::ProjMPS)
    # Determine the site
    site = projV.center

    # Get the blocks
    left = block(projV, site - 1)
    right = block(projV, site + 1)
    A = projV.psi[site]
    B = conj(projV.phi[site])

    # Contract blocks with tensors
    prod = contract(left, B, 1, 1)
    prod = contract(prod, A, 1, 1)
    prod = trace(prod, 1, 3)
    prod = contract(prod, right, 1, 1)
    prod = trace(prod, 1, 2)

    if projV.squared == true
        return prod[1]*conj(prod[1])
    end
    return prod[1]
end
