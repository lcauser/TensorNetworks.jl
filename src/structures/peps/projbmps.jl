mutable struct ProjbMPO <: AbstractProjMPS
    psi::MPS
    env::SingleEnvironment
    blocks::Vector{Array{Complex{Float64}, 3}}
    center::Int
    direction::Bool
    level::Int
end

"""
    ProjbMPS(O::MPS, P::MPS)

Construct a projection of the dot of two boundary MPSs.
"""
function ProjbMPO(psi::MPS, env::SingleEnvironment, direction::Bool, level::Int; kwargs...)
    # Get key arguments
    center::Int = get(kwargs, :center, 1)

    # Create Projector
    blocks = [edgeblock(ProjbMPS) for i=1:length(psi)]
    projV = ProjbMPS(psi, env, blocks, 0, direction, level)
    movecenter!(projV, center)
    return projV
end

length(projV::ProjbMPS) = length(projV.psi)


"""
    edgeblock(::Type{ProjbMPS})

Return an edge block.
"""
function edgeblock(::Type{ProjbMPS})
    return convert(Array{Complex{Float64}, 3}, ones((1, 1, 1)))
end

"""
    buildleft!(projV::ProjbMPS, idx::Int)

Expand the left block using the previous.
"""
function buildleft!(projV::ProjbMPS, idx::Int)
    # Fetch the block to the left and the tensors
    left = block(projV, idx-1)
    A1 = conj(projV.psi[idx])
    A2 = block(projV.env, projV.direction ? projV.level + 1 : projV.level - 1)[idx]

    # Fetch PEPS tensors
    site1 = !projV.env.direction ? projV.level : idx
    site2 = projV.env.direction ? projV.level : idx
    M = conj(projV.env.psi[site1, site2])

    # Contract the block with the tensors
    prod = contract(left, A1, 1, 1)
    if projV.direction == false
        prod = contract(prod, M, [1, 3], [1, 3])
        prod = contract(prod, A2, [1, 3], [1, 2])
    else
        prod = contract(prod, M, [1, 3], [1, 2])
        prod = contract(prod, A2, [1, 3], [1, 2])
    end

    # Save the block
    projV[idx] = prod
end


"""
    buildright!(projV::ProjbMPS, idx::Int)

Expand the right block using the previous.
"""
function buildright!(projV::ProjbMPS, idx::Int)
    # Fetch the block to the left and the tensors
    right = block(projV, idx+1)
    A1 = conj(projV.psi[idx])
    A2 = block(projV.env, projV.direction ? projV.level + 1 : projV.level - 1)[idx]

    # Fetch PEPS tensors
    site1 = !projV.env.direction ? projV.level : idx
    site2 = projV.env.direction ? projV.level : idx
    M = conj(projV.env.psi[site1, site2])

    # Contract the block with the tensors
    prod = contract(A2, right, 3, 3)
    if projV.direction == false
        prod = contract(M, prod, [2, 4], [2, 4])
        prod = contract(A1, prod, [2, 3], [2, 4])
    else
        prod = contract(M, prod, [3, 4], [2, 4])
        prod = contract(A1, prod, [2, 3], [2, 4])
    end

    # Save the block
    projV[idx] = prod
end


function calculate(projV::ProjbMPS)
    idx = projV.center
    # Fetch the block to the left and the tensors
    left = block(projV, idx-1)
    right = block(projV, idx+1)
    A1 = conj(projV.psi[idx])
    A2 = block(projV.env, projV.direction ? projV.level + 1 : projV.level - 1)[idx]

    # Fetch PEPS tensors
    site1 = !projV.env.direction ? projV.level : idx
    site2 = projV.env.direction ? projV.level : idx
    M = conj(projV.env.psi[site1, site2])

    # Contract the block with the tensors
    prod = contract(left, A1, 1, 1)
    if projV.direction == false
        prod = contract(prod, M, [1, 3], [1, 3])
        prod = contract(prod, A2, [1, 3], [1, 2])
    else
        prod = contract(prod, M, [1, 3], [1, 2])
        prod = contract(prod, A2, [1, 3], [1, 2])
    end

    prod = contract(prod, right, [1, 2, 3], [1, 2, 3])
    return prod[1]
end

"""
    project(projV::ProjbMPS)

Determine the projection onto the sites at the center in the given direciton.
"""
function project(projV::ProjbMPS)
    idx = projV.center
    # Fetch the block to the left and the tensors
    left = block(projV, idx-1)
    right = block(projV, idx+1)
    A1 = conj(projV.psi[idx])
    A2 = block(projV.env, projV.direction ? projV.level + 1 : projV.level - 1)[idx]

    # Fetch PEPS tensors
    site1 = !projV.env.direction ? projV.level : idx
    site2 = projV.env.direction ? projV.level : idx
    M = conj(projV.env.psi[site1, site2])

    # Contract the block with the tensors
    prod = contract(A2, right, 3, 3)
    if projV.direction == false
        prod = contract(M, prod, [2, 4], [2, 4])
        prod = contract(left, prod, [2, 3], [1, 3])
    else
        prod = contract(M, prod, [3, 4], [2, 4])
        prod = contract(left, prod, [2, 3], [1, 3])
    end

    return prod
end
