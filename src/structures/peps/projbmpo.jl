mutable struct ProjbMPO <: AbstractProjMPS
    psi::MPO
    env::Environment
    blocks::Vector{Array{Complex{Float64}, 4}}
    center::Int
    direction::Bool
    level::Int
    mpo::MPO
end

"""
    ProjbMPO(O::MPO, P::MPO)

Construct a projection of the dot of two boundary MPOs.
"""
function ProjbMPO(psi::MPO, env::Environment, direction::Bool, level::Int; kwargs...)
    # Get key arguments
    squared::Bool = get(kwargs, :squared, false)
    center::Int = get(kwargs, :center, 1)

    # Create Projector
    blocks = [edgeblock(ProjbMPO) for i=1:length(psi)]
    mpo = copy(block(env, direction ? level + 1 : level - 1))
    if env.dropoff != 0
        movecenter!(mpo, length(mpo))
        movecenter!(mpo, 1; maxbonddim=env.dropoff)
    end
    projV = ProjbMPO(psi, env, blocks, 0, direction, level, mpo)
    movecenter!(projV, center)
    return projV
end

length(projV::ProjbMPO) = length(projV.psi)


"""
    edgeblock(::Type{ProjbMPO})

Return an edge block.
"""
function edgeblock(::Type{ProjbMPO})
    return convert(Array{Complex{Float64}, 4}, ones((1, 1, 1, 1)))
end

"""
    buildleft!(projV::ProjbMPO, idx::Int)

Expand the left block using the previous.
"""
function buildleft!(projV::ProjbMPO, idx::Int)
    # Fetch the block to the left and the tensors
    left = block(projV, idx-1)
    A1 = conj(projV.psi[idx])
    A2 = projV.mpo[idx]

    # Fetch PEPS tensors
    site1 = !projV.env.direction ? projV.level : idx
    site2 = projV.env.direction ? projV.level : idx
    M1 = conj(projV.env.psi[site1, site2])
    M2 = projV.env.phi[site1, site2]

    # Contract the block with the tensors
    prod = contract(left, A1, 1, 1)
    if projV.env.direction == false && projV.direction == false
        prod = contract(prod, M1, [1, 4], [1, 3])
        prod = contract(prod, M2, [1, 3, 7], [1, 3, 5])
        prod = contract(prod, A2, [1, 3, 5], [1, 2, 3])
    elseif projV.env.direction == false && projV.direction == true
        prod = contract(prod, M1, [1, 4], [1, 2])
        prod = contract(prod, M2, [1, 3, 7], [1, 2, 5])
        prod = contract(prod, A2, [1, 3, 5], [1, 2, 3])
    elseif projV.env.direction == true && projV.direction == false
        prod = contract(prod, M1, [1, 4], [2, 4])
        prod = contract(prod, M2, [1, 3, 7], [2, 4, 5])
        prod = contract(prod, A2, [1, 3, 5], [1, 2, 3])
    else
        prod = contract(prod, M1, [1, 4], [2, 1])
        prod = contract(prod, M2, [1, 3, 7], [2, 1, 5])
        prod = contract(prod, A2, [1, 4, 6], [1, 2, 3])
    end

    # Save the block
    projV[idx] = prod
end


"""
    buildright!(projV::ProjbMPO, idx::Int)

Expand the right block using the previous.
"""
function buildright!(projV::ProjbMPO, idx::Int)
    # Fetch the block to the left and the tensors
    right = block(projV, idx+1)
    A1 = conj(projV.psi[idx])
    A2 = projV.mpo[idx]

    # Fetch PEPS tensors
    site1 = !projV.env.direction ? projV.level : idx
    site2 = projV.env.direction ? projV.level : idx
    M1 = conj(projV.env.psi[site1, site2])
    M2 = projV.env.phi[site1, site2]

    # Contract the block with the tensors
    prod = contract(A2, right, 4, 4)
    if projV.env.direction == false && projV.direction == false
        prod = contract(M2, prod, [2, 4], [3, 6])
        prod = contract(M1, prod, [2, 4, 5], [5, 7, 3])
        prod = contract(A1, prod, [2, 3, 4], [2, 4, 6])
    elseif projV.env.direction == false && projV.direction == true
        prod = contract(M2, prod, [3, 4], [3, 6])
        prod = contract(M1, prod, [3, 4, 5], [5, 7, 3])
        prod = contract(A1, prod, [2, 3, 4], [2, 4, 6])
    elseif projV.env.direction == true && projV.direction == false
        prod = contract(M2, prod, [1, 3], [3, 6])
        prod = contract(M1, prod, [1, 3, 5], [5, 7, 3])
        prod = contract(A1, prod, [2, 3, 4], [2, 4, 6])
    else
        prod = contract(M2, prod, [4, 3], [3, 6])
        prod = contract(M1, prod, [4, 3, 5], [5, 7, 3])
        prod = contract(A1, prod, [2, 3, 4], [1, 3, 6])
    end

    # Save the block
    projV[idx] = prod
end


function calculate(projV::ProjbMPO)
    idx = projV.center
    # Fetch the block to the left and the tensors
    left = block(projV, idx-1)
    right = block(projV, idx+1)
    A1 = conj(projV.psi[idx])
    A2 = projV.mpo[idx]

    # Fetch PEPS tensors
    site1 = !projV.env.direction ? projV.level : idx
    site2 = projV.env.direction ? projV.level : idx
    M1 = conj(projV.env.psi[site1, site2])
    M2 = projV.env.phi[site1, site2]

    # Contract the block with the tensors
    prod = contract(left, A1, 1, 1)
    if projV.env.direction == false && projV.direction == false
        prod = contract(prod, M1, [1, 4], [1, 3])
        prod = contract(prod, M2, [1, 3, 7], [1, 3, 5])
        prod = contract(prod, A2, [1, 3, 5], [1, 2, 3])
    elseif projV.env.direction == false && projV.direction == true
        prod = contract(prod, M1, [1, 4], [1, 2])
        prod = contract(prod, M2, [1, 3, 7], [1, 2, 5])
        prod = contract(prod, A2, [1, 3, 5], [1, 2, 3])
    elseif projV.env.direction == true && projV.direction == false
        prod = contract(prod, M1, [1, 4], [2, 4])
        prod = contract(prod, M2, [1, 3, 7], [2, 4, 5])
        prod = contract(prod, A2, [1, 3, 5], [1, 2, 3])
    else
        prod = contract(prod, M1, [1, 4], [2, 1])
        prod = contract(prod, M2, [1, 3, 7], [2, 1, 5])
        prod = contract(prod, A2, [1, 4, 6], [1, 2, 3])
    end

    prod = contract(prod, right, [1, 2, 3, 4], [1, 2, 3, 4])
    return prod[1]
end

"""
    project(projV::ProjbMPO)

Determine the projection onto the sites at the center in the given direciton.
"""
function project(projV::ProjbMPO)
    idx = projV.center
    # Fetch the block to the left and the tensors
    left = block(projV, idx-1)
    right = block(projV, idx+1)
    A1 = conj(projV.psi[idx])
    A2 = projV.mpo[idx]

    # Fetch PEPS tensors
    site1 = !projV.env.direction ? projV.level : idx
    site2 = projV.env.direction ? projV.level : idx
    M1 = conj(projV.env.psi[site1, site2])
    M2 = projV.env.phi[site1, site2]

    # Contract the block with the tensors
    prod = contract(A2, right, 4, 4)
    if projV.env.direction == false && projV.direction == false
        prod = contract(M2, prod, [2, 4], [3, 6])
        prod = contract(M1, prod, [2, 4, 5], [5, 7, 3])
        prod = contract(left, prod, [2, 3, 4], [1, 3, 5])
    elseif projV.env.direction == false && projV.direction == true
        prod = contract(M2, prod, [3, 4], [3, 6])
        prod = contract(M1, prod, [3, 4, 5], [5, 7, 3])
        prod = contract(left, prod, [2, 3, 4], [1, 3, 5])
    elseif projV.env.direction == true && projV.direction == false
        prod = contract(M2, prod, [1, 3], [3, 6])
        prod = contract(M1, prod, [1, 3, 5], [5, 7, 3])
        prod = contract(left, prod, [2, 3, 4], [1, 3, 5])
    else
        prod = contract(M2, prod, [4, 3], [3, 6])
        prod = contract(M1, prod, [4, 3, 5], [5, 7, 3])
        prod = contract(left, prod, [2, 3, 4], [2, 4, 5])
    end

    return prod
end
