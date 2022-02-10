mutable struct ProjbMPS <: AbstractProjMPS
    psi::GMPS
    env::Environment
    blocks::Vector{Array{Complex{Float64}, 4}}
    center::Int
    direction::Bool
    level::Int
    mpo::GMPS
end

"""
    ProjbMPS(O::MPO, P::MPO)

Construct a projection of the dot of two boundary MPSs.
"""
function ProjbMPS(psi::GMPS, env::Environment, direction::Bool, level::Int; kwargs...)
    # Get key arguments
    squared::Bool = get(kwargs, :squared, false)
    center::Int = get(kwargs, :center, 1)

    # Create Projector
    blocks = [edgeblock(2+length(env.objects)) for i=1:length(psi)]
    mpo = copy(block(env, direction ? level + 1 : level - 1))
    if env.dropoff != 0
        movecenter!(mpo, length(mpo))
        movecenter!(mpo, 1; maxbonddim=env.dropoff)
    end
    projV = ProjbMPS(psi, env, blocks, 0, direction, level, mpo)
    movecenter!(projV, center)
    return projV
end

length(projV::ProjbMPS) = length(projV.psi)

function block(projV::ProjbMPS, idx::Int)
    (idx < 1 || idx > length(projV)) && return edgeblock(length(projV.env.objects)+2)
    return projV.blocks[idx]
end


"""
    buildleft!(projV::ProjbMPS, idx::Int)

Expand the left block using the previous.
"""
function buildleft!(projV::ProjbMPS, idx::Int)
    # Fetch the block to the left and the tensors
    left = block(projV, idx-1)
    A1 = conj(projV.psi[idx])
    A2 = projV.mpo[idx]

    # Determine sites
    site1 = !projV.env.direction ? projV.level : idx
    site2 = projV.env.direction ? projV.level : idx

    # Fetch vector blocks
    M1 = conj(projV.env.objects[1][site1, site2])
    M2 = projV.env.objects[end][site1, site2]

    # Determine rank
    r = length(projV.env.objects)

    # Contract the block with the tensors
    prod = contract(left, A1, 1, 1)
    if projV.env.direction == false && projV.direction == false
        # Building down
        prod = contract(prod, M1, [1, 2+r], [1, 3])
        for i = 1:r-2
            M = projV.env.objects[1+i][site1, site2]
            prod = contract(prod, M, [1, 2+r-i, length(size(prod))], [1, 3, 5])
        end
        prod = contract(prod, M2, [1, 3, length(size(prod))], [1, 3, 5])
        prod = contract(prod, A2, [2*i-1 for i = 1:1+r], [i for i=1:1+r])
    elseif projV.env.direction == false && projV.direction == true
        # Building up
        prod = contract(prod, M1, [1, 2+r], [1, 2])
        for i = 1:r-2
            M = projV.env.objects[1+i][site1, site2]
            prod = contract(prod, M, [1, 2+r-i, length(size(prod))], [1, 2, 5])
        end
        prod = contract(prod, M2, [1, 3, length(size(prod))], [1, 2, 5])
        prod = contract(prod, A2, [2*i-1 for i = 1:1+r], [i for i=1:1+r])
    elseif projV.env.direction == true && projV.direction == false
        # Building right
        prod = contract(prod, M1, [1, 2+r], [2, 4])
        for i = 1:r-2
            M = projV.env.objects[1+i][site1, site2]
            prod = contract(prod, M, [1, 2+r-i, length(size(prod))], [2, 4, 5])
        end
        prod = contract(prod, M2, [1, 3, length(size(prod))], [2, 4, 5])
        prod = contract(prod, A2, [2*i-1 for i = 1:1+r], [i for i=1:1+r])
    else
        # Building left
        prod = contract(prod, M1, [1, 2+r], [2, 1])
        for i = 1:r-2
            M = projV.env.objects[1+i][site1, site2]
            prod = contract(prod, M, [1, 2+r-i, length(size(prod))], [2, 1, 5])
        end
        prod = contract(prod, M2, [1, 3, length(size(prod))], [2, 1, 5])
        prod = contract(prod, A2, [1 , [2*i+2 for i = 1:r]...], [i for i=1:1+r])
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
    A2 = projV.mpo[idx]

    # Determine sites
    site1 = !projV.env.direction ? projV.level : idx
    site2 = projV.env.direction ? projV.level : idx

    # Fetch vector blocks
    M1 = conj(projV.env.objects[1][site1, site2])
    M2 = projV.env.objects[end][site1, site2]

    # Determine rank
    r = length(projV.env.objects)

    # Contract the block with the tensors
    prod = contract(A2, right, length(size(A2)), length(size(right)))
    if projV.env.direction == false && projV.direction == false
        # Building down
        prod = contract(M2, prod, [2, 4], [1+r, 2+2*r])
        for i = 1:r-2
            M = projV.env.objects[r-i][site1, site2]
            prod = contract(M, prod, [2, 4, 6], [length(size(prod))-(r+1-i), length(size(prod)), 3])
        end
        prod = contract(M1, prod, [2, 4, 5], [length(size(prod))-2, length(size(prod)), 3])
        prod = contract(A1, prod, [1+i for i = 1:r+1], [2*i for i = 1:r+1])
    elseif projV.env.direction == false && projV.direction == true
        # Building up
        prod = contract(M2, prod, [3, 4], [1+r, 2+2*r])
        for i = 1:r-2
            M = projV.env.objects[r-i][site1, site2]
            prod = contract(M, prod, [3, 4, 6], [length(size(prod))-(r+1-i), length(size(prod)), 3])
        end
        prod = contract(M1, prod, [3, 4, 5], [length(size(prod))-2, length(size(prod)), 3])
        prod = contract(A1, prod, [1+i for i = 1:r+1], [2*i for i = 1:r+1])
    elseif projV.env.direction == true && projV.direction == false
        # Building right
        prod = contract(M2, prod, [1, 3], [1+r, 2+2*r])
        for i = 1:r-2
            M = projV.env.objects[r-i][site1, site2]
            prod = contract(M, prod, [1, 3, 6], [length(size(prod))-(r+1-i), length(size(prod)), 3])
        end
        prod = contract(M1, prod, [1, 3, 5], [length(size(prod))-2, length(size(prod)), 3])
        prod = contract(A1, prod, [1+i for i = 1:r+1], [2*i for i = 1:r+1])
    else
        # Building left
        prod = contract(M2, prod, [4, 3], [1+r, 2+2*r])
        for i = 1:r-2
            M = projV.env.objects[r-i][site1, site2]
            prod = contract(M, prod, [4, 3, 6], [length(size(prod))-(r+1-i), length(size(prod)), 3])
        end
        prod = contract(M1, prod, [4, 3, 5], [length(size(prod))-2, length(size(prod)), 3])
        prod = contract(A1, prod, [1+i for i = 1:r+1], [[2*i-1 for i = 1:r]..., length(size(prod))])
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
    A2 = projV.mpo[idx]

    # Determine sites
    site1 = !projV.env.direction ? projV.level : idx
    site2 = projV.env.direction ? projV.level : idx

    # Fetch vector blocks
    M1 = conj(projV.env.objects[1][site1, site2])
    M2 = projV.env.objects[end][site1, site2]

    # Determine rank
    r = length(projV.env.objects)

    # Contract the block with the tensors
    prod = contract(left, A1, 1, 1)
    if projV.env.direction == false && projV.direction == false
        # Building down
        prod = contract(prod, M1, [1, 2+r], [1, 3])
        for i = 1:r-2
            M = projV.env.objects[1+i][site1, site2]
            prod = contract(prod, M, [1, 2+r-i, length(size(prod))], [1, 3, 5])
        end
        prod = contract(prod, M2, [1, 3, length(size(prod))], [1, 3, 5])
        prod = contract(prod, A2, [2*i-1 for i = 1:1+r], [i for i=1:1+r])
    elseif projV.env.direction == false && projV.direction == true
        # Building up
        prod = contract(prod, M1, [1, 2+r], [1, 2])
        for i = 1:r-2
            M = projV.env.objects[1+i][site1, site2]
            prod = contract(prod, M, [1, 2+r-i, length(size(prod))], [1, 2, 5])
        end
        prod = contract(prod, M2, [1, 3, length(size(prod))], [1, 2, 5])
        prod = contract(prod, A2, [2*i-1 for i = 1:1+r], [i for i=1:1+r])
    elseif projV.env.direction == true && projV.direction == false
        # Building right
        prod = contract(prod, M1, [1, 2+r], [2, 4])
        for i = 1:r-2
            M = projV.env.objects[1+i][site1, site2]
            prod = contract(prod, M, [1, 2+r-i, length(size(prod))], [2, 4, 5])
        end
        prod = contract(prod, M2, [1, 3, length(size(prod))], [2, 4, 5])
        prod = contract(prod, A2, [2*i-1 for i = 1:1+r], [i for i=1:1+r])
    else
        # Building left
        prod = contract(prod, M1, [1, 2+r], [2, 1])
        for i = 1:r-2
            M = projV.env.objects[1+i][site1, site2]
            prod = contract(prod, M, [1, 2+r-i, length(size(prod))], [2, 1, 5])
        end
        prod = contract(prod, M2, [1, 3, length(size(prod))], [2, 1, 5])
        prod = contract(prod, A2, [1 , [2*i+2 for i = 1:r]...], [i for i=1:1+r])
    end

    prod = contract(prod, right, [i for i = 1:length(size(prod))], [i for i = 1:length(size(prod))])
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
    A2 = projV.mpo[idx]

    # Determine sites
    site1 = !projV.env.direction ? projV.level : idx
    site2 = projV.env.direction ? projV.level : idx

    # Fetch vector blocks
    M1 = conj(projV.env.objects[1][site1, site2])
    M2 = projV.env.objects[end][site1, site2]

    # Determine rank
    r = length(projV.env.objects)

    # Contract the block with the tensors
    prod = contract(A2, right, length(size(A2)), length(size(right)))
    if projV.env.direction == false && projV.direction == false
        # Building down
        prod = contract(M2, prod, [2, 4], [1+r, 2+2*r])
        for i = 1:r-2
            M = projV.env.objects[r-i][site1, site2]
            prod = contract(M, prod, [2, 4, 6], [length(size(prod))-(r+1-i), length(size(prod)), 3])
        end
        prod = contract(M1, prod, [2, 4, 5], [length(size(prod))-2, length(size(prod)), 3])
        prod = contract(left, prod, [1+i for i = 1:r+1], [2*i-1 for i = 1:r+1])
    elseif projV.env.direction == false && projV.direction == true
        # Building up
        prod = contract(M2, prod, [3, 4], [1+r, 2+2*r])
        for i = 1:r-2
            M = projV.env.objects[r-i][site1, site2]
            prod = contract(M, prod, [3, 4, 6], [length(size(prod))-(r+1-i), length(size(prod)), 3])
        end
        prod = contract(M1, prod, [3, 4, 5], [length(size(prod))-2, length(size(prod)), 3])
        prod = contract(left, prod, [1+i for i = 1:r+1], [2*i-1 for i = 1:r+1])
    elseif projV.env.direction == true && projV.direction == false
        # Building right
        prod = contract(M2, prod, [1, 3], [1+r, 2+2*r])
        for i = 1:r-2
            M = projV.env.objects[r-i][site1, site2]
            prod = contract(M, prod, [1, 3, 6], [length(size(prod))-(r+1-i), length(size(prod)), 3])
        end
        prod = contract(M1, prod, [1, 3, 5], [length(size(prod))-2, length(size(prod)), 3])
        prod = contract(left, prod, [1+i for i = 1:r+1], [2*i-1 for i = 1:r+1])
    else
        # Building left
        prod = contract(M2, prod, [4, 3], [1+r, 2+2*r])
        for i = 1:r-2
            M = projV.env.objects[r-i][site1, site2]
            prod = contract(M, prod, [4, 3, 6], [length(size(prod))-(r+1-i), length(size(prod)), 3])
        end
        prod = contract(M1, prod, [4, 3, 5], [length(size(prod))-2, length(size(prod)), 3])
        prod = contract(left, prod, [1+i for i = 1:r+1], [[2*i for i = 1:r]..., length(size(prod))-1])
    end

    return conj(prod)
end


project(projV::ProjbMPS, A0, direction, nsites) = project(projV)
