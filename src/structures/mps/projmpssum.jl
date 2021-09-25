mutable struct ProjMPSSum <: AbstractProjMPS
    projs::Vector{Proj} where Proj <: AbstractProjMPS
    center::Int
end

"""
    projMPS(psi::MPS, phi::MPS)

Construct a projection on the dot product of psi and phi.
"""
function ProjMPSSum(projVs::Vector{Proj}; kwargs...) where Proj <: AbstractProjMPS
    # Get key arguments
    center::Int = get(kwargs, :center, 1)

    # Create Projector
    projVs = ProjMPSSum(projVs, center)
    movecenter!(projVs, center)
    return projVs
end


"""
    buildleft!(projVs::ProjMPSSum, idx::Int)

Expand the left block using the previous.
"""
function buildleft!(projVs::ProjMPSSum, idx::Int)
    for projV in projVs.projs
        buildleft!(projV, idx)
    end
end


"""
    buildright!(projVs::ProjMPSSum, idx::Int)

Expand the right block using the previous.
"""
function buildright!(projVs::ProjMPSSum, idx::Int)
    for projV in projVs.projs
        buildright!(projV, idx)
    end
end


"""
    movecenter!(projVs::ProjMPSSum, idx::Int)

Expand the right block using the previous.
"""
function movecenter!(projVs::ProjMPSSum, idx::Int)
    for projV in projVs.projs
        movecenter!(projV, idx)
    end
end

"""
    project(projVs::ProjMPSSum, A, direction::Bool = 0, nsites::Int = 2)

Determine the projection onto the sites at the center in the given direciton.
"""
function project(projVs::ProjMPSSum, A, direction::Bool = 0, nsites::Int = 2)
    prod = 1
    for i = 1:length(projVs.projs)
        if i == 1
            prod = project(projVs.projs[i], A, direction, nsites)
        else
            prod += project(projVs.projs[i], A, direction, nsites)
        end
    end
    return prod
end


function calculate(projVs::ProjMPSSum)
    prod = 1
    for i = 1:length(projVs.projs)
        if i == 1
            prod = calculate(projVs.projs[i])
        else
            prod += calculate(projVs.projs[i])
        end
    end
    return prod
end
