abstract type AbstractProjMPS end

"""
    length(projV::AbstractProjMPS)

Determine the number of sites in the projected MPS.
"""
length(projV::AbstractProjMPS) = length(projV.objects[1])


"""
    center(projV::AbstractProjMPS)

Return the center of the block.
"""
center(projV::AbstractProjMPS) = projV.center


"""
    rank(projV::AbstractProjMPS)

Return the tensor rank of a projection.
"""
rank(projV::AbstractProjMPS) = projV.rank


### Handle blocks
"""
    edgeblock(num::Int)

Return an edge block.
"""
function edgeblock(num::Int)
    return ones(ComplexF64, [1 for i=1:num]...)
end


"""
    block(projV::AbstractProjMPS, idx::Int)

Fetch the block at a given site.
"""
function block(projV::AbstractProjMPS, idx::Int)
    (idx < 1 || idx > length(projV)) && return edgeblock(length(projV.objects))
    return projV.blocks[idx]
end

### Indexing a projection
getindex(projV::AbstractProjMPS, idx::Int) = block(projV, idx)
function setindex!(projV::AbstractProjMPS, x, idx::Int)
    projV.blocks[idx] = x
    return projV
end

"""
    movecenter!(projV::AbstractProjMPS, idx::Int)

Move the center of the projection.
"""
function movecenter!(projV::AbstractProjMPS, idx::Int)
    if center(projV) == 0
        for i = 1:idx-1
            buildleft!(projV, i)
        end
        N = length(projV)
        for i = 1:N-idx
            buildright!(projV, N+1-i)
        end
    else
        if idx > center(projV)
            for i = 1:idx-center(projV)
                buildleft!(projV, center(projV) - 1 + i)
            end
        elseif idx < center(projV)
            for i = 1:center(projV)-idx
                buildright!(projV, center(projV) + 1 - i)
            end
        end
    end
    projV.center = idx
end


### Define the * operation; will be used when optimizing sites.
*(projV::AbstractProjMPS, A) = project(projV, A)
*(A, projV::AbstractProjMPS) = project(projV, A)
