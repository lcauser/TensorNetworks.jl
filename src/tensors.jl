import LinearAlgebra.svd

function combineidxs(x, idxs::Vector{Int})
    # Make a copy and get the dimensions of the grouping indexs
    y = copy(x)
    idxs = sort(idxs)
    dims = size(x)[idxs]

    # Permute indexs to the end
    for i = 1:length(idxs)
        y = moveidx(y, idxs[i]+1-i, -1)
    end

    # Determine current shape and new shape
    shape = size(y)
    newshape = [shape[i] for i = 1:length(shape)-length(idxs)]
    push!(newshape, prod(shape[length(shape)+1-length(idxs):end]))
    return reshape(y, tuple(newshape...)), (idxs, dims)
end

function uncombineidxs(x, cmb)
    # Make a copy
    y = copy(x)
    offset = length(size(y))

    # Reshape back to give old indices
    newdims = zeros(Int, (offset + length(cmb[2]) - 1))
    newdims[1:offset-1] .= size(y)[1:offset-1]
    newdims[offset:end] .= cmb[2]
    y = reshape(y, tuple(newdims...))

    # Permute indices back to the correct places
    println(size(y))
    for i = 1:length(size(cmb[1]))
        y = moveidx(y, offset-1+i, cmb[1][i])
    end
    return y
end

function moveidx(x, currentidx::Int, newidx::Int)
    # Check to see if they're the same
     currentidx == newidx && return copy(x)

    # Determine the dimensions and idx location
    dims = size(x)
    newidx = newidx == -1 ? length(dims) : newidx

    # Determine the new ordering
    ordering = []
    i = 1
    for idx = 1:length(dims)
        (idx == currentidx && newidx > currentidx) && (i+=1)
        if idx == newidx
            push!(ordering, currentidx)
        else
            push!(ordering, i)
            i += 1
        end
        (idx == currentidx && newidx < currentidx) && (i+=1)
    end

    # Permute
    y = permutedims(x, ordering)
    return y
end


function svd(x, idx::int; kwargs...)
    # Get truncation parameters
    cutoff::Float64 = get(kwargs, :cutoff, 0)
    maxdim::Int = get(kwargs, :maxdim, 0)
    mindim::Int = get(kwargs, :mindim, 1)

    # Make a copy
    y = copy(x)


end
