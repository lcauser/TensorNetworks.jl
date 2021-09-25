function contract(x, y, idx1::Int, idx2::Int)
    sz = length(size(x))
    dims1 = [i == idx1 ? 0 : i for i = 1:sz]
    dims2 = [j == idx2 ? 0 : sz + j for j = 1:length(size(y))]
    return tensorcontract(x, dims1, y, dims2)
end

function contract(x, y, idxs1::Vector{Int}, idxs2::Vector{Int})
    length(idxs1) != length(idxs2) && error("The length of contracting indexs differ.")
    labels = [-i for i = 1:length(idxs1)]
    dims1 = [i in idxs1 ? -findall(x -> x == i, idxs1)[1] : i for i = 1:length(size(x))]
    dims2 = [i in idxs2 ? -findall(x -> x == i, idxs2)[1] : i+length(size(x)) for i = 1:length(size(y))]
    return tensorcontract(x, dims1, y, dims2)
end

function tensorproduct(x, y)
    sz1 = length(size(x))
    sz2 = length(size(y))

    dims1 = [i for i = 1:sz1]
    dims2 = [sz1+i for i = 1:sz2]
    #dims3 = [i for i = 1:sz1+sz2]

    return tensorproduct(x, dims1, y, dims2)
end


function trace(x, idx1, idx2)
    sz = length(size(x))
    return tensortrace(x, [i == idx1 || i == idx2 ? 0 : i for i = 1:sz])
end


function combineidxs(x, idxs::Vector{})
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
    return reshape(y, tuple(newshape...)), (copy(idxs), dims)
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
    for i = 1:size(cmb[1])[1]
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


function svd(x, idx::Int; kwargs...)
    # Get truncation parameters
    cutoff::Float64 = get(kwargs, :cutoff, 0)
    maxdim::Int = get(kwargs, :maxdim, 0)
    mindim::Int = get(kwargs, :mindim, 1)

    # Make a copy, get the dimensions
    y = copy(x)
    sz = length(size(y))

    # See if index is -1
    idx = idx == -1 ? sz : idx

    # Group all indexs together and push SVD axis to the end
    idxs = []
    for i = 1:sz
        i != idx && push!(idxs, i)
    end
    y, cmb = combineidxs(y, idxs)
    y = moveidx(y, 1, -1)

    # Apply SVD; try divide and conquer, else use rectangular approach
    local t
    try
        t = svd(y, alg=LinearAlgebra.DivideAndConquer())
    catch e
        println(size(x))
        println(size(y))
        t = svd(y, alg=LinearAlgebra.QRIteration())
    end
    # Assign SVD to individiual matrices
    U = t.U
    S = t.S
    V = t.Vt

    # Determine the number of singular values to keep
    sz = size(S)[1]
    mindim = min(mindim, sz)
    maxdim = (maxdim == 0 || maxdim > sz) ? sz : maxdim
    idxs = findfirst(S == 0)
    maxdim = idxs == nothing ? maxdim : min(maxdim, idxs-1)
    maxdim = maxdim == 0 ? 1 : maxdim
    if cutoff != 0
        S2 = S.^2
        S2cum = reverse(cumsum(reverse(S2))) / sum(S2)
        idxs = findlast([x > cutoff for x = S2cum])
        idxs = idxs == nothing ? 1 : idxs
        maxdim = min(maxdim, idxs)
    end
    vals = max(maxdim, mindim)

    # Truncate
    U = U[:, 1:vals]
    S = diagm(S[1:vals])
    V = V[1:vals, :]

    # Ungroup indexs
    U = moveidx(U, 2, 1)
    U = uncombineidxs(U, cmb)

    return U, S, V
end


function qr(x, idx::Int)
    # Make a copy, get the dimensions
    y = copy(x)
    sz = length(size(y))

    # See if index is -1
    idx = idx == -1 ? sz : idx

    # Group all indexs together and push QR axis to the end
    idxs = []
    for i = 1:sz
        i != idx && push!(idxs, i)
    end
    y, cmb = combineidxs(y, idxs)
    y = moveidx(y, 1, -1)

    # Do the QR decomposition
    t = qr(y)
    Q = Matrix(t.Q)
    R = t.R

    # Ungroup the indexs
    Q = moveidx(Q, 2, 1)
    Q = uncombineidxs(Q, cmb)
    return Q, R
end


function lq(x, idx::Int)
    # Make a copy, get the dimensions
    y = copy(x)
    sz = length(size(y))

    # See if index is -1
    idx = idx == -1 ? sz : idx

    # Group all indexs togethe
    idxs = []
    for i = 1:sz
        i != idx && push!(idxs, i)
    end
    y, cmb = combineidxs(y, idxs)

    # Do the LQ decomposition
    t = lq(y)
    Q = Matrix(t.Q)
    L = t.L

    # Ungroup the indexs
    Q = uncombineidxs(Q, cmb)
    return L, Q
end


function flatten(x)
    return reshape(x, (prod(size(x))))
end


"""
    exp(x, outeridxs::Vector{Int})
    exp(x, outeridx::Int)

Calculate the exponential of a tensor with the specified tensors being the outer
index.
"""
function exp(x, outeridxs::Vector{Int})
    # Group indexs together
    x, cmb1 = combineidxs(x, outeridxs)
    x, cmb2 = combineidxs(x, [i for i=1:length(size(x))-1])
    x = moveidx(x, 2, 1)

    # Exponentiate
    x = exp(x)

    # Reshape to original form
    x = moveidx(x, 2, 1)
    x = uncombineidxs(x, cmb2)
    x = uncombineidxs(x, cmb1)

    return x
end

function exp(x, outeridx::Int)
    return exp(x, [outeridx])
end
