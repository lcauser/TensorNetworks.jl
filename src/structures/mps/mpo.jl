"""
    MPO(dim::Int, length::Int)

Create a MPO with physical dimension dim.
"""
function MPO(dim::Int, length::Int)
    tensors = []
    for i = 1:length
        push!(tensors, zeros(ComplexF64, (1, dim, dim, 1)))
    end
    return GMPS(2, dim, tensors, 0)
end


"""
    ismpo(psi::GMPS)

Check if a GMPS is of rank 2.
"""
function ismpo(O::GMPS)
    rank(O) == 2 && return true
    return false
end


# Manipulations of MPO
"""
    adjoint(O::GMPS)

Calculate the Hermitian conjugate of an operator (MPO).
"""
function adjoint(O::GMPS)
    rank(O) != 2 && error("The generalised MPS must be of rank 2 (an MPO).")
    O2 = deepcopy(O)
    for i = 1:length(O2)
        O2[i] = conj(moveidx(O2[i], 2, 3))
    end
    return O2
end


### Create MPOs
"""
    randomMPO(dim::Int, length::Int, bonddim::Int)

Create a MPO with random entries.
"""
function randomMPO(dim::Int, length::Int, bonddim::Int)
    return randomGMPS(2, dim, length, bonddim)
end


"""
    productMPO(sites::Int, A::Array{Complex{Float64}, 4})
    productMPO(sites::Int, A)

Create a product MPO of some fixed tensor.
A can be a vector for product state entries, or larger dimensional tensor which
is truncated at the edge sites.
"""
function productMPO(sites::Int, A::Array{Complex{Float64}, 4})
    tensors = Array{Complex{Float64}, 4}[]
    push!(tensors, A[1:1, :, :, :])
    for i = 2:sites-1
        push!(tensors, A)
    end
    push!(tensors, A[:, :, :, end:end])
    return GMPS(2, size(A)[2], tensors, 0)
end

function productMPO(sites::Int, A)
    if length(size(A)) == 2
        A = reshape(A, (1, size(A)[1], size(A)[2], 1))
    end
    A = convert(Array{Complex{Float64}, 4}, A)
    return productMPO(sites, A)
end


"""
    productMPO(st::Sitetypes, names::Vector{String})

Create a product operator from the names of local operators on a sitetype.
"""
function productMPO(st::Sitetypes, names::Vector{String})
    tensors = []
    for i = 1:length(names)
        A = convert(Array{Complex{Float64}, 4},
                    reshape(op(st, names[i]), (1, st.dim, st.dim, 1)))
        push!(tensors, A)
    end
    return GMPS(2, st.dim, tensors, 0)
end


### Products
"""
    applyMPO(O::GMPO, psi::GMPS; kwargs...)
    applyMPO(psi::GMPS, O::GMPS; kwargs...)
    applyMPO(O::GMPS, O::GMPS; kwargs...)

Apply an operator (MPO) O to a state (MPS) psi, O|Ψ> or <Ψ|O.
Or apply an operator (MPO) O1 to an operator (MPO) O2.
"""
function applyMPO(arg1::GMPS, arg2::GMPS; kwargs...)
    # Check the arguments share the same physical dimensions and length
    dim(arg1) != dim(arg2) && error("GMPS must share the same physical dims.")
    length(arg1) != length(arg2) && error("GMPS must share the same length.")

    # Determine which is psi and which is O
    if rank(arg1)==1 && rank(arg2)==2
        return MPOMPSProduct(adjoint(arg2), arg1; kwargs...)
    elseif rank(arg1)==2 && rank(arg2)==1
        return MPOMPSProduct(arg1, arg2; kwargs...)
    elseif rank(arg1)==2 && rank(arg2)==2
        return MPOMPOProduct(arg1, arg2; kwargs...)
    else
        error("Unallowed combinations of MPS ranks.")
        return false
    end
end
*(O::GMPS, psi::GMPS) = applyMPO(O, psi)


function MPOMPSProduct(O::GMPS, psi::GMPS; kwargs...)
    # Create an empty MPS
    phi = GMPS(1, dim(psi), length(psi))

    # Loop through applying the MPO, and move the gauge across
    for i = 1:length(psi)
        A = psi[i]
        M = O[i]
        B = contract(M, A, 3, 2)
        B, cmb1 = combineidxs(B, [1, 4])
        B, cmb2 = combineidxs(B, [2, 3])
        B = moveidx(B, 1, 2)
        phi[i] = B
        if i > 1
            moveright!(phi, i-1)
        end
    end

    # Orthogonalize to first site with truncation
    movecenter!(phi, 1; kwargs...)
    return phi
end

function MPOMPOProduct(O1::GMPS, O2::GMPS; kwargs...)
    # Create empty MPO
    O = GMPS(2, dim(O1), length(O1))

    # Loop through applying the MPO, and move the gauge across
    for i = 1:length(O1)
        M1 = O1[i]
        M2 = O2[i]
        M = contract(M1, M2, 3, 2)
        M, cmb1 = combineidxs(M, [1, 4])
        M, cmb2 = combineidxs(M, [2, 4])
        M = moveidx(M, 3, 1)
        O[i] = M
        if i > 1
            moveright!(O, i-1)
        end
    end

    # Orthogonalize to first site with truncation
    movecenter!(O, 1; kwargs...)
    return O
end

"""
    inner(phi::GMPS, psi::GMPS)
    inner(phi::GMPS, O::GMPS, psi::GMPS)
    inner(phi::GMPS, ..., psi::GMPS)
    inner(O1:GMPS, phi::GMPS, O2::GMPS, psi::GMPS)
    dot(phi::GMPS, psi::GMPS)
    *(phi::GMPS, psi::GMPS)

Calculate the inner product of some operator with respect to a bra and ket.
"""
function inner(args::GMPS...)
    # Check to make sure all arguments have the same properties
    length(args) < 2 && error("There must be atleast 2 MPS arguments (rank 1).")
    dims = [dim(arg)==dim(args[1]) for arg in args]
    sum(dims) != length(args) && error("GMPS must share the same physical dim.")
    lengths = [length(arg)==length(args[1]) for arg in args]
    sum(lengths) != length(args) && error("GMPS must share the same length.")

    # Check for a bra and ket
    ranks = [rank(arg) for arg in args]
    sum([rank == 1 for rank in ranks]) != 2 && error("The inner product must have a braket structure.")
    ranks[end] != 1 && error("The final GMPS must be rank 1 (MPS).")

    # Re-arrange to form a braket
    idx = 0
    for j = 1:length(ranks)
        idx = j
        ranks[j] == 1 && break
    end
    for j = idx-1:-1:1
        psi = args[j+1]
        O = adjoint(args[j])
        args[j] = psi
        args[j+1] = O
    end

    # Calculate the inner product
    prod = ones(ComplexF64, ([1 for i=1:length(args)]...))
    for site = 1:length(args[1])
        prod = contract(prod, conj(args[1][site]), 1, 1)
        for j = 1:length(args)-1
            prod = contract(prod, args[1+j][site], [1, length(size(prod))-1], [1, 2])
        end
    end

    return prod[[1 for i=1:length(args)]...]
end
dot(phi::GMPS, psi::GMPS) = inner(phi, psi)
*(phi::GMPS, psi::GMPS) = inner(phi, psi)


"""
    trace(O::GMPS)
    trace(O1::GMPS, O2::GMPS)
    trace(O1::GMPS, O2::GMPS, ...)

Determine the trace of an MPO / product of MPOs..
"""
function trace(args::GMPS...)
    # Check to make sure all arguments have the same properties
    length(args) < 1 && error("There must be atleast 1 MPO arguments (rank 2).")
    dims = [dim(arg)==dim(args[1]) for arg in args]
    sum(dims) != length(args) && error("GMPS must share the same physical dim.")
    lengths = [length(arg)==length(args[1]) for arg in args]
    sum(lengths) != length(args) && error("GMPS must share the same length.")

    # Check that they're MPOs
    ranks = [rank(arg) for arg in args]
    sum([rank != 2 for rank in ranks]) > 0 && error("Arguments must be GMPS of rank 2 (MPO).")

    # Calculate the trace
    prod = ones(ComplexF64, ([1 for i=1:length(args)]...))
    for site = 1:length(args[1])
        prod = contract(prod, conj(args[1][site]), 1, 1)
        for j = 1:length(args)-1
            prod = contract(prod, args[1+j][site], [1, length(size(prod))-1], [1, 2])
        end
        prod = trace(prod, 1, length(size(prod))-1)
    end

    return prod[[1 for i=1:length(args)]...]
end

### Add MPOs
"""
    addMPOs(O1::MPO, O2::MPO; kwargs...)

Add two MPOs and apply SVD to truncate.
"""
function addMPOs(O1::GMPS, O2::GMPS; kwargs...)
    # Check MPOs are the same
    O1 != O2 && error("GMPOs must share the same properties.")
    rank(O1) != 2 && error("GMPOs must be rank 2 (MPO).")

    # Fetch information
    d = dim(O1)
    N = length(O1)

    # Create empty MPO
    O = GMPS(2, d, N)

    # Iterative create tensors
    for i = 1:N
        dims1 = size(O1[i])
        dims2 = size(O2[i])
        D1 = i == 1 ? 1 : dims1[1] + dims2[1]
        D2 = i == N ? 1 : dims1[4] + dims2[4]

        # Create tensor
        M = zeros(ComplexF64, (D1, d, d, D2))
        if i == 1
            M[1:1, :, :, 1:dims1[4]] = O1[i]
            M[1:1, :, :, dims1[4]+1:D2] = O2[i]
        elseif i == N
            M[1:dims1[1], :, :, 1:1] = O1[i]
            M[dims1[1]+1:D1, :, :, 1:1] = O2[i]
        else
            M[1:dims1[1], :, :, 1:dims1[4]] = O1[i]
            M[dims1[1]+1:D1, :, :, dims1[4]+1:D2] = O2[i]
        end

        # Store tensor
        O[i] = M
    end

    # Apply SVD in an attempt to reduce the bond dimension
    for site = 1:N-1
        M = O[site]
        U, S, V = svd(M, 4; cutoff=cutoff, maxdim=maxdim, mindim=mindim)
        M = contract(U, S, 4, 1)
        O[site] = M
        O[site+1] = contract(V, O[site+1], 2, 1)
    end

    for site = N:-1:2
        M = O[site]
        U, S, V = svd(M, 1; cutoff=cutoff, maxdim=maxdim, mindim=mindim)
        M = contract(S, U, 2, 1)
        O[site] = M
        O[site-1] = contract(O[site-1], V, 4, 2)
    end
    return O
end
+(O1::GMPS, O2::GMPS) = addMPOs(O1, O2; cutoff=1e-15)


### Automatically construct MPOs from operator lists
"""
     MPO(st::Sitetypes, H::OpList; kwargs...)

Construct an MPO from an operator list.
"""
function MPO(st::Sitetypes, H::OpList; kwargs...)
    # Truncation information
    cutoff::Float64 = get(kwargs, :cutoff, 1e-15)
    maxdim::Int = get(kwargs, :maxdim, 0)
    mindim::Int = get(kwargs, :mindim, 1)

    # System properties
    N = length(H)
    d = st.dim

    # Create empty MPO
    ten = zeros(ComplexF64, (2, d, d, 2))
    ten[1, :, :, 1] = op(st, "id")
    ten[2, :, :, 2] = op(st, "id")
    O = GMPS(2, d, N)
    O[1] = deepcopy(ten[1:1, :, :, 1:2])
    for i = 2:N-1
        O[i] = deepcopy(ten[1:2, :, :, 1:2])
    end
    O[N] = deepcopy(ten[1:2, :, :, 2:2])

    # Loop through each term and determine the site range
    maxrng = siterange(H)
    rngs = [[] for i = 1:maxrng]
    for i = 1:length(H.ops)
        rng = H.sites[i][end] - H.sites[i][1] + 1
        push!(rngs[rng], i)
    end

    # Loop through all the possible ranges of interactions
    for i = 1:maxrng
        # Loop through sites
        nextterms = [[] for j=1:i]
        coeffs = [[] for j=1:i]
        ingoings = [[] for j=1:i]
        outgoings = [[] for j=1:i]

        for site = 1:N
            # Find all the terms which start at the site
            idxs = []
            for j = rngs[i]
                if H.sites[j][1] == site
                    push!(idxs, j)
                end
            end

            if i == 1
                # Just adding to top right corner
                for idx in idxs
                    O[site][1, :, :, end] += H.coeffs[idx]*op(st, H.ops[idx][1])
                end
            else
                # Add new terms starting at this site
                for j = 1:length(idxs)
                    # Fetch operator information
                    ops = H.ops[idxs[j]]
                    sites = H.sites[idxs[j]]
                    coeff = H.coeffs[idxs[j]]

                    # Loop through each site in the operator
                    outgoing = 0
                    for k = 1:i
                        # Decide ingoing and outgoing idxs
                        ingoing = outgoing
                        for l = 1:length(outgoings[k])+1
                            outgoing = l
                            !(outgoing in outgoings[k]) && break
                        end
                        outgoing = k == i ? 0 : outgoing

                        # Determine what the operator is
                        op =  site+k-1 in sites ? ops[argmax([s == site+k-1 for s = sites])] : "id"

                        # Add to list
                        push!(nextterms[k], op)
                        push!(coeffs[k], k == 1 ? H.coeffs[idxs[j]] : 1)
                        push!(ingoings[k], ingoing)
                        push!(outgoings[k], outgoing)
                    end
                end

                # Pull the terms
                terms = nextterms[1]
                ins = ingoings[1]
                outs = outgoings[1]
                cos = coeffs[1]
                for j = 1:i-1
                    nextterms[j] = nextterms[j+1]
                    ingoings[j] = ingoings[j+1]
                    outgoings[j] = outgoings[j+1]
                    coeffs[j] = coeffs[j+1]
                end
                nextterms[i] = []
                ingoings[i] = []
                outgoings[i] = []
                coeffs[i] = []

                # Expand the tensor to account for all terms
                if length(terms) != 0
                    ingoinglen = sum([ingoing != 0 for ingoing in ins])
                    outgoinglen = sum([outgoing != 0 for outgoing in outs])
                    ingoingsrt = size(O[site])[1] - 1
                    outgoingsrt = size(O[site])[4] - 1
                    O[site] = expand(O[site], ingoinglen, outgoinglen)

                    # Add the terms to the tensor
                    for j = 1:length(terms)
                        # Find the idxs of each
                        x = ins[j] == 0 ? 1 : ingoingsrt + ins[j]
                        y = outs[j] == 0 ? outgoingsrt + 1 + outgoinglen : outgoingsrt + outs[j]

                        # Set the tensor
                        O[site][x, :, :, y] = cos[j] * op(st, terms[j])
                    end
                end
            end
        end
    end

    # Apply SVD in an attempt to reduce the bond dimension
    for site = 1:N-1
        M = O[site]
        U, S, V = svd(M, 4; cutoff=cutoff, maxdim=maxdim, mindim=mindim)
        M = contract(U, S, 4, 1)
        O[site] = M
        O[site+1] = contract(V, O[site+1], 2, 1)
    end

    for site = N:-1:2
        M = O[site]
        U, S, V = svd(M, 1; cutoff=cutoff, maxdim=maxdim, mindim=mindim)
        M = contract(S, U, 2, 1)
        O[site] = M
        O[site-1] = contract(O[site-1], V, 4, 2)
    end
    return O
end


function expand(O::Array{}, D1, D2)
    dims = size(O)
    newO = zeros(ComplexF64, (dims[1]+D1, dims[2], dims[3], dims[4]+D2))
    d1 = dims[1] == 1 ? 1 : dims[1]-1
    newO[1:d1, :, :, 1:dims[4]-1] = O[1:d1, :, :, 1:dims[4]-1]
    newO[1:d1, :, :, end] = O[1:d1, :, :, end]
    newO[dims[1]+D1, :, :, dims[4]+D2] = O[dims[1], :, :, dims[4]]
    return newO
end
