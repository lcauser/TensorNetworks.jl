mutable struct OpList
    length::Int
    ops::Vector{Vector{String}}
    sites::Vector{Vector{Int}}
    coeffs::Vector{Complex{Float64}}
end

"""
    OpList(length::Int)

Create a list of operators acting on a lattice.
"""
OpList(length::Int) = OpList(length, [], [], [])

length(oplist::OpList) = oplist.length

function deepcopy(oplist::OpList)
    oplist2 = OpList(copy(oplist.length))
    oplist2.sites = copy(oplist.sites)
    oplist2.ops = copy(oplist.ops)
    oplist2.coeffs = copy(oplist.coeffs)
    return oplist2
end

### Add to the list
"""
    add!(oplist::OpList, ops::Vector{String}, sites::Vector{Int},
         coeff::Complex{Float64} = 1)
    add!(oplist::OpList, op::String, site::Int, coeff::Complex{Float64} = 1)

Add an operator to the list defined by local operators at given sites.
"""
function add!(oplist::OpList, ops::Vector{String}, sites::Vector{Int},
              coeff::Number = 1.0)
    # Validate the data
    length(ops) != length(sites) && error("The lists must be the same length.")

    # Ordet the operators and sites
    perms = sortperm(sites)
    sites = sites[perms]
    ops = ops[perms]

    lastsite = 0
    for site in sites
        (0 > site || site > oplist.length) && error("The sites must be between 1 and $(oplist.length).")
        sum([site == site2 for site2 in sites]) > 1 && error("There are two or more operators on the same site.")
        site <= lastsite && error("The site list must be ordered.")
        lastsite = site
    end

    # Add to list
    push!(oplist.ops, ops)
    push!(oplist.sites, sites)
    push!(oplist.coeffs, coeff)
end

function add!(oplist::OpList, op::String, site::Int, coeff::Number = 1.0)
    add!(oplist, [op], [site], coeff)
end


"""
    add(oplist1::OpList, oplist2::OpList)

Join two oplists.
"""
function add(oplist1::OpList, oplist2::OpList)
    oplist = OpList(max(oplist1.length, oplist2.length))
    oplist.sites = copy(oplist1.sites)
    append!(oplist.sites, oplist2.sites)
    oplist.ops = copy(oplist1.ops)
    append!(oplist.ops, oplist2.ops)
    oplist.coeffs = copy(oplist1.coeffs)
    append!(oplist.coeffs, oplist2.coeffs)
    return oplist
end


function *(x::Number, y::OpList)
    y = deepcopy(y)
    for i = 1:length(y.coeffs)
        y.coeffs[i] *= x
    end
    return y
end


### Determine properties about the operator list
"""
    siteRange(oplist::OpList)

Determine the interaction range within an operator list.
"""
function siterange(oplist::OpList)
    rng = 1
    for sites in oplist.sites
        rng = max(rng, sites[end]-sites[1]+1)
    end

    return rng
end


"""
    siteindexs(oplist::OpList, site::Int)

Determine the indexs of operators in a list which start at a given site.
"""
function siteindexs(oplist::OpList, site::Int)
    idxs = []
    for i = 1:length(oplist.sites)
        if oplist.sites[i][1] == site
            push!(idxs, i)
        end
    end
    return idxs
end


"""
    totensor(oplist::OpList, st::Sitetypes, idx::Int)

Construct the tensor from an operator in an oplist.
"""
function totensor(oplist::OpList, st::Sitetypes, idx::Int)
    # Fetch the relevent information
    ops = oplist.ops[idx]
    sites = oplist.sites[idx]
    rng = min(siterange(oplist), oplist.length - sites[1] + 1)

    # Create the tensor through a tensor product
    prod = ones((1, 1))
    i = 1
    for site = 1:rng
        if sites[1]+site-1 in sites
            oper = ops[i]
            i += 1
        else
            oper = "id"
        end
        oper = reshape(op(st, oper), (1, st.dim, st.dim, 1))
        prod = contract(prod, oper, length(size(prod)), 1)
    end
    prod = trace(prod, 1, length(size(prod)))

    return oplist.coeffs[idx]*prod
end


"""
    sitetensor(oplist::OpList, st::Sitetypes, idx::Int)

Return the tensor for all operators starting at a site.
"""
function sitetensor(oplist::OpList, st::Sitetypes, idx::Int)
    # Validate the idx
    (idx < 0 || idx > oplist.length) && error("Site index is out of range.")

    # Get the indexs which start at the site
    idxs = siteindexs(oplist, idx)
    length(idxs) == 0 && return false

    # Loop through adding them
    ten = 1
    for i = 1:length(idxs)
        if i == 1
            ten = totensor(oplist, st, idxs[i])
        else
            ten += totensor(oplist, st, idxs[i])
        end
    end

    return ten
end



### Calculate inner products of each operator in the operator list with respect
### to two MPSs
"""
    inner(st::Sitetypes, psi::MPS, oplist::OpList, phi::MPS)

Efficently calculate the expectations of a list of operators between two
MPS.
"""
function inner(st::Sitetypes, psi::MPS, oplist::OpList, phi::MPS)
    # Create a projection on to the MPSs
    projV = ProjMPS(phi, psi)

    # Loop through each site
    expectations = [0.0 + 0.0im for i = 1:length(oplist.sites)]
    for site = 1:length(psi)
        # Move the projection center to the site
        movecenter!(projV, site)

        # Find the indexs of operators which start at the site
        idxs = siteindexs(oplist, site)

        # Loop through each and calculate the expectation
        for idx in idxs
            # Determine the range of the operator, and fetch relevent blocks
            rng = oplist.sites[idx][end] - oplist.sites[idx][1] + 1
            left = block(projV, site-1)
            right = block(projV, site+rng)

            # Loop through the middle sites applying operators where necersary
            for i = 1:rng
                # Fetch the site tensors
                A = conj(psi[site-1+i])
                B = phi[site-1+i]

                # Apply the operator
                idx2 = findfirst([site2 == (site - 1 + i) for site2 = oplist.sites[idx]])
                if idx2 != nothing
                    B = contract(op(st, oplist.ops[idx][idx2]), B, 2, 2)
                    B = moveidx(B, 1, 2)
                end

                # Contract to left block
                left = contract(left, A, 1, 1)
                left = contract(left, B, 1, 1)
                left = trace(left, 1, 3)
            end

            # Contract with right and store
            prod = contract(left, right, 1, 1)
            prod = trace(prod, 1, 2)
            expectations[idx] = oplist.coeffs[idx] * prod[1]
        end
    end

    return expectations
end

"""
    applyop!(psi::MPS, st::Sitetypes, ops, sites, coeff::Number=1)

Apply local operators to an MPS.
"""
function applyop!(psi::MPS, st::Sitetypes, ops, sites, coeff::Number=1)
    for i = 1:length(sites)
        # Fetch the tensors
        A = psi[sites[i]]
        O = op(st, ops[i])

        # Apply the operator
        A = contract(O, A, 2, 2)
        A = moveidx(A, 1, 2)
        psi[sites[i]] = A
    end
end


### Automatically construct MPOs from operator lists
"""
    MPO(H::OpList, st::Sitetypes; kwargs...)

Constructs an MPO from an operator list by sequentially adding terms and
applying SVD.
"""
function MPO(H::OpList, st::Sitetypes; kwargs...)
    # Truncation information
    cutoff::Float64 = get(kwargs, :cutoff, 1e-12)
    maxdim::Int = get(kwargs, :maxdim, 0)
    mindim::Int = get(kwargs, :mindim, 1)

    # Fetch information
    d = st.dim
    N = length(H)

    # Create empty MPO
    O = MPO(d, N)

    for i = 1:length(H.ops)
        # Fetch operator info
        ops = H.ops[i]
        sites = H.sites[i]
        coeff = H.coeffs[i]

        # Create full operator list
        opers = String[]
        for j = 1:N
            op = j in sites ? ops[argmax([j == site for site = sites])] : "id"
            push!(opers, op)
        end

        # Create MPO and add
        O2 = productMPO(st, opers)
        O2[sites[1]] *= coeff
        println(i)
        O = addMPOs(O, O2; cutoff=cutoff, maxdim=maxdim, mindim=mindim)
        println([bonddim(O, i) for i = 1:N-1])
    end

    return O
end
