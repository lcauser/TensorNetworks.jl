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
    projV = ProjMPS(psi, phi)

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
