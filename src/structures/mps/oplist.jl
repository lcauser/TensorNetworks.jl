mutable struct OpList{Q<:Number}
    length::Int
    ops::Vector{Vector{String}}
    sites::Vector{Vector{Int}}
    coeffs::Vector{Q}
end

"""
    OpList(length::Int)

Create a list of operators acting on a lattice.
"""
OpList(length::Int) = OpList(length, Vector{String}[], Vector{Int}[], ComplexF64[])

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
         coeff<:Number = 1)
    add!(oplist::OpList, op::String, site::Int, coeff<:Number = 1)

Add an operator to the list defined by local operators at given sites.
"""
function add!(oplist::OpList, ops::Vector{String}, sites::Vector{Int},
              coeff::Q = 1.0) where {Q<:Number}
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

function add!(oplist::OpList, op::String, site::Int, coeff::Q = 1.0) where {Q<:Number}
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
+(oplist1::OpList, oplist2::OpList) = add(oplist1, oplist2)


function *(x::Q, y::OpList) where {Q<:Number}
    y = deepcopy(y)
    for i = 1:length(y.coeffs)
        y.coeffs[i] *= x
    end
    return y
end

function *(x::OpList, y::Q) where {Q<:Number}
    return *(y, x)
end

function /(x::OpList, y::Q) where {Q<:Number}
    return *((1/y), x)
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
### TODO: swap arguments st and oplist
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
