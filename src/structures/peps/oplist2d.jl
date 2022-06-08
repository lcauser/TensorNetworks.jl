mutable struct OpList2d
    length::Int
    ops::Vector{Vector{String}}
    sites::Vector{Vector{Int}}
    directions::Vector{Bool}
    coeffs::Vector{ComplexF64}
end

OpList2d(length::Int) = OpList2d(length, [], [], [], [])

length(oplist::OpList2d) = oplist.length

function deepcopy(oplist::OpList2d)
    oplist2 = OpList2d(copy(oplist.length))
    oplist2.sites = copy(oplist.sites)
    oplist2.ops = copy(oplist.ops)
    oplist2.coeffs = copy(oplist.coeffs)
    oplist2.directions = copy(oplist.directions)
    return oplist2
end


"""
    totensor(oplist::OpList2d, st::Sitetypes, idx::Int)

Construct the tensor from an operator in an oplist.
"""
function totensor(oplist::OpList2d, st::Sitetypes, idx::Int)
    # Fetch the relevent information
    ops = oplist.ops[idx]
    rng = length(ops)

    # Create the tensor through a tensor product
    prod = ones((1, 1))
    for site = 1:rng
        oper = ops[site]
        oper = reshape(op(st, oper), (1, st.dim, st.dim, 1))
        prod = contract(prod, oper, length(size(prod)), 1)
    end
    prod = trace(prod, 1, length(size(prod)))

    return oplist.coeffs[idx]*prod
end


"""
    siteindexs(oplist::OpList2d, site::Vector{Int}, direction::Bool=false)

Determine the indexs of operators in a list which start at a given site and
direction
"""
function siteindexs(oplist::OpList2d, sites::Vector{Int}, direction::Bool=false)
    idxs = []
    for i = 1:length(oplist.sites)
        if oplist.sites[i] == sites && oplist.directions[i] == direction
            push!(idxs, i)
        end
    end
    return idxs
end


"""
    sitetensor(st::Sitetypes, oplist::OpList2d, sites::Vector{Int},
               direction::Bool=false)

Determine the sum of all tensors starting at a site in a given direction.
"""
function sitetensor(st::Sitetypes, oplist::OpList2d, sites::Vector{Int},
                    direction::Bool=false)
    # Find all indexs of relevent operators
    idxs = siteindexs(oplist, sites, direction)

    # Determine the size of the gate
    sz = 0
    for idx in idxs
        sz = max(sz, Int(length(oplist.ops[idx])))
    end
    sz == 0 && return false

    # Loop through each site and construct the tensor
    gate = zeros(ComplexF64, [st.dim for i = 1:2*sz]...)
    for idx in idxs
        ops = oplist.ops[idx]
        prod = 1
        for i = 1:sz
            oper = i > length(ops) ? "id" : ops[i]
            prod = i == 1 ? op(st, oper) : tensorproduct(prod, op(st, oper))
        end
        gate += oplist.coeffs[idx]*prod
    end

    return gate
end


"""
    add!(oplist::OpList2d, ops::Vector{String}, sites::Vector{Int},
         direction::Bool=false, coeff::Complex{Float64} = 1)

Add an operator to the list defined by local operators at a starting site.
"""
function add!(oplist::OpList2d, ops::Vector{String}, sites::Vector{Int},
              direction::Bool=false, coeff::Number = 1.0)
    # Validate the data
    length(sites) != 2 && error("The sites must be a vector of size 2.")

    for site in sites
        (0 > site || site > oplist.length) && error("The sites must be between 1 and $(oplist.length).")
    end
    if (sites[1] + length(ops) - 1) > length(oplist) && direction == true
        error("The operators span further than the length of the lattice.")
    elseif (sites[2] + length(ops) - 1) > length(oplist) && direction == false
        error("The operators span further than the length of the lattice.")
    end

    # Add to list
    push!(oplist.ops, ops)
    push!(oplist.sites, sites)
    push!(oplist.coeffs, coeff)
    push!(oplist.directions, direction)
end

function *(x::Number, y::OpList2d)
    y = deepcopy(y)
    for i = 1:length(y.coeffs)
        y.coeffs[i] *= x
    end
    return y
end
