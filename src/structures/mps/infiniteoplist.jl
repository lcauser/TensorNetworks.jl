#=
    Create an operator list for an infinite system.
    Terms are added which are translationally invariant.
=#

mutable struct InfiniteOpList{T<:Number}
    ops::Vector{Vector{String}}
    sites::Vector{Vector{Int}}
    coeffs::Vector{T}
end

"""
    InfiniteOpList()

Create a list of operators acting on an infinite lattice.
"""
InfiniteOpList() = InfiniteOpList(Vector{String}[], Vector{Int}[], ComplexF64[])

length(oplist::InfiniteOpList) = oplist.length


### Add to the list
"""
    add!(oplist::InfiniteOpList, ops::Vector{String}, sites::Vector{Int},
         coeff::Complex{Float64} = 1)
    add!(oplist::InfiniteOpList, op::String, site::Int,
         coeff::Complex{Float64} = 1)

Add an operator to the list defined by local operators at given sites.
Sites should be ordeded from 1 to the string length.
"""
function add!(oplist::InfiniteOpList, ops::Vector{String}, sites::Vector{Int},
              coeff::Q = 1.0) where {Q<:Number}
    # Validate the data
    length(ops) != length(sites) && error("The lists must be the same length.")

    # Order the operators and sites
    perms = sortperm(sites)
    sites = sites[perms]
    ops = ops[perms]

    # Make the first site 1
    sites .+= 1 - sites[1]

    # Add to list
    push!(oplist.ops, ops)
    push!(oplist.sites, sites)
    push!(oplist.coeffs, coeff)
end

function add!(oplist::InfiniteOpList, op::String, site::Int,
              coeff::Q = 1.0) where {Q<:Number}
    add!(oplist, [op], [site], coeff)
end


"""
    add(oplist1::OpList, oplist2::OpList)

Join two oplists.
"""
function add(oplist1::InfiniteOpList, oplist2::InfiniteOpList)
    oplist = InfiniteOpList()
    oplist.sites = copy(oplist1.sites)
    append!(oplist.sites, oplist2.sites)
    oplist.ops = copy(oplist1.ops)
    append!(oplist.ops, oplist2.ops)
    oplist.coeffs = copy(oplist1.coeffs)
    append!(oplist.coeffs, oplist2.coeffs)
    return oplist
end
+(oplist1::InfiniteOpList, oplist2::InfiniteOpList) = add(oplist1, oplist2)


function *(x::Q, y::InfiniteOpList) where {Q<:Number}
    y = deepcopy(y)
    for i = 1:length(y.coeffs)
        y.coeffs[i] *= x
    end
    return y
end

function *(x::InfiniteOpList, y::Q) where {Q<:Number}
    return *(y, x)
end

function /(x::InfiniteOpList, y::Q) where {Q<:Number}
    return *((1/y), x)
end

### Determine properties about the operator list
"""
    siteRange(oplist::InfiniteOpList)

Determine the interaction range within an infinite operator list.
"""
function siterange(oplist::InfiniteOpList)
    rng = 1
    for sites in oplist.sites
        rng = max(rng, sites[end]-sites[1]+1)
    end

    return rng
end


### Compiling the operator list as a tensor
"""
    totensor(st::Sitetypes, oplist::InfiniteOpList)

Output the operator list as a tensor. Used for, e.g., writing down the local
term in an infinite Hamiltonian as a full rank tensor.
"""
function totensor(st::Sitetypes, oplist::InfiniteOpList)
    # Find the length of the operator list 
    rng = siterange(oplist)

    # Create an empty tensor 
    ten = zeros(ComplexF64, [st.dim for _ = 1:2*rng]...)

    # Add all the terms to the operator 
    for i in eachindex(oplist.ops)
        rng_i = oplist.sites[i][end] - oplist.sites[i][begin] + 1
        coeff = rng - rng_i + 1 

        # Determine the operator 
        oper = oplist.coeffs[i] * ones(ComplexF64, )
        for j = 1:rng_i
            term = if j in oplist.sites[i]
                oplist.ops[i][findfirst(oplist.sites[i] .== j)]
            else
                "id"
            end
            oper = tensorproduct(oper, op(st, term))
        end

        # Add the term with translational invariance 
        for j = 1:rng - rng_i + 1
            oper_full = (1 / coeff) * ones(ComplexF64, )
            for _ = 1:j-1
                oper_full = tensorproduct(oper_full, op(st, "id"))
            end
            oper_full = tensorproduct(oper_full, oper)
            for _ = j+rng_i:rng
                oper_full = tensorproduct(oper_full, op(st, "id"))
            end
            ten += oper_full
        end
    end

    return ten
end


### Compiling the infinite operator list into finite size operator lists
### and MPOs
"""
    OpList(st::Sitetypes, oplist::InfiniteOpList)

Returns a local term in an infite 1D Hamiltonian as an MPO.
"""
function OpList(oplist::InfiniteOpList)
    # Find the length of the operator list 
    rng = siterange(oplist)

    # Create an empty finite operator list  
    finite_oplist = OpList(rng)
    for i in eachindex(oplist.ops)
        # Determine the range and a coefficient
        rng_i = oplist.sites[i][end] - oplist.sites[i][begin] + 1
        coeff = rng - rng_i + 1 

        # Add to the operator list translationally invariant 
        for j = 1:rng - rng_i + 1
            add!(finite_oplist, oplist.ops[i], oplist.sites[i] .+ (j - 1),
                 oplist.coeffs[i] / coeff)
        end
    end

    return finite_oplist
end


"""
    MPO(st::Sitetypes, oplist::InfiniteOpList)

Returns a local term in an infite 1D Hamiltonian as an MPO.
"""
function MPO(st::Sitetypes, oplist::InfiniteOpList)
    return MPO(st, OpList(oplist))
end


"""
    OpList(st::Sitetypes, oplist::InfiniteOpList, N::Int)

Returns a finite translationally invariant operator list of size N 
from an infinite operator list.
"""
function OpList(oplist::InfiniteOpList, N::Int)
    # Create an empty finite operator list  
    finite_oplist = OpList(N)
    for i in eachindex(oplist.ops)
        # Determine the range
        rng_i = oplist.sites[i][end] - oplist.sites[i][begin] + 1 

        # Add to the operator list translationally invariant 
        for j = 1:N - rng_i + 1
            add!(finite_oplist, oplist.ops[i], oplist.sites[i] .+ (j - 1),
                 oplist.coeffs[i])
        end
    end

    return finite_oplist
end

"""
    MPO(st::Sitetypes, oplist::InfiniteOpList, N::Int)

Returns a finite MPO of size N from an infinite operator list.
"""
function MPO(st::Sitetypes, oplist::InfiniteOpList, N::Int)
    return MPO(st, OpList(oplist), N)
end
