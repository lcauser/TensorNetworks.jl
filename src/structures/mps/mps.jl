"""
    MPS(dim::Int, length::Int)

Create a MPS with physical dimension dim.
"""
function MPS(dim::Int, length::Int)
    tensors = []
    for i = 1:length
        push!(tensors, zeros(ComplexF64, (1, dim, 1)))
    end
    return GMPS(1, dim, tensors, 0)
end


"""
    ismps(psi::GMPS)

Check if a GMPS is of rank 1.
"""
function ismps(psi::GMPS)
    rank(psi) == 1 && return true
    return false
end


### Create MPS
"""
    randomMPS(dim::Int, length::Int, bonddim::Int)

Create a MPS with random entries.
"""
function randomMPS(dim::Int, length::Int, bonddim::Int)
    return randomGMPS(1, dim, length, bonddim)
end


"""
    productMPS(sites::Int, A::Array{Complex{Float64}, 3})
    productMPS(sites::Int, A)

Create a product MPS of some fixed tensor.
A can be a vector for product state entries, or larger dimensional tensor which
is truncated at the edge sites.
"""
function productMPS(sites::Int, A::Array{Complex{Float64}, 3})
    tensors = Array{Complex{Float64}, 3}[]
    push!(tensors, copy(A[1:1, :, :]))
    for i = 2:sites-1
        push!(tensors, copy(A))
    end
    push!(tensors, copy(A[:, :, end:end]))
    return GMPS(1, size(A)[2], tensors, 0)
end

function productMPS(sites::Int, A)
    if length(size(A)) == 1
        A = reshape(A, (1, size(A)[1], 1))
    end
    A = convert(Array{Complex{Float64}, 3}, A)
    return productMPS(sites, A)
end

"""
    productMPS(st::Sitetypes, names::Vector{String})

Create a product state from the names of local states on a sitetype.
"""
function productMPS(st::Sitetypes, names::Vector{String})
    tensors = []
    for i = 1:length(names)
        A = convert(Array{Complex{Float64}, 3},
                    reshape(state(st, names[i]), (1, st.dim, 1)))
        push!(tensors, A)
    end
    return GMPS(1, st.dim, tensors, 0)
end


### Calculate inner products of each operator in the operator list with respect
### to two MPSs
"""
    inner(st::Sitetypes, psi::GMPS, oplist::OpList, phi::GMPS)

Efficently calculate the expectations of a list of operators between two
MPS.
"""
function inner(st::Sitetypes, psi::GMPS, oplist::OpList, phi::GMPS)
    # Create a projection on to the MPSs
    projV = ProjMPS(psi, phi; rank=1, squared=false)

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
    applyop!(st::Sitetypes, psi::GMPS, ops, sites, coeff::Number=1)

Apply local operators to an MPS.
"""
function applyop!(st::Sitetypes, psi::GMPS, ops, sites, coeff::Q=1.0) where Q<:Number
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
