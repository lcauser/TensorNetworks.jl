mutable struct ProjMPS <: AbstractProjMPS
    objects::Vector{GMPS}
    blocks::Vector{Array{Complex{Float64}}}
    squared::Bool
    rank::Int
    center::Int
    coeff::Real
end


"""
    projMPS(args::GMPS...; kwargs...)

Construct a projection on an inner product.
"""
function ProjMPS(args::GMPS...; kwargs...)
    # Get key arguments
    squared::Bool = get(kwargs, :squared, false)
    center::Int = get(kwargs, :center, 1)
    coeff::Number = get(kwargs, :coeff, 1.0)
    projrank::Int = get(kwargs, :rank, 1)

    # Check the rank
    squared == true && projrank == 1 && error("Squared and rank one are incomptible.")

    # Check to make sure all arguments have the same properties
    dims = [dim(arg)==dim(args[1]) for arg in args]
    sum(dims) != length(args) && error("GMPS must share the same physical dim.")
    lengths = [length(arg)==length(args[1]) for arg in args]
    sum(lengths) != length(args) && error("GMPS must share the same length.")

    # Make sure the arguments are the right order
    ranks = [rank(arg) for arg in args]
    length(args) < 2 && error("The projection must have a braket structure")
    (ranks[1] != 1 || ranks[end] != 1) && error("The projection must have a braket structure")

    # Construct the blocks
    blocks = [edgeblock(length(args)) for i = 1:length(args[1])]
    projV = ProjMPS(collect(args), blocks, squared, projrank, 0, coeff)
    movecenter!(projV, center)
    return projV
end


"""
    buildleft!(projV::ProjMPS, idx::Int)

Expand the left block using the previous.
"""
function buildleft!(projV::ProjMPS, idx::Int)
    # Fetch the block to the left and the tensors
    left = block(projV, idx-1)
    A1 = conj(projV.objects[1][idx])
    A2 = projV.objects[end][idx]

    # Contract the block with the tensors
    prod = contract(left, A1, 1, 1)
    for i = 1:length(projV.objects)-2
        M = projV.objects[1+i][idx]
        prod = contract(prod, M, [1, length(size(prod))-1], [1, 2])
    end
    prod = contract(prod, A2, [1, length(size(prod))-1], [1, 2])

    # Save the block
    projV[idx] = prod
end


"""
    buildright!(projV::ProjMPS, idx::Int)

Expand the right block using the previous.
"""
function buildright!(projV::ProjMPS, idx::Int)
    # Fetch the block to the right and the tensors
    #println("---")
    #println(idx)
    right = block(projV, idx+1)
    A1 = conj(projV.objects[1][idx])
    A2 = projV.objects[end][idx]
    #println(size(right))
    #println(size(A1))
    #println(size(A2))

    # Contract the block with the tensors
    prod = contract(A2, right, 3, length(size(right)))
    for i = 1:length(projV.objects)-2
        M = projV.objects[length(projV.objects)-i][idx]
        prod = contract(M, prod, [3, 4], [2, length(size(prod))])
    end
    prod = contract(A1, prod, [2, 3], [2, length(size(prod))])

    # Save the block
    projV[idx] = prod
end


"""
    product(projV::ProjMPS, A, direction::Bool = 0, nsites::Int = 2)

Determine the product of the projection on proposed sites.
"""
function product(projV::ProjMPS, A, direction::Bool = 0, nsites::Int = 2)
    # There is two routes for projection; the bra and ket are the same, and
    # hence the projection is a rank-2 matrix on the site vectors. Otherwise,
    # the bra and ket are different...
    if projV.rank == 2 && projV.squared == false
        # Determine the site
        site = direction ? projV.center - nsites + 1 : projV.center

        # Get the blocks
        left = block(projV, site - 1)
        right = block(projV, site + nsites)

        # Move vector D to beginning
        prod = moveidx(left, length(size(left)), 2)

        # Contract for nsites over all middle MPO objects
        for i = 1:nsites
            for j = 1:length(projV.objects)-2
                if j == 1
                    prod = contract(prod, projV.objects[j+1][site-1+i], 1+2*i, 1)
                else
                    prod = contract(prod, projV.objects[j+1][site-1+i],
                                    [1+2*i, length(size(prod))-1], [1, 2])
                end
            end
            prod = moveidx(prod, length(size(prod))-1, 2*i+2)
        end

        # Contract the given site
        prod = contract(prod, A, [2*k for k = 1:nsites+1], [k for k=1:nsites+1])
        prod = contract(prod, right, [k for k=2+nsites:length(size(prod))],
                        [k for k=2:length(size(right))])
    else
        # Fetch the projection
        prod = project(projV, A, direction, nsites)

        # Contract with tensor
        prod2 = projV.squared == true ? conj(prod) : 1
        prod = contract(prod, A, [k for k=1:length(size(prod))], [k for k=1:length(size(prod))])[1]
        prod *= prod2
    end
    return prod * projV.coeff
end


"""
    project(projV::ProjMPS, A, direction::Bool = 0, nsites::Int = 2)

Determine the projection onto the sites at the center in the given direciton.
"""
function project(projV::ProjMPS, A, direction::Bool = 0, nsites::Int = 2)
    # Determine the site
    site = direction ? projV.center - nsites + 1 : projV.center

    # Get the blocks
    left = block(projV, site - 1)
    right = block(projV, site + nsites)

    #if projV.rank == 2
    #    error("Projection not supported for rank-2 objects.")
    #end

    # Move vector for D to beginning
    prod = moveidx(left, length(size(left)), 1)

    # Loop through nsites
    for i = 1:nsites
        # Contract with first vector
        prod = contract(prod, conj(projV.objects[1][site-1+i]), 1+i, 1)

        # Contract with MPO tensors
        for j = 1:length(projV.objects)-2
            prod = contract(prod, projV.objects[1+j][site-1+i],
                            [1+i, length(size(prod))-1], [1, 2])
        end
        prod = moveidx(prod, length(size(prod))-1, 1+i)
    end

    # Contract with right
    prod = contract(prod, right, [k for k=2+nsites:length(size(prod))],
                    [k for k=1:length(size(right))-1])
    return prod
end

"""
    calculate(projV::ProjMPS)

Calculate the fully contracted projection.
"""
function calculate(projV::ProjMPS)
    # Determine the site
    site = projV.center

    # Get the blocks
    left = block(projV, site - 1)
    right = block(projV, site + 1)
    A1 = conj(projV.objects[1][site])
    A2 = projV.objects[end][site]

    # Contract the block with the tensors
    prod = contract(left, A1, 1, 1)
    for i = 1:length(projV.objects)-2
        M = projV.objects[1+i][site]
        prod = contract(prod, M, [1, length(size(prod))-1], [1, 2])
    end
    prod = contract(prod, A2, [1, length(size(prod))-1], [1, 2])
    #println("----")
    #println(site)
    #println(size(prod))
    #println(size(right))
    prod = contract(prod, right, [i for i=1:length(size(prod))], [i for i=1:length(size(prod))])

    return projV.coeff*prod[1]
end
