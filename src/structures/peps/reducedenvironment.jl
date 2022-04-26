mutable struct ReducedEnvironment
    env::Environment
    reducedtensors::Array{Array{ComplexF64, 3}, 1}
    residues::Array{Array{ComplexF64, 4}, 1}
    sites::Array{Array{Int, 1}, 1}
    rank::Int
    squared::Bool
    posdef::Bool
    hermitian::Bool
    fullenv::Array{ComplexF64}
    partialenv::Array{ComplexF64}
    partialsite::Bool
end

function ReducedEnvironment(env::Environment, dir::Bool = false; kwargs...)
    # Get key arguments
    rank::Int = get(kwargs, :rank, 2)
    squared::Bool = get(kwargs, :squared, false)
    posdef::Bool = get(kwargs, :positive_definite, false)
    hermitian::Bool = get(kwargs, :hermitian, false)

    # Create empty tensors
    tensors = [zeros(ComplexF64, 1, 1, 1), zeros(ComplexF64, 1, 1, 1)]
    residues = [zeros(ComplexF64, 1, 1, 1, 1), zeros(ComplexF64, 1, 1, 1, 1)]
    renv = ReducedEnvironment(env, tensors, residues, [[0, 0], [0, 0]], rank,
                              squared, posdef, hermitian, [], [], 0)
    reducedtensors!(renv, dir)
    return renv
end


function reducedtensors!(renv::ReducedEnvironment, dir::Bool = false)
    # Determine how to split tensors
    if !renv.env.direction
        site11 = renv.env.center
        site12 = !dir ? renv.env.centerMPS : renv.env.centerMPS - 1
        site21 = site11
        site22 = site12 + 1
        axis1 = 4
        axis2 = 1
    else
        site11 = !dir ? renv.env.centerMPS : renv.env.centerMPS - 1
        site12 = renv.env.center
        site21 = site11 + 1
        site22 = site12
        axis1 = 3
        axis2 = 2
    end

    # Update tensors
    A1, R1 = reducedtensor(renv.env.objects[end][site11, site12], axis1)
    A2, R2 = reducedtensor(renv.env.objects[end][site21, site22], axis2)
    renv.reducedtensors = [R1, R2]
    renv.residues = [A1, A2]
    renv.sites = [[site11, site12], [site21, site22]]
end

function fulltensor(A::Array{ComplexF64, 4}, R::Array{ComplexF64}, axis::Int)
    axis2 = axis == 4 || axis == 3 ? 1 : 2
    A = contract(A, R, axis, axis2)
    A = moveidx(A, 4, axis)
    return A
end

function updatetensors!(renv::ReducedEnvironment)
    axis1 = renv.env.direction ? 4 : 3
    axis2 = renv.env.direction ? 1 : 2
    A1 = fulltensor(renv.residues[1], renv.reducedtensors[1], axis1)
    A2 = fulltensor(renv.residues[2], renv.reducedtensors[2], axis2)
    renv.env.objects[end][renv.sites[1]] = A1
    renv.env.objects[end][renv.sites[2]] = A2
end

function build!(renv::ReducedEnvironment, dir::Bool = false)
    # Determine the reduced tensors
     reducedtensors!(renv, dir)

     # Determine centers and fetch relevent blocks
     centerMPS = !renv.env.direction ? renv.sites[1][2] : renv.sites[1][1]
     left = MPSblock(renv.env, centerMPS-1)
     right = MPSblock(renv.env, centerMPS+2)
     M1 = block(renv.env, renv.env.center-1)[centerMPS]
     M2 = block(renv.env, renv.env.center+1)[centerMPS]
     M3 = block(renv.env, renv.env.center-1)[centerMPS+1]
     M4 = block(renv.env, renv.env.center+1)[centerMPS+1]

     # Get env rank
     r = length(renv.env.objects)

     # Contract to get the reduced environment
     if !renv.env.direction
         ### Grow left block
         left = contract(left, M1, 1, 1)
         M = renv.squared == false ? conj(renv.residues[1]) : conj(renv.env.objects[1][renv.sites[1]...])
         left = contract(left, M, [1, 2+r], [1, 2])

         # Contract with PEPO
         if r == 3
             M = renv.env.objects[2][renv.sites[1]...]
             idxs1 = renv.squared ? [1, 4, 9] : [1, 4]
             idxs2 = renv.squared ? [1, 2, 5] : [1, 2]
             left = contract(left, M, idxs1, idxs2)
         end

         # Contract with residues
         left = contract(left, renv.residues[1], [1, 3], [1, 2])
         if r != 3 && !renv.squared
             idxs1 = [1, 3, 5]
             idxs2 = [1, 2, 3]
         elseif r != 3 && renv.squared
             idxs1 = [1, 3, 6]
             idxs2 = [1, 2, 3]
         elseif r == 3 && !renv.squared
             idxs1 = [1, 3, 5, 9]
             idxs2 = [1, 2, 3, 4]
         else
             idxs1 = [1, 3, 5, 8]
             idxs2 = [1, 2, 3, 4]
         end
         left = contract(left, M2, idxs1, idxs2)

         ### Grow right block
         right = contract(M4, right, 2+r, 2+r)
         right = contract(renv.residues[2], right, [3, 4], [1+r, 2+2*r])

         # Contract with PEPO & Bra
         if r == 3
             M = renv.env.objects[2][renv.sites[2]...]
             right = contract(M, right, [3, 4], [5, 8])
             M = !renv.squared ? conj(renv.residues[2]) : conj(renv.env.objects[1][renv.sites[2]...])
             idxs1 = renv.squared ? [3, 4, 5] : [3, 4]
             idxs2 = renv.squared ? [8, 10, 3] : [8, 10]
             right = contract(M, right, idxs1, idxs2)
             idxs = !renv.squared ? [2, 4, 8, 10] : [2, 4, 7, 9]
             right = contract(M3, right, [2, 3, 4, 5], idxs)
         else
             M = !renv.squared ? conj(renv.residues[2]) : conj(renv.env.objects[1][renv.sites[2]...])
             right = contract(M, right, [3, 4], [4, 6])
             idxs = !renv.squared ? [2, 4, 6] : [2, 5, 7]
             right = contract(M3, right, [2, 3, 4], idxs)
         end
     else
         ### Grow left block
         left = contract(left, M1, 1, 1)
         M = renv.squared == false ? conj(renv.residues[1]) : conj(renv.env.objects[1][renv.sites[1]...])
         left = contract(left, M, [1, 2+r], [2, 1])

         # Contract with PEPO
         if r == 3
             M = renv.env.objects[2][renv.sites[1]...]
             idxs1 = renv.squared ? [1, 4, 9] : [1, 4]
             idxs2 = renv.squared ? [2, 1, 5] : [2, 1]
             left = contract(left, M, idxs1, idxs2)
         end

         # Contract with residues
         left = contract(left, renv.residues[1], [1, 3], [2, 1])
         if r != 3 && !renv.squared
             idxs1 = [1, 4, 6]
             idxs2 = [1, 2, 3]
         elseif r != 3 && renv.squared
             idxs1 = [1, 4, 7]
             idxs2 = [1, 2, 3]
         elseif r == 3 && !renv.squared
             idxs1 = [1, 4, 6, 10]
             idxs2 = [1, 2, 3, 4]
         else
             idxs1 = [1, 4, 6, 9]
             idxs2 = [1, 2, 3, 4]
         end
         left = contract(left, M2, idxs1, idxs2)

         ### Grow right block
         right = contract(M4, right, 2+r, 2+r)
         right = contract(renv.residues[2], right, [4, 3], [1+r, 2+2*r])

         # Contract with PEPO & Bra
         if r == 3
             M = renv.env.objects[2][renv.sites[2]...]
             right = contract(M, right, [4, 3], [5, 8])
             M = !renv.squared ? conj(renv.residues[2]) : conj(renv.env.objects[1][renv.sites[2]...])
             idxs1 = renv.squared ? [4, 3, 5] : [4, 3]
             idxs2 = renv.squared ? [8, 10, 3] : [8, 10]
             right = contract(M, right, idxs1, idxs2)
             idxs = !renv.squared ? [1, 3, 7, 10] : [1, 3, 6, 9]
             right = contract(M3, right, [2, 3, 4, 5], idxs)
         else
             M = !renv.squared ? conj(renv.residues[2]) : conj(renv.env.objects[1][renv.sites[2]...])
             right = contract(M, right, [4, 3], [4, 6])
             idxs = !renv.squared ? [1, 3, 6] : [1, 4, 7]
             right = contract(M3, right, [2, 3, 4], idxs)
         end
     end

     ### Contract the left and right together
     if r != 3 && !renv.squared
         idxs = [1, 4]
     elseif r != 3 && renv.squared
         idxs = [1, 2, 5]
     elseif r == 3 && !renv.squared
         idxs = [1, 3, 7]
     else
         idxs = [1, 2, 3, 6]
     end
     prod = contract(left, right, idxs, idxs)

     # If necersary, make hermitian
     if !renv.squared && renv.hermitian
         # Find the closest possible hermitian approximant
         dims = size(left)
         if r == 3
             prod = moveidx(prod, 4, 3)
             prod = moveidx(prod, 8, 7)
             idxs1 = [1, 2, 5, 6]
             idxs2 = [1, 2, 3, 4]
         else
             idxs1 = [1, 3]
             idxs2 = [1, 2]
         end
         prod, cmb1 = combineidxs(prod, idxs1)
         prod, cmb2 = combineidxs(prod, idxs2)
         prod = 0.5*(prod + conj(transpose(prod)))

         if renv.posdef
             # Make it positive definite
             F = eigen(prod)
             vals = diagm([real(val) < 0 ? 0.0 : val for val in F.values])
             prod = contract(F.vectors, vals, 2, 1)
             prod = contract(prod, conj(transpose(F.vectors)), 2, 1)
         end

         # Reshape back to the original form
         prod = uncombineidxs(prod, cmb2)
         prod = uncombineidxs(prod, cmb1)

         # Move indexs back
         if r == 3
             prod = moveidx(prod, 4, 3)
             prod = moveidx(prod, 8, 7)
         end
     end
     renv.fullenv = prod

end


function partialcontract!(renv::ReducedEnvironment, site::Bool)
    # Fetch the reduced tensor
    red = renv.reducedtensors[site ? 2 : 1]

    # Contract with the environment
    r = length(renv.env.objects)
    if !renv.squared
        if r == 3
            if !site
                prod = contract(red, renv.fullenv, [1, 3], [1,2])
                prod = contract(conj(red), prod, [1, 3], [3, 2])
            else
                prod = contract(renv.fullenv, conj(red), [5, 6], [2, 3])
                prod = contract(prod, red, [5, 6], [3, 2])
            end
        else
            if !site
                prod = contract(red, renv.fullenv, 1, 2)
                prod = contract(conj(red), prod, [1, 3], [3, 2])
            else
                prod = contract(renv.fullenv, conj(red), 3, 2)
                prod = contract(prod, red, [3, 5], [2, 3])
            end
        end
    else
        if r == 3
            if !site
                prod = contract(red, renv.fullenv, [1, 3], [2, 1])
            else
                prod = contract(renv.fullenv, red, [3, 4], [3, 2])
                prod = moveidx(prod, 1, 3)
            end
        else
            if !site
                prod = contract(red, renv.fullenv, [1, 3], [2, 1])
                prod = moveidx(prod, 3, 2)
            else
                prod = contract(renv.fullenv, red, [4, 3], [2, 3])
                prod = moveidx(prod, 1, 3)
            end
        end
    end

    renv.partialsite = site
    renv.partialenv = prod
end


function product(renv::ReducedEnvironment, A::Array{ComplexF64, 3})
    # Contract with the environment
    r = length(renv.env.objects)
    if !renv.squared
        if r == 3
            if !renv.partialsite
                prod = contract(renv.partialenv, A, [2, 5, 6], [1, 3, 2])
            else
                prod = contract(renv.partialenv, A, [3, 4, 6], [3, 1, 2])
            end
        else
            prod = contract(renv.partialenv, A, [2, 4], [1, 2])
        end
    else
        prod = contract(renv.partialenv, A, [1, 2, 3], [1, 2, 3])[1]
        prod *= conj(renv.partialenv)
    end

    return prod
end

function fullcontract(renv::ReducedEnvironment)
    # Contract with the environment
    r = length(renv.env.objects)
    A = renv.reducedtensors[!renv.partialsite ? 2 : 1]
    if !renv.squared
        if r == 3
            if !renv.partialsite
                prod = contract(renv.partialenv, conj(A), [1, 3, 4], [1, 2, 3])
                prod = contract(prod, A, [1, 2, 3], [1, 3, 2])[1]
            else
                prod = contract(renv.partialenv, conj(A), [1, 2, 5], [1, 3, 2])
                prod = contract(prod, A, [1, 2, 3], [3, 1, 2])[1]
            end
        else
            prod = contract(renv.partialenv, conj(A), [1, 3], [1, 2])
            prod = contract(prod, A, [1, 2, 3], [1, 2, 3])[1]
        end
    else
        prod = contract(renv.partialenv, A, [1, 2, 3], [1, 2, 3])[1]
        prod *= conj(prod)
    end

    return prod
end


function increasedim!(renv::ReducedEnvironment, dim)
    dims1 = size(renv.reducedtensors[1])
    dims2 = size(renv.reducedtensors[2])

    F1 = zeros(ComplexF64, (dims1[1], dim, dims1[3])) + 1e-4*randn(Float64, dims1[1], dim, dims1[3])
    F2 = zeros(ComplexF64, (dim, dims2[2], dims2[3])) + 1e-4*randn(Float64, dim, dims2[2], dims2[3])
    F1[:, 1:dims1[2], :] = renv.reducedtensors[1]
    F2[1:dims2[1], :, :] = renv.reducedtensors[2]

    renv.reducedtensors = [F1, F2]
end
