mutable struct ReducedEnvironment
    env::Environment
    reducedtensors::Array{Array{ComplexF64, 3}, 2}
    residues::Array{Array{ComplexF64, 4}, 2}
    sites::Array{Array{Int, 2}, 2}
    rank::Int
    squared::Bool
    posdef::Bool
    hermitian::Bool
end

function ReducedEnvironment(env::Environment, dir::Bool = false; kwargs...)
    # Get key arguments
    rank::Int = get(kwargs, :rank, 2)
    squared::Bool = get(kwargs, :squared, false)
    posdef::Bool = get(kwargs, :positive_definite, false)
    hermitian::Bool = get(kwargs, :hermitian, true)

    # Create empty tensors
    tensors = [zeros(ComplexF64, 1, 1, 1), zeros(ComplexF64, 1, 1, 1)]
    residues = [zeros(ComplexF64, 1, 1, 1, 1), zeros(ComplexF64, 1, 1, 1, 1)]
    renv = ReducedEnvironment(env, tensors, residues, [[0, 0], [0, 0]], rank,
                              squared, posdef, hermitian)
    reducedtensors!(renv, dir)
    return renv
end


function reducedtensors!(renv::ReducedEnvironment, dir::Bool = false)
    # Determine how to split tensors
    if !env.direction
        site11 = env.center
        site12 = !dir ? renv.env.centerMPS : renv.env.centerMPS - 1
        site21 = site11
        site22 = site12 + 1
        axis1 = 4
        axis2 = 1
    else
        site11 = !dir ? renv.env.centerMPS : renv.env.centerMPS - 1
        site12 = renv.env.center
        site21 = !dir ? site11 : site11 - 1
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
     right = MPSblock(env, centerMPS+2)
     M1 = block(env, center-1)[centerMPS]
     M2 = block(env, center+1)[centerMPS]
     M3 = block(env, center-1)[centerMPS+1]
     M4 = block(env, center+1)[centerMPS+1]

     # Get env rank
     r = length(renv.env.objects)

     # Contract to get the reduced environment
     if !dir
         # Grow left block
         left = contract(left, M1, 1, 1)
         left = contract(left, conj(renv.residues[1]), [1, 2+r], [1, 2])
         for i = 1:r-2
             # Fetch blocks
             M = renv.env.objects[1+i][renv.sites[1]]

             # Determine idxs
             idxs1 = [1, 2+r-i]
             idxs2 = [1, 2]
             if i != 1 && i != r-2
                 push!(idxs1, length(size(left)))
                 push!(idxs2, 5)
             end

             # Contract
             left = contract(left, M, idxs1, idxs2)
         end
         left = contract(left, renv.residues[2], [1, 3], [1, 2])
         left = contract(left, M2, [1, 3, 5, [5+3*i for i=1:r-2]...], [1, 2, 3, [3+i for i=1:r-2]...])

         # Grow right block
     else

     end


end
