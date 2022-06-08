"""
    iGMPS(rank::Int, dim::Int, tensors::Vector{Array{Complex{Float64}}},
          singulars::Vector{Array{Float64}}, norms::Vector{Float64})
    iGMPS(rank::Int, dim::Int, size::Int)

Create an infinite generalized matrix product state.
"""
mutable struct iGMPS
    rank::Int
    dim::Int
    tensors::Vector{Array{ComplexF64}}
    singulars::Vector{Array{Float64}}
    norms::Vector{Float64}
    squared::Bool
    lefts::Vector{Array{ComplexF64}}
    rights::Vector{Array{ComplexF64}}
end

function iGMPS(rank::Int, dim::Int, length::Int; kwargs...)
    # Create empty tensors
    tensors = []
    for i = 1:length
        push!(tensors, zeros(ComplexF64, (1, [dim for i=1:rank]..., 1)))
    end

    # Singular values
    singulars = [ones(Float64, 1) for i=1:length]
    norms = [0.0 for i = 1:length]

    # Environment
    squared::Bool = get(kwargs, :squared, true)
    return iGMPS(rank, dim, tensors, singulars, norms, squared, [], [])
end

# Properties
dim(psi::iGMPS) = psi.dim
rank(psi::iGMPS) = psi.rank
length(psi::iGMPS) = length(psi.tensors)
maxbonddim(psi::iGMPS) = maximum([size(t)[1] for t in psi.tensors])


"""
    iMPS(dim::Int, size::Int)
    iMPS(length::Int, A::Array{ComplexF64, 1})
    iMPS(length::Int, A::Vector{ComplexF64})

Construct an infinite MPS.
"""
function iMPS(dim::Int, length::Int; kwargs...)
    return iGMPS(1, dim, length; kwargs...)
end

function iMPS(length::Int, A::Array{ComplexF64, 1}; kwargs...)
    psi = iGMPS(1, size(A)[1], length; kwargs...)
    for i = 1:length
        psi.tensors[i] = reshape(deepcopy(A), 1, size(A)[1], 1)
    end
    return psi
end

#function iMPS(size::Int, A::Vector{ComplexF64})
#    return iMPS(size::Int, convert(Array{ComplexF64, 1}, A))
#end


"""
    iMPO(dim::Int, length::Int)
    iMPO(length::Int, A::Array{ComplexF64, 2})
    iMPO(length::Int, Vector{Vector{ComplexF64}})

Construct an infinite MPO.
"""
function iMPO(dim::Int, length::Int)
    return iGMPS(2, dim, length)
end


function iMPO(length::Int, A::Array{ComplexF64, 2}; kwargs...)
    psi = iGMPS(2, size(A)[1], length; kwargs...)
    for i = 1:length
        psi.tensors[i] = reshape(deepcopy(A), 1, size(A)[1], size(A)[1], 1)
    end
    return psi
end

function iMPO(length::Int, A::Vector{Vector{ComplexF64}})
    return iMPO(length::Int, convert(Array{ComplexF64, 2}, A))
end

"""
    buildleft(psi::iGMPS, left::Array{ComplexF64}, j::Int)

Build on the left block for an environment for an iMPS, for cell
position j
"""
function buildleft(psi::iGMPS, left::Array{ComplexF64}, j::Int)
    if psi.rank == 1 && psi.squared == true
        # Wavefunction
        left = contract(left, diagm(psi.singulars[j]), 1, 1)
        left = contract(left, diagm(psi.singulars[j]), 1, 1)
        left = contract(left, conj(psi.tensors[j]), 1, 1)
        left = contract(left, psi.tensors[j], [1, 2], [1, 2])
    elseif psi.rank == 1 && psi.squared == false
        # Probability vector
        left = contract(left, diagm(psi.singulars[j]), 1, 1)
        left = contract(left, psi.tensors[j], 1, 1)
        left = contract(left, ones(ComplexF64, dim(psi)), 1, 1)
    elseif psi.rank == 2
        # MPO trace
        left = contract(left, diagm(psi.singulars[j]), 1, 1)
        left = contract(left, trace(psi.tensors[j], 2, 3), 1, 1)
    end
    return left
end


"""
    buildright(psi::iGMPS, right::Array{ComplexF64}, j::Int)

Build on the right block for an environment for an iMPS, for cell
position j.
"""
function buildright(psi::iGMPS, right::Array{ComplexF64}, j::Int)
    if psi.rank == 1 && psi.squared == true
        # Wavefunction
        right = contract(psi.tensors[j], right, 3, 2)
        right = contract(conj(psi.tensors[j]), right, [2, 3], [2, 3])
        right = contract(diagm(psi.singulars[j]), right, 2, 2)
        right = contract(diagm(psi.singulars[j]), right, 2, 2)
    elseif psi.rank == 1 && psi.squared == false
        # Probability vector
        right = contract(psi.tensors[j], right, 3, 1)
        right = contract(ones(ComplexF64, dim(psi)), right, 1, 2)
        right = contract(diagm(psi.singulars[j]), right, 2, 1)
    elseif psi.rank == 2
        # MPO trace
        right = contract(trace(psi.tensors[j], 2, 3), right, 2, 1)
        right = contract(diagm(psi.singulars[j]), right, 2, 1)
    end
    return right
end


"""
    randomenv(psi::iGMPS)

Create a random boundary for an iMPS.
"""
function randomenv(psi::iGMPS)
    if psi.rank == 1 && psi.squared == true
        # Wavefunction
        left = rand(ComplexF64, size(psi.tensors[1])[1], size(psi.tensors[1])[1])
        right = rand(ComplexF64, size(psi.tensors[end])[end], size(psi.tensors[end])[end])
    else
        # Probability vector & MPO
        left = rand(ComplexF64, size(psi.tensors[1])[1])
        right = rand(ComplexF64, size(psi.tensors[end])[end])
    end
    return left, right
end


"""
    function build!(psi::iGMPS; kwargs...)

Build the environment for an iMPS to perform calculations. Key arguments:
    - tol::Float64 : Convergence tolerance for the environment. Default is 1e-10.
    - maxiter::Int : Maximum number of iterations to build environment by.
"""
function build!(psi::iGMPS; kwargs...)
    # Get convergence tolerance
    tol::Float64 = get(kwargs, :tol, 1e-10)
    maxiter::Int = get(kwargs, :maxiter, 1000)

    # Vector to store environment
    lefts = []
    rights = []

    # Make random initial environments
    left, right = randomenv(psi)
    norm_cum = log(norm(left)) + log(norm(right))
    left /= norm(left)
    right /= norm(right)
    idxs = (rank(psi) == 1 && psi.squared == true) ? [1, 2] : 1
    norm_prev = real(norm_cum + log(contract(left, right, idxs, idxs)[1]))

    # Iteratively build left and right blocks and check for convergence
    for i = 1:maxiter
        left_prev = deepcopy(left)
        right_prev = deepcopy(right)
        for j = 1:length(psi)
            left = buildleft(psi, left, j)
            right = buildright(psi, right, length(psi)+1-j)
        end
        left /= norm(left)
        right /= norm(right)

        #println("---")
        #println(sum(abs.(left .- left_prev)))
        #println(sum(abs.(right .- right_prev)))
        converge = sum(abs.(left .- left_prev)) < tol
        converge = sum(abs.(right .- right_prev)) < tol ? converge : false
        converge && break
    end

    psi.lefts = [left]
    psi.rights = [right]

    # Build the environments to consider all sites
    for i = 1:length(psi)-1
        left = buildleft(psi, left, i)
        right = buildright(psi, right, length(psi)+1-i)
        push!(psi.lefts, left)
        push!(psi.rights, right)
    end
end


function inner(st::Sitetypes, psi::iGMPS, ops::OpList)
    vals = zeros(ComplexF64, length(ops.ops))
    for i = 1:length(ops.ops)
        vals[i] = inner(st, psi, ops.ops[i], ops.sites[i], ops.coeffs[i])
    end
    return vals
end

function inner(st::Sitetypes, psi::iGMPS, ops::Vector{String}, sites::Vector{Int}, coeff::ComplexF64)
    # Determine the range
    rng = sites[end] - sites[1] + 1

    # Loop over all periods
    val = 0.0
    for i = 1:length(psi)
        # Find relevent boundaries
        left = psi.lefts[i]
        right = psi.rights[length(psi) - ((rng + i) % length(psi))]

        # Contract with and without observables
        prod_obs = left
        prod_norm = left
        for j = 1:rng
            # Determine site and fetch relevent tensors
            site = (j + i - 2) % length(psi) + 1
            S = diagm(psi.singulars[site])
            A = psi.tensors[site]
            M = j in sites ? ops[findfirst([j == s for s in sites])] : "id"
            M = op(st, M)

            # Perform the contractions
            if psi.rank == 1 && psi.squared == true
                # Contract to find observable
                prod_obs = contract(prod_obs, S, 1, 1)
                prod_obs = contract(prod_obs, S, 1, 1)
                prod_obs = contract(prod_obs, conj(A), 1, 1)
                prod_obs = contract(prod_obs, M, 2, 1)
                prod_obs = contract(prod_obs, A, [1, 3], [1, 2])

                # Contract to find norm
                prod_norm = contract(prod_norm, S, 1, 1)
                prod_norm = contract(prod_norm, S, 1, 1)
                prod_norm = contract(prod_norm, conj(A), 1, 1)
                prod_norm = contract(prod_norm, A, [1, 2], [1, 2])
            elseif psi.rank == 1 && psi.squared == false
                # Contract to find observable
                prod_obs = contract(prod_obs, S, 1, 1)
                prod_obs = contract(prod_obs, A, 1, 1)
                prod_obs = contract(prod_obs, M, 1, 1)
                prod_obs = contract(prod_obs, ones(dim(psi)), 2, 1)

                # Contract to find norm
                prod_norm = contract(prod_norm, S, 1, 1)
                prod_norm = contract(prod_norm, A, 1, 1)
                prod_norm = contract(prod_norm, ones(dim(psi)), 1, 1)
            else
                # Contract to find observable
                prod_obs = contract(prod_obs, S, 1, 1)
                prod_obs = contract(prod_obs, A, 1, 1)
                prod_obs = contract(prod_obs, M, 2, 1)
                prod_obs = trace(prod_obs, 1, 3)

                # Contract to find norm
                prod_norm = contract(prod_norm, S, 1, 1)
                prod_norm = contract(prod_norm, trace(A, 2, 3), 1, 1)
            end
        end

        # Contract with right
        idxs = (psi.rank == 1 && psi.squared == true) ? [1, 2] : 1
        prod_obs = contract(prod_obs, right, idxs, idxs)[1]
        prod_norm = contract(prod_norm, right, idxs, idxs)[1]
        val += prod_obs / prod_norm
    end
    return coeff * val / length(psi)
end

function norm(psi::iGMPS)
    prod = psi.lefts[1]
    norm = 0
    for i = 1:length(psi)
        prod = buildleft(psi, prod, i)
        norm += (psi.rank == 1 && psi.squared == true) ? 2*psi.norms[i] : psi.norms[i]
    end
    idxs = (psi.rank == 1 && psi.squared == true) ? [1, 2] : 1
    prod = contract(prod, psi.rights[1], idxs, idxs)
    norm += log(prod[1])
    norm -= log(contract(psi.lefts[1], psi.rights[1], idxs, idxs)[1])
    return norm / length(psi)
end
