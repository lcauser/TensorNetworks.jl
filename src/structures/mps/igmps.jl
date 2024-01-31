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


### Building environments
"""
    buildleft(phi::iGMPS, psi::iGMPS, left::Array{ComplexF64}, j::Int)
    buildleft(psi::iGMPS, left::Array{ComplexF64}, j::Int)

Build on the left block for an environment for an iMPS, for cell
position j
"""
function buildleft(phi::iGMPS, psi::iGMPS, left::Array{ComplexF64}, j::Int)
    if psi.rank == 1 && psi.squared == true
        return buildleft_phi_psi(phi, psi, left, j)
    elseif psi.rank == 1 && psi.squared == false
        return buildleft_flat_psi(psi, left, j)
    elseif psi.rank == 2
        return buildleft_rho(psi, left, j)
    end
end
buildleft(psi::iGMPS, left::Array{ComplexF64}, j::Int) = buildleft(psi, psi, left, j)

"""
    buildright(phi::iGMPS, psi::iGMPS, right::Array{ComplexF64}, j::Int)
    buildright(psi::iGMPS, right::Array{ComplexF64}, j::Int)

Build on the right block for an environment for an iMPS, for cell
position j.
"""
function buildright(phi::iGMPS, psi::iGMPS, right::Array{ComplexF64}, j::Int)
    if psi.rank == 1 && psi.squared == true
        return buildright_phi_psi(phi, psi, right, j)
    elseif psi.rank == 1 && psi.squared == false
        return buildright_flat_psi(psi, right, j)
    elseif psi.rank == 2
        return buildright_rho(psi, right, j)
    end
end
buildright(psi::iGMPS, left::Array{ComplexF64}, j::Int) = buildright(psi, psi, left, j)


"""
    randomenv(psi::iGMPS)
    randomenv(phi::iGMPS, psi::iGMPS)

Create a random boundary for an iMPS.
"""
randomenv(phi::iGMPS, psi::iGMPS) = randomenv_phi_psi(phi, psi)
function randomenv(psi::iGMPS)
    if psi.rank == 1 && psi.squared == false
        return randomenv_flat_psi(psi)
    elseif psi.rank == 1 && psi.squared == true
        return randomenv_psi(psi)
    else
        return randomenv_rho(psi)
    end
end


"""
    function build(phi::iGMPS, psi::iGMPS; kwargs...)
    function build!(psi::iGMPS; kwargs...)

Build the environment for an iMPS to perform calculations. Key arguments:
    - tol::Float64 : Convergence tolerance for the environment. Default is 1e-10.
    - maxiter::Int : Maximum number of iterations to build environment by.
"""
function build(psis::iGMPS...; kwargs...)
    # Checks on psi 
    (length(psis) == 0 || length(psis) > 2) && error("Only one or two GMPS supported currently.")

    # Get convergence tolerance
    tol::Float64 = get(kwargs, :tol, 1e-10)
    maxiter::Int = get(kwargs, :maxiter, 1000)

    # Make random initial environments
    left, right = randomenv(psis...)
    #norm_cum = log(norm(left)) + log(norm(right))
    left /= norm(left)
    right /= norm(right)
    #idxs = (rank(psi) == 1 && psi.squared == true) ? [1, 2] : 1
    #norm_prev = real(norm_cum + log(contract(left, right, idxs, idxs)[1]))

    # Iteratively build left and right blocks and check for convergence
    for _ = 1:maxiter
        left_prev = deepcopy(left)
        right_prev = deepcopy(right)
        for j = 1:length(psis[1])
            left = buildleft(psis..., left, j)
            right = buildright(psis..., right, length(psis[1])+1-j)
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

    lefts = [left]
    rights = [right]

    # Build the environments to consider all sites
    for i = 1:length(psis[1])-1
        left = buildleft(psis..., left, i)
        right = buildright(psis..., right, length(psis[1])+1-i)
        push!(lefts, left)
        pushfirst!(rights, right)
    end

    return lefts, rights
end

function build!(psi::iGMPS; kwargs...)
    lefts, rights = build(psi; kwargs...)
    psi.lefts = lefts
    psi.rights = rights
    return nothing
end


### Calculating observables
"""
    inner(st::Sitetypes, psi::iGMPS, ops::OpList)
    inner(st::Sitetypes, rho::iGMPS, ops::OpList)
    inner(st::Sitetypes, phi::iGMPS, ops::OpList, psi::iGMPS)

Calculate the observables with respect to iMPS/iMPO.
"""
function inner(st::Sitetypes, psi::iGMPS, ops::OpList)
    vals = zeros(ComplexF64, length(ops.ops))
    for i = 1:length(ops.ops)
        vals[i] = inner(st, psi, ops.ops[i], ops.sites[i], ops.coeffs[i])
    end
    return vals
end

function inner(st::Sitetypes, phi::iGMPS, ops::OpList, psi::iGMPS)
    vals = zeros(ComplexF64, length(ops.ops))
    lefts, rights = build(phi, psi)
    for i = 1:length(ops.ops)
        vals[i] = inner(st, phi, ops.ops[i], psi, ops.sites[i], ops.coeffs[i], lefts, rights)
    end
    return vals
end

function inner(st::Sitetypes, phi::iGMPS, ops::Vector{String}, psi::iGMPS, sites::Vector{Int}, coeff::ComplexF64; kwargs...)
    return inner(st, phi, ops, psi, sites, coeff, build(phi, psi)...; kwargs...)
end

function inner(st::Sitetypes, psi::iGMPS, ops::Vector{String}, sites::Vector{Int}, coeff::ComplexF64; kwargs...)
    return inner(st, psi, ops, psi, sites, coeff, psi.lefts, psi.rights; kwargs...)
end

function inner(st::Sitetypes, phi::iGMPS, ops::Vector{String}, psi::iGMPS, sites::Vector{Int}, coeff::ComplexF64,
               lefts, rights; kwargs...)
    # Determine the range
    rng = sites[end] - sites[1] + 1

    # Loop over all periods
    val = 0.0
    for i = 1:length(psi)
        # Find relevent boundaries
        left = lefts[i]
        site = (i + rng - 2) % length(psi) + 1
        right = rights[site]
        #right = rights[length(psi) - ((rng + i) % length(psi))]

        # Contract with and without observables
        prod_obs = left
        prod_norm = left
        for j = 1:rng
            # Determine site and fetch relevent tensors
            site = (j + i - 2) % length(psi) + 1
            S_phi = diagm(phi.singulars[site])
            S = diagm(psi.singulars[site])
            A_phi = phi.tensors[site]
            A = psi.tensors[site]
            M = j in sites ? ops[findfirst([j == s for s in sites])] : "id"
            M = op(st, M)

            # Perform the contractions
            if psi.rank == 1 && psi.squared == true
                # Contract to find observable
                prod_obs = contract(prod_obs, S_phi, 1, 1)
                prod_obs = contract(prod_obs, S, 1, 1)
                prod_obs = contract(prod_obs, conj(A_phi), 1, 1)
                prod_obs = contract(prod_obs, M, 2, 1)
                prod_obs = contract(prod_obs, A, [1, 3], [1, 2])

                # Contract to find norm
                prod_norm = contract(prod_norm, S_phi, 1, 1)
                prod_norm = contract(prod_norm, S, 1, 1)
                prod_norm = contract(prod_norm, conj(A_phi), 1, 1)
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
        normal::Bool = get(kwargs, :normalize, true)
        val += normal ? prod_obs / prod_norm : prod_obs
    end
    return coeff * val / length(psi)
end


"""
    norm(psi::iGMPS)

Calculate the norm of an iMPS.
"""
function norm(psi::iGMPS)
    prod = psi.lefts[1]
    norm = 0
    for i = 1:length(psi)
        #prod = buildleft(psi, prod, i)
        norm += (psi.rank == 1 && psi.squared == true) ? 2*psi.norms[i] : psi.norms[i]
    end
    idxs = (psi.rank == 1 && psi.squared == true) ? [1, 2] : 1
    prod = contract(prod, psi.rights[1], idxs, idxs)
    norm += log(prod[1])
    norm -= log(contract(psi.lefts[1], psi.rights[1], idxs, idxs)[1])
    return norm / length(psi)
end


### Specifics for iGMPS 
function buildleft_phi_psi(phi::iGMPS, psi::iGMPS, left::Array{ComplexF64}, j::Int)
    left = contract(left, diagm(phi.singulars[j]), 1, 1)
    left = contract(left, diagm(psi.singulars[j]), 1, 1)
    left = contract(left, conj(phi.tensors[j]), 1, 1)
    left = contract(left, psi.tensors[j], [1, 2], [1, 2])
    return left
end

function buildleft_flat_psi(psi::iGMPS, left::Array{ComplexF64}, j::Int)
    left = contract(left, diagm(psi.singulars[j]), 1, 1)
    left = contract(left, ones(ComplexF64, dim(psi)), 1, 1)
    left = contract(left, psi.tensors[j], 2, 1)
    return left
end

function buildright_phi_psi(phi::iGMPS, psi::iGMPS, right::Array{ComplexF64}, j::Int)
    right = contract(psi.tensors[j], right, 3, 2)
    right = contract(conj(phi.tensors[j]), right, [2, 3], [2, 3])
    right = contract(diagm(psi.singulars[j]), right, 2, 2)
    right = contract(diagm(phi.singulars[j]), right, 2, 2)
    return right
end

function buildright_flat_psi(psi::iGMPS, right::Array{ComplexF64}, j::Int)
    right = contract(psi.tensors[j], right, 3, 1)
    right = contract(diagm(psi.singulars[j]), right, 2, 1)
    right = contract(ones(ComplexF64, dim(psi)), right, 1, 2)
    return right
end

function randomenv_phi_psi(phi::iGMPS, psi::iGMPS)
    left = rand(ComplexF64, size(phi.tensors[1])[1], size(psi.tensors[1])[1])
    right = rand(ComplexF64, size(phi.tensors[end])[end], size(psi.tensors[end])[end])
    return left, right
end

function randomenv_psi(psi::iGMPS)
    left = rand(ComplexF64, size(psi.tensors[1])[1], size(psi.tensors[1])[1])
    right = rand(ComplexF64, size(psi.tensors[end])[end], size(psi.tensors[end])[end])
    return left, right
end

function randomenv_flat_psi(psi::iGMPS)
    left = rand(ComplexF64, size(psi.tensors[1])[1])
    right = rand(ComplexF64, size(psi.tensors[end])[end])
    return left, right
end

### Specifics for iGMPO
function buildleft_rho(rho::iGMPS, left::Array{ComplexF64}, j::Int)
    left = contract(left, diagm(rho.singulars[j]), 1, 1)
    left = contract(left, trace(rho.tensors[j], 2, 3), 1, 1)
    return left
end

function buildright_rho(rho::iGMPS, right::Array{ComplexF64}, j::Int)
    right = contract(trace(rho.tensors[j], 2, 3), right, 2, 1)
    right = contract(diagm(rho.singulars[j]), right, 2, 1)
    return right
end

function randomenv_rho(rho::iGMPS)
    left = rand(ComplexF64, size(rho.tensors[1])[1])
    right = rand(ComplexF64, size(rho.tensors[end])[end])
    return left, right
end