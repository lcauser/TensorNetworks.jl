#=
    Uniform matrix product states are an infinite ansatz for correlated
    one dimensional systems. We assumed mixed canonical representation.
    For details on the workings, see:
    https://scipost.org/SciPostPhysLectNotes.7/pdf
=#

mutable struct uMPS
    dim::Int
    Al::Array{ComplexF64}
    Ar::Array{ComplexF64}
    C::Array{ComplexF64}
end

function randomUMPS(dim::Int, bonddim::Int; kwargs...)
    # Create random cell and contract into two cell 
    A = rand(ComplexF64, bonddim, dim, bonddim)
    A /= norm(A)
    return UMPS(A)
end

function UMPS(A::Array{T}) where {T<:Number}
    # Orthogonalise
    Al, Ar, C, λ = mixedCanonical(A)

    return uMPS(size(A)[2], Al, Ar, C)
end

# Properties
dim(psi::uMPS) = psi.dim
maxbonddim(psi::uMPS) = size(psi.Al)[1]


### Orthogonalisation 
"""
    leftOrthonormalise(A::Array{<:Number}, L::Array{<:Number}, η<:Real=1e-12)
    leftOrthonormalise(A::Array{<:Number}, η<:Real=1e-12)

Iteratively find the left canonical orthogonalisation of an array A.
L is some initial guess for orthogonalisation, η is a tolerance.
"""
function leftOrthonormalise(A::Array{T}, L::Array{Q}, η::S=1e-12) where {T<:Number, Q<:Number, S<:Real}
    # Normalise L
    L /= norm(L)

    δ = 1
    local Al, λ
    while δ > η
        # QR Decomposition
        Lold = deepcopy(L)
        Al, L = qr(contract(L, A, 2, 1), 3)

        # Find the fixed point 
        #f(X) = _lo_eigsolve(X, A, Al)
        #_, L = eigsolve(f, L, 1, :LM; tol=δ/10)
        #_, L = qr(Ls[1], 2)

        # Determine error 
        λ = norm(L)
        L = L ./ λ
        δ = norm(L .- Lold)
    end
    
    return Al, L, λ
end

function leftOrthonormalise(A::Array{T}, η::S=1e-12) where {T<:Number, S<:Real}
    return leftOrthonormalise(A, rand(ComplexF64, size(A)[[1, 3]]...), η)
end

"""
    rightOrthonormalise(A::Array{<:Number}, R::Array{<:Number}, η<:Real=1e-12)
    rightOrthonormalise(A::Array{<:Number}, η<:Real=1e-12)

Find the right canonical orthogonalisation of an array A.
R is some initial guess for orthogonalisation, η is a tolerance.
"""
function rightOrthonormalise(A::Array{T}, R::Array{Q}, η::S=1e-12) where {T<:Number, Q<:Number, S<:Real}
    # Normalise R
    R /= norm(R)

    δ = 1
    local Ar, λ
    while δ > η
        # QR Decomposition
        Rold = deepcopy(R)
        R, Ar = lq(contract(A, R, 3, 1), 1)

        # Find the fixed point 
        #f(X) = _ro_eigsolve(X, A, Ar)
        #_, R = eigsolve(f, R, 1, :LM; maxiter=1)
        #R, _ = lq(R[1], 1)

        # Determine error 
        λ = norm(R)
        R = R ./ λ
        δ = norm(R .- Rold)
    end

    return Ar, R, λ
end

function rightOrthonormalise(A::Array{T}, η::S=1e-12) where {T<:Number, S<:Real}
    return rightOrthonormalise(A, rand(ComplexF64, size(A)[[1, 3]]...), η)
end

"""
    mixedCanonical(A::Array{<:Number}, η<:Real=1e-14)

Put some tensor A into mixed canonical form.
"""
function mixedCanonical(A::Array{T}, η::S=1e-14) where {T<:Number, S<:Real}
    Al, _, λ = leftOrthonormalise(A, η)
    Ar, C, _ = rightOrthonormalise(Al, η)
    #U, C, V = svd(C, 2) 
    #Al = contract(conj(U), contract(Al, U, 3, 1), 1, 1)
    #Ar = contract(V, contract(Ar, conj(V), 3, 2), 2, 1)

    return Al, Ar, C, λ
end


function _lo_eigsolve(X::T, A::Array{S}, Al::Array{Q}) where {T, S<:Number, Q<:Number}
    prod = contract(X, A, 1, 1)
    prod = contract(prod, Al, [1, 2], [1, 2])
    return prod
end


function _ro_eigsolve(X::T, A::Array{S}, Ar::Array{Q}) where {T, S<:Number, Q<:Number}
    prod = contract(Ar, X, 3, 1)
    prod = contract(A, prod, [2, 3], [2, 3])
    return prod
end


### Expectations of operator lists 

"""
    inner(st::Sitetypes, psi::uMPS, oplist::InfiniteOpList)

Calculate the expectation value of operators for a uniform MPS.
"""
function inner(st::Sitetypes, psi::uMPS, oplist::InfiniteOpList)
    # Contract the centre 
    prod = contract(conj(psi.C), psi.C, 1, 1)

    # Loop through each operator in the list 
    expectations = zeros(ComplexF64, length(oplist.ops))
    for i in eachindex(oplist.ops)
        ex = deepcopy(prod)
        for j = 1:oplist.sites[i][end]
            ex = contract(ex, conj(psi.Ar), 1, 1)
            if j in oplist.sites[i]
                idx = findfirst(oplist.sites[i] .== j)
                ex = contract(ex, op(st, oplist.ops[i][idx]), 2, 1)
                ex = moveidx(ex, 3, 2)
            end
            ex = contract(ex, psi.Ar, [1, 2], [1, 2])
        end
        expectations[i] = trace(ex, 1, 2)[] * oplist.coeffs[i]
    end

    return expectations
end


### Expectation of MPOs 
"""
    inner(psi::uMPS, O::GMPS)

Calculate the expectation of an MPO with a uniform MPS.
"""
function inner(psi::uMPS, O::GMPS)
    # Contract the centre 
    prod = contract(conj(psi.C), psi.C, 1, 1)
    prod = tensorproduct(prod, ones(ComplexF64, 1))
    prod = moveidx(prod, 3, 2)

    # Contract with MPO 
    for i in eachindex(O.tensors)
        prod = contract(prod, conj(psi.Ar), 1, 1)
        prod = contract(prod, O.tensors[i], [1, 3], [1, 2])
        prod = contract(prod, psi.Ar, [1, 3], [1, 2])
    end

    prod = trace(prod, 1, 3)
    return prod[]
end


### Finding the left and right contributions to a translationally invariant
### operator 

"""
    leftEnvironment(psi::uMPS, O::GMPS, h<:Number; kwargs...) 
    leftEnvironment(psi::uMPS, O::GMPS, h<:Number, HL::Array{<:Number}; kwargs...) 

Calculate the contributions of a translationally invariant opeartor from the
left of the orthogonal center, with each local term shifted by h.This is
calculated analytically as a geometric sum, but numerically through iterative means.
There is an option to provide an inital guess, HL.
"""
function leftEnvironment(psi::uMPS, O::GMPS, h::Q, HL::Array{S}; kwargs...) where {Q<:Number, S<:Number}
    # Contract with the MPO to find the constant 
    V = diagm(ones(ComplexF64, maxbonddim(psi)))
    V = tensorproduct(V, ones(ComplexF64, 1))
    V = moveidx(V, 3, 2)
    for i in eachindex(O.tensors)
        V = contract(V, conj(psi.Al), 1, 1)
        V = contract(V, O[i], [1, 3], [1, 2])
        V = contract(V, psi.Al, [1, 3], [1, 2])
    end
    V = V[:, 1, :] .- h*diagm(ones(ComplexF64, maxbonddim(psi)))

    # Iteratively solve 
    f(x) = _left_env_projection(psi, x)

    tol::Float64 = get(kwargs, :tol, 1e-10)
    HL, _ = linsolve(f, V, HL; tol=tol)

    return HL
end

function leftEnvironment(psi::uMPS, O::GMPS, h::Q; kwargs...) where {Q<:Number}
    HL = rand(ComplexF64, maxbonddim(psi), maxbonddim(psi))
    return leftEnvironment(psi, O, h, HL; kwargs...)
end

function _left_env_projection(psi::uMPS, HL::Array{T}) where {T<:Number}
    # Find the right fixed point
    R = contract(conj(psi.C), psi.C, 1, 1)

    # Contract the guess L with the tensors Al
    prod = contract(HL, conj(psi.Al), 1, 1)
    prod = contract(prod, psi.Al, [1, 2], [1, 2])
    prod2 = contract(HL, R, [1, 2], [1, 2]) .* diagm(ones(ComplexF64, size(prod)[1]))
    return HL .- prod .+ prod2
end

"""
    rightEnvironment(psi::uMPS, O::GMPS, h<:Number; kwargs...) 
    rightEnvironment(psi::uMPS, O::GMPS, h<:Number, HR::Array{<:Number}; kwargs...) 

Calculate the contributions of a translationally invariant opeartor from the
right of the orthogonal center, with each local term shifted by h. This is
calculated analytically as a geometric sum, but numerically through iterative means.
There is an option to provide an inital guess, HR.
"""
function rightEnvironment(psi::uMPS, O::GMPS, h::Q, HR::Array{S}; kwargs...) where {Q<:Number, S<:Number}
    # Contract with the MPO to find the constant 
    V = diagm(ones(ComplexF64, maxbonddim(psi)))
    V = tensorproduct(V, ones(ComplexF64, 1))
    V = moveidx(V, 3, 2)
    for i in reverse(eachindex(O.tensors))
        V = contract(psi.Ar, V, 3, 3)
        V = contract(O[i], V, [3, 4], [2, 4])
        V = contract(conj(psi.Ar), V, [2, 3], [2, 4])
    end
    V = V[:, 1, :] .- h*diagm(ones(ComplexF64, maxbonddim(psi)))

    # Iteratively solve 
    f(x) = _right_env_projection(psi, x)

    tol::Float64 = get(kwargs, :tol, 1e-10)
    HR, _ = linsolve(f, V, HR; tol=tol)

    return HR
end

function rightEnvironment(psi::uMPS, O::GMPS, h::Q; kwargs...) where {Q<:Number}
    HR = rand(ComplexF64, maxbonddim(psi), maxbonddim(psi))
    return rightEnvironment(psi, O, h, HR; kwargs...)
end


function _right_env_projection(psi::uMPS, HR::Array{T}) where {T<:Number}
    # Calculate left fixed point
    L = contract(psi.C, conj(psi.C), 2, 2)

    # Contract the guess L with the tensors Al
    prod = contract(psi.Ar, HR, 3, 2)
    prod = contract(conj(psi.Ar), prod, [2, 3], [2, 3])
    prod2 = contract(L, HR, [1, 2], [1, 2]) .* diagm(ones(ComplexF64, size(prod)[1]))
    return HR .- prod .+ prod2
end