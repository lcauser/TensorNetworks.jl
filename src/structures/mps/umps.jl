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
    mixedCanonical(A::Array{<:Number}, η<:Real=1e-12)

Put some tensor A into mixed canonical form.
"""
function mixedCanonical(A::Array{T}, η::S=1e-12) where {T<:Number, S<:Real}
    Al, _, λ = leftOrthonormalise(A, η)
    Ar, C, _ = rightOrthonormalise(A, η)
    U, C, V = svd(C, 2) 
    Al = contract(conj(U), contract(Al, U, 3, 1), 1, 1)
    Ar = contract(conj(V), contract(Ar, V, 3, 1), 1, 1)

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


### Expectation values 
