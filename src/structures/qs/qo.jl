"""
    QO(dim::Int, length::Int)

Create a QO with physical dimension dim.
"""
function QO(dim::Int, length::Int)
    return GQS(2, dim, length, zeros(ComplexF64, [dim for _ = 1:2*length]...))
end



"""
    isQO(psi::GQS)

Check if a GQS is of rank 2.
"""
function isQO(O::GQS)
    rank(O) == 2 && return true
    return false
end


# Manipulations of QO
"""
    adjoint(O::GQS)

Calculate the Hermitian conjugate of an operator (QO).
"""
function adjoint(O::GQS)
    rank(O) != 2 && error("The generalised MPS must be of rank 2 (an QO).")
    O2 = deepcopy(O)
    for i = 1:O2.length
        O2.tensor = moveidx(O2.tensor, 2*i-1, 2*i)
    end
    O2.tensor = conj(O2.tensor)
    return O2
end


### Create QOs
"""
    randomQO(dim::Int, length::Int)

Create a QO with random entries.
"""
function randomQO(dim::Int, length::Int)
    return randomGQS(2, dim, length)
end


"""
    productQO(sites::Int, A::Array{Complex{Float64}, 2})

Create a product QO of some fixed tensor.
A can be a vector for product state entries, or larger dimensional tensor which
is truncated at the edge sites.
"""
function productQO(sites::Int, A::Array{Complex{Float64}, 2})
    tensor = deepcopy(A)
    for _ = 1:sites-1
        tensor = tensorproduct(tensor, A)
    end
    return GQS(2, length(A), sites, tensor)
end


"""
    productQO(st::Sitetypes, names::Vector{String})

Create a product operator from the names of local operators on a sitetype.
"""
function productQO(st::Sitetypes, names::Vector{String})
    tensor = 1
    for i = 1:length(names)
        tensor = i == 1 ? op(st, names[i]) : tensorproduct(tensor, op(st, names[i]))
    end
    return GQS(2, st.dim, length(names), tensor)
end


### Products
"""
    applyQO(O::GQO, psi::GQS; kwargs...)
    applyQO(psi::GQS, O::GQS; kwargs...)
    applyQO(O::GQS, O::GQS; kwargs...)

Apply an operator (QO) O to a state (MPS) psi, O|Ψ> or <Ψ|O.
Or apply an operator (QO) O1 to an operator (QO) O2.
"""
function applyQO(arg1::GQS, arg2::GQS; kwargs...)
    # Check the arguments share the same physical dimensions and length
    dim(arg1) != dim(arg2) && error("GQS must share the same physical dims.")
    length(arg1) != length(arg2) && error("GQS must share the same length.")

    # Determine which is psi and which is O
    if rank(arg1)==1 && rank(arg2)==1
        return inner(arg1, arg2; kwargs...)
    elseif rank(arg1)==1 && rank(arg2)==2
        return QOQSProduct(adjoint(arg2), arg1; kwargs...)
    elseif rank(arg1)==2 && rank(arg2)==1
        return QOQSProduct(arg1, arg2; kwargs...)
    elseif rank(arg1)==2 && rank(arg2)==2
        return QOQOProduct(arg1, arg2; kwargs...)
    else
        error("Unallowed combinations of quantum state ranks.")
        return false
    end
end
*(O::GQS, psi::GQS) = applyQO(O, psi)


function QOQSProduct(O::GQS, psi::GQS; kwargs...)
    # Create a copy 
    phi = deepcopy(psi)

    # Multiply the tensors 
    phi.tensor = contract(O.tensor, psi.tensor, [2*i for i = 1:length(psi)], collect(1:length(psi)))
    return phi
end

function QOQOProduct(O1::GQS, O2::GQS; kwargs...)
    # Create a copy 
    O = deepcopy(psi)

    # Multiply the tensors 
    phi.tensor = contract(O1.tensor, O2.tensor, [2*i for i = 1:length(O)], [2*i-1 for i = 1:length(O)])

    # Permute indices
    for i = 1:length(O)
        O.tensor = moveidx(O.tensor, length(psi)+i, 2*i)
    end
    return O
end

"""
    inner(phi::GQS, psi::GQS)
    inner(phi::GQS, O::GQS, psi::GQS)
    inner(phi::GQS, ..., psi::GQS)
    inner(O1:GQS, phi::GQS, O2::GQS, psi::GQS)
    dot(phi::GQS, psi::GQS)
    *(phi::GQS, psi::GQS)

Calculate the inner product of some operator with respect to a bra and ket.
"""
function inner(args::GQS...)
    # Check to make sure all arguments have the same properties
    length(args) < 2 && error("There must be atleast 2 MPS arguments (rank 1).")
    dims = [dim(arg)==dim(args[1]) for arg in args]
    sum(dims) != length(args) && error("GQS must share the same physical dim.")
    lengths = [length(arg)==length(args[1]) for arg in args]
    sum(lengths) != length(args) && error("GQS must share the same length.")

    # Check for a bra and ket
    ranks = [rank(arg) for arg in args]
    sum([rank == 1 for rank in ranks]) != 2 && error("The inner product must have a braket structure.")
    ranks[end] != 1 && error("The final GQS must be rank 1 (MPS).")

    # Re-arrange to form a braket
    idx = 0
    for j = 1:length(ranks)
        idx = j
        ranks[j] == 1 && break
    end
    for j = idx-1:-1:1
        psi = args[j+1]
        O = adjoint(args[j])
        args[j] = psi
        args[j+1] = O
    end

    # Calculate the inner product
    prod = args[end]
    for i = 1:length(args)-2
        prod = QOQSProduct(args[length(args)-i], prod)
    end
    prod = sum(conj(args[1].tensor) .* prod.tensor)

    return prod
end
dot(phi::GQS, psi::GQS) = inner(phi, psi)


"""
    trace(O::GQS)
    trace(O1::GQS, O2::GQS)
    trace(O1::GQS, O2::GQS, ...)

Determine the trace of an QO / product of QOs..
"""
function trace(args::GQS...)
    # Check to make sure all arguments have the same properties
    length(args) < 1 && error("There must be atleast 1 QO arguments (rank 2).")
    dims = [dim(arg)==dim(args[1]) for arg in args]
    sum(dims) != length(args) && error("GQS must share the same physical dim.")
    lengths = [length(arg)==length(args[1]) for arg in args]
    sum(lengths) != length(args) && error("GQS must share the same length.")

    # Check that they're QOs
    ranks = [rank(arg) for arg in args]
    sum([rank != 2 for rank in ranks]) > 0 && error("Arguments must be GQS of rank 2 (QO).")

    # Calculate the trace
    prod = args[end]
    for i = 1:length(args)-1
        prod = QOQSProduct(args[length(args)-i], prod)
    end
    
    # Trace indexs 
    prod = prod.tensor
    for _ = 1:length(args[1])
        prod = trace(prod, 1, 2)
    end

    return prod[]
end


### Automatically construct QOs from operator lists
"""
     QO(st::Sitetypes, H::OpList; kwargs...)

Construct an QO from an operator list.
"""
function QO(st::Sitetypes, H::OpList; kwargs...)
    # System properties
    N = length(H)
    d = st.dim

    # Make the operator
    Hops = zeros(ComplexF64, [st.dim for _ = 1:2*length(H)]...)
    for i = 1:length(H.ops)
        oper = ones(ComplexF64)
        k = 1
        for j = 1:length(H)
            oper_term = H.sites[i][k] == j ? H.ops[i][k] : "id"
            k = min(H.sites[i][k] == j ? k + 1 : k, length(H.sites[i]))
            oper = tensorproduct(oper, op(st, oper_term))
        end
        Hops += H.coeffs[i] * oper
    end

    return GQS(2, d, N, Hops)
end


### Exponentiate a quantum operator 
function exp(O::GQS)
    O2 = GQS(O.rank, O.dim, O.length)
    O2.tensor = exp(O.tensor, [2*i for i = 1:O.length])
    return O2
end
