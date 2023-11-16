"""
    QS(dim::Int, length::Int)

Create a QS with physical dimension dim.
"""
function QS(dim::Int, length::Int)
    return GQS(1, dim, length, zeros(ComplexF64, [dim for _ = 1:length]...))
end


"""
    isQS(psi::GQS)

Check if a GQS is of rank 1.
"""
function isQS(psi::GQS)
    rank(psi) == 1 && return true
    return false
end


### Create QS
"""
    randomQS(dim::Int, length::Int)

Create a QS with random entries.
"""
function randomQS(dim::Int, length::Int)
    return randomGQS(1, dim, length)
end


"""
    productQS(sites::Int, A<:AbstractArray)

Create a product QS of some fixed local state.
"""
function productQS(sites::Int, A::Vector{R}) where R<:Number
    tensor = deepcopy(A)
    for _ = 1:sites-1
        tensor = tensorproduct(tensor, A)
    end
    return GQS(1, length(A), sites, tensor)
end

"""
    productQS(st::Sitetypes, names::Vector{String})

Create a product state from the names of local states on a sitetype.
"""
function productQS(st::Sitetypes, names::Vector{String})
    tensor = 1
    for i = 1:length(names)
        tensor = i == 1 ? state(st, names[i]) : tensorproduct(tensor, state(st, names[i]))
    end
    return GQS(1, st.dim, length(names), tensor)
end


### Calculate inner products of each operator in the operator list with respect
### to two QSs
"""
    inner(st::Sitetypes, psi::GQS, oplist::OpList, phi::GQS)

Efficently calculate the expectations of a list of operators between two
QS.
"""
function inner(st::Sitetypes, psi::GQS, oplist::OpList, phi::GQS)
    # Loop through each operator
    expectations = [0.0 + 0.0im for i = 1:length(oplist.sites)]
    for idx = 1:length(oplist.ops)
        prod = phi.tensor
        for i = 1:length(oplist.ops[idx])
            prod = contract(prod, op(st, oplist.ops[idx][i]), oplist.sites[idx][i], 2)
            prod = moveidx(prod, length(size(prod)), oplist.sites[idx][i])
        end
        expectations = oplist.coeffs[idx] * sum(conj(psi.tensor) .* prod)
    end
    return expectations
end

"""
    applyop!(st::Sitetypes, psi::GQS, ops, sites, coeff::Number=1)

Apply local operators to an QS.
"""
function applyop!(st::Sitetypes, psi::GQS, ops, sites, coeff = 1.0)
    for i = 1:length(ops)
        psi.tensor = contract(psi.tensor, op(st, ops[i]), sites[i], 2)
        psi.tensor = moveidx(psi.tensor, length(size(psi.tensor)), sites[i])
    end
    psi.tensor *= coeff 
end