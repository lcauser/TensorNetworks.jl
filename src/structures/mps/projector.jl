mutable struct MPSProjector <: AbstractMPS
    dim::Int
    leftMPS::GMPS
    rightMPS::GMPS
    constant::ComplexF64
end

"""
    MPSProjector(dim::Int, leftMPS::GMPS, rightMPS::GMPS, constant::ComplexF64)
    MPSProjector(left::GMPS, right::GMPS)

Create a projector from a ket and bra MPS.
"""
function MPSProjector(left::GMPS, right::GMPS; kwargs...)
    # Ensure MPS share same properties
    dim(left) != dim(right) && error("The MPS must share the same physical dimension.")
    length(left) != length(right) && error("The MPS must be the same length.")

    # Fetch constant
    constant::ComplexF64 = get(kwargs, :constant, 0.0)
    constant = constant == 0.0 ? inner(left, right) : constant

    # Create the projection
    return MPSProjector(dim(left), left, right, constant)
end

function MPSProjector(psi::GMPS; kwargs...)
    return MPSProjector(psi, psi; kwargs...)
end


### Products
# Outer product with MPS
"""
    applyMPO(O::MPSProjector, psi::GMPS, hermitian=false)

Apply an MPS Projector to an MPS. Specify whether to apply the hermitian conjugate.
"""
function applyMPO(O::MPSProjector, psi::GMPS, hermitian=false)
    if !hermitian
        return (inner(O.leftMPS, psi) / O.constant) * deepcopy(O.rightMPS)
    else
        return (inner(psi, O.rightMPS) / O.constant) * conj(O.leftMPS)
    end
end

*(O::MPSProjector, psi::GMPS) = applyMPO(O, psi)
*(psi::GMPS, O::MPSProjector) = applyMPO(O, psi, true)
