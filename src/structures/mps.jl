using TensorOperations
import Base: *

mutable struct MPS
    dim::Int
    tensors::Vector{Array{Complex{Float64}, 3}}
end

function randomMPS(dim::Int, length::Int, bonddim::Int)
    tensors = []
    for i = 1:length
        D1 = i == 1 ? 1 : bonddim
        D2 = i == length ? 1 : bonddim
        push!(tensors, randn((D1, dim, D2)) + 1im*randn((D1, dim, D2)))
    end

    return MPS(dim, tensors)
end

Base.length(psi::MPS) = length(psi.tensors)
dim(psi::MPS) = psi.dim
Base.getindex(psi::MPS, i) = psi.tensors[i]
function Base.setindex!(psi::MPS, x, i)
    psi.tensors[i] == x
end

function bonddim(psi::MPS, site::Int)
    (site < 1 || site >= length(psi)) && return nothing
    return psi
end

function dot(psi::MPS, phi::MPS)
    prod = ones((1, 1))
    for i = 1:length(psi)
        A1 = psi[i]
        A2 = phi[i]
        @tensor begin
            prod[D2, d1, D3] := prod[D1, D2] * conj(A1[D1, d1, D3])
            prod[d1, D3, d2, D4] := prod[D2, d1, D3] * A2[D2, d2, D4]
            prod[D3, D4] := prod[d, D3, d, D4]
        end
    end

    return prod[1, 1]
end

*(psi::MPS, phi::MPS) = dot(psi, phi)
