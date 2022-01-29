"""
    PEPS(dim::Int, tensors::Array{Array{ComplexF64, 6}})
    PEPS(dim::Int, size::Int)

Initiate a PEPS.
"""
mutable struct PEPO <: AbstractPEPO
    dim::Int
    tensors::Array{Array{ComplexF64, 5}, 2}
end

function PEPO(dim::Int, height::Int, length::Int)
    tensors = Array{Array{ComplexF64, 6}, 2}(undef, (height, length))
    for i = 1:height
        for j = 1:length
            tensors[i, j] = zeros(ComplexF64, 1, 1, 1, 1, dim, dim)
        end
    end

    return PEPO(dim, tensors)
end

function PEPO(dim::Int, size::Int)
    return PEPO(dim, size, size)
end

"""
    productPEPO(size::Int, A::Array{Complex{Float64}, 6})
    productPEPO(height::Int, len::Int, A)
    productPEPO(size::Int, A)
    productPEPO(st::Sitetypes, names::Array{String, 2})
    productPEPO(st::Sitetypes, names::Vector{Vector{String}})

Create a product state PEPO.
"""

function productPEPO(height::Int, length::Int, A::Array{Complex{Float64}, 6})
    psi = PEPS(size(A)[6], height, length)
    for i = 1:height
        for j = 1:length
            tensor = A
            if i == 1
                tensor = tensor[:, end:end, :, :, :]
            elseif i == height
                tensor = tensor[:, :, end:end, :, :]
            end
            if j == 1
                tensor = tensor[end:end, :, :, :, :]
            elseif j == length
                tensor = tensor[:, :, :, end:end, :]
            end
            psi[i, j] = tensor
        end
    end
    return psi
end


function productPEPO(size::Int, A::Array{Complex{Float64}, 6})
    return productPEPO(size, size, A)
end


function productPEPO(height::Int, len::Int, A)
    if length(size(A)) == 2
        A = reshape(A, (1, 1, 1, 1, size(A)[1], size(A)[2]))
    end
    A = convert(Array{ComplexF64, 6}, A)
    return productPEPO(height, len, A)
end


function productPEPO(size::Int, A)
    return productPEPO(size, size, A)
end


function productPEPO(st::Sitetypes, names::Array{String, 2})
    height, len = size(names)
    O = PEPO(st.dim, height, len)
    for i = 1:height
        for j = 1:len
            A = op(st, names[i, j])
            A = reshape(A, (1, 1, 1, 1, st.dim, st.dim))
            A = convert(Array{ComplexF64, 6}, A)
            O[i, j] = A
        end
    end
    return O
end

function productPEPO(st::Sitetypes, names::Vector{Vector{String}})
    len = 0
    for i = 1:length(names)
        len = i == 1 ? length(names[1]) : len
        len != length(names[i]) && error("PEPS must have a rectangle structure.")
    end
    namesArr = Array{String, 2}(undef, length(names), len)
    for i = 1:length(names)
        for j = 1:len
            namesArr[i, j] = names[i][j]
        end
    end
    return productPEPO(st,namesArr)
end

function randomPEPO(dim::Int, length::Int, bonddim::Int)
    O = PEPO(dim, length)
    for i = 1:length
        for j = 1:length
            A = randn(ComplexF64, bonddim, bonddim, bonddim, bonddim, dim, dim)
            if j == 1
                A = A[1:1, :, :, :, :, :]
            end
            if j == length
                A = A[:, :, :, 1:1, :, :]
            end
            if i == 1
                A = A[:, 1:1, :, :, :, :]
            end
            if i == length
                A = A[:, :, 1:1, :, :, :]
            end
            O[i, j] = A
        end
    end
    return O
end
