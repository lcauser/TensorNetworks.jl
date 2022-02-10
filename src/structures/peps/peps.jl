"""
    PEPS(dim::Int, tensors::Array{Array{ComplexF64, 5}})
    PEPS(dim::Int, height::Int, length::Int)
    PEPS(dim::Int, size::Int)

Initiate a PEPS.
"""

function PEPS(dim::Int, height::Int, length::Int)
    tensors = Array{Array{ComplexF64, 5}, 2}(undef, (height, length))
    for i = 1:height
        for j = 1:length
            tensors[i, j] = zeros(ComplexF64, 1, 1, 1, 1, dim)
        end
    end

    return GPEPS(1, dim, tensors)
end

function PEPS(dim::Int, size::Int)
    return PEPS(dim, size, size)
end

"""
    productPEPS(height::Int, length::Int, A::Array{Complex{Float64}, 5})
    productPEPS(size::Int, A::Array{Complex{Float64}, 5})
    productPEPS(height::Int, len::Int, A)
    productPEPS(size::Int, A)
    productPEPS(st::Sitetypes, names::Array{String, 2})
    productPEPS(st::Sitetypes, names::Vector{Vector{String}})

Create a product state PEPS.
"""

function productPEPS(height::Int, length::Int, A::Array{Complex{Float64}, 5})
    psi = GPEPS(1, size(A)[5], height, length)
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


function productPEPS(size::Int, A::Array{Complex{Float64}, 5})
    return productPEPS(size, size, A)
end


function productPEPS(height::Int, len::Int, A)
    if length(size(A)) == 1
        A = reshape(A, (1, 1, 1, 1, size(A)[1]))
    end
    A = convert(Array{ComplexF64, 5}, A)
    return productPEPS(height, len, A)
end


function productPEPS(size::Int, A)
    return productPEPS(size, size, A)
end


function productPEPS(st::Sitetypes, names::Array{String, 2})
    height, len = size(names)
    psi = GPEPS(1, st.dim, height, len)
    for i = 1:height
        for j = 1:len
            A = state(st, names[i, j])
            A = reshape(A, (1, 1, 1, 1, st.dim))
            A = convert(Array{ComplexF64, 5}, A)
            psi[i, j] = A
        end
    end
    return psi
end

function productPEPS(st::Sitetypes, names::Vector{Vector{String}})
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
    return productPEPS(st,namesArr)
end


"""
    randomPEPS(dim::Int, length::Int, height::Int, bonddim::Int)
    randomPEPS(dim::Int, size::Int, bonddim::Int)

Return a random PEPS.
"""
function randomPEPS(dim::Int, length::Int, height::Int, bonddim::Int)
    return randomGPEPS(1, dim, length, height, bonddim)
end

function randomPEPS(dim::Int, size::Int, bonddim::Int)
    return randomGPEPS(1, dim, size, size, bonddim)
end
