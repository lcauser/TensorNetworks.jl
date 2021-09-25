"""
    PEPS(dim::Int, tensors::Array{Array{ComplexF64, 3}})
    PEPS(dim::Int, height::Int, length::Int)
    PEPS(dim::Int, size::Int)

Initiate a PEPS.
"""
mutable struct PEPS <: AbstractPEPS
    dim::Int
    tensors::Array{Array{ComplexF64, 5}, 2}
end

function PEPS(dim::Int, height::Int, length::Int)
    tensors = Array{Array{ComplexF64, 5}, 2}(undef, (height, length))
    for i = 1:height
        for j = 1:length
            tensors[i, j] = zeros(ComplexF64, 1, 1, 1, 1, dim)
        end
    end

    return PEPS(dim, tensors)
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
    psi = PEPS(size(A)[5], height, length)
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
    psi = PEPS(st.dim, height, len)
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

### Apply gate
function applygate!(psi::PEPS, site::Vector{Int}, gate, direction::Bool=false; kwargs...)
    if !direction
        # Move gauges
        moveright!(psi, site[1], site[2]-1)
        movedown!(psi, site[1]-1, site[2])
        moveup!(psi, site[1]+1, site[2])
        movedown!(psi, site[1]-1, site[2]+1)
        moveup!(psi, site[1]+1, site[2]+1)
        moveleft!(psi, site[1], site[2]+2)

        # Contract tensors
        A1 = psi[site[1], site[2]]
        A2 = psi[site[1], site[2]+1]
        prod = contract(A1, A2, 4, 1)
        prod = contract(prod, gate, 4, 2)
        prod = trace(prod, 7, 10)
        dims1 = size(prod)[[1, 2, 3, 7]]
        dims2 = size(prod)[[4, 5, 6, 8]]
        prod, cmb1 = combineidxs(prod, [1, 2, 3, 7])
        prod, cmb2 = combineidxs(prod, [1, 2, 3, 4])

        # Split with SVD
        U, S, V = svd(prod, -1; kwargs...)
        S = sqrt.(S)
        U = contract(U, S, 2, 1)
        V = contract(S, V, 2, 1)

        # Reshape and restore
        A1 = reshape(U, [dims1..., size(S)[2]]...)
        A1 = moveidx(A1, 5, 4)
        A2 = reshape(V, [size(S)[1], dims2...]...)
        psi[site[1], site[2]] = A1
        psi[site[1], site[2]+1] = A2
    else
        # Move gauges
        moveright!(psi, site[1], site[2]-1)
        moveleft!(psi, site[1], site[2]+1)
        movedown!(psi, site[1]-1, site[2])
        moveright!(psi, site[1]+1, site[2]-1)
        moveleft!(psi, site[1]+1, site[2]+1)
        movedown!(psi, site[1]+2, site[2])

        # Contract tensors
        A1 = psi[site[1], site[2]]
        A2 = psi[site[1]+1, site[2]]
        prod = contract(A1, A2, 3, 2)
        prod = contract(prod, gate, 4, 2)
        prod = trace(prod, 7, 10)
        dims1 = size(prod)[[1, 2, 3, 7]]
        dims2 = size(prod)[[4, 5, 6, 8]]
        prod, cmb1 = combineidxs(prod, [1, 2, 3, 7])
        prod, cmb2 = combineidxs(prod, [1, 2, 3, 4])

        # Split with SVD
        U, S, V = svd(prod, -1; kwargs...)
        S = sqrt.(S)
        U = contract(U, S, 2, 1)
        V = contract(S, V, 2, 1)

        # Reshape and restore
        A1 = reshape(U, [dims1..., size(S)[2]]...)
        A1 = moveidx(A1, 5, 3)
        A2 = reshape(V, [size(S)[1], dims2...]...)
        A2 = moveidx(A2, 1, 2)
        psi[site[1], site[2]] = A1
        psi[site[1]+1, site[2]] = A2
    end
end


function rescale!(psi::PEPS, maxabs::Real=0)
    if maxabs==0
        for i = 1:length(psi)
            for j = 1:length(psi)
                maxabs += log(maximum(abs.(psi[i, j])))
            end
        end
        maxabs = exp(maxabs/(length(psi)^2))
    end

    for i = 1:length(psi)
        for j = 1:length(psi)
            psi[i, j] *= maxabs / (maximum(abs.(psi[i, j])))
        end
    end
end

### Move singular values
function movedown!(psi::PEPS, i::Int, j::Int)
    if i < N && i >=1 && j >= 1 && j <= N
        U, S, V = svd(psi[i, j], 3)
        psi[i, j] = U
        S = contract(S, V, 2, 1)
        S = contract(S, psi[i+1, j], 2, 2)
        S = moveidx(S, 1, 2)
        psi[i+1, j] = S
    end
end

function moveup!(psi::PEPS, i::Int, j::Int)
    if i <= N && i > 1 && j >= 1 && j <= N
        U, S, V = svd(psi[i, j], 2)
        psi[i, j] = U
        S = contract(S, V, 2, 1)
        S = contract(S, psi[i-1, j], 2, 3)
        S = moveidx(S, 1, 3)
        psi[i-1, j] = S
    end
end

function moveleft!(psi::PEPS, i::Int, j::Int)
    if i <= N && i >= 1 && j > 1 && j <= N
        U, S, V = svd(psi[i, j], 1)
        psi[i, j] = U
        S = contract(S, V, 2, 1)
        S = contract(S, psi[i, j-1], 2, 4)
        S = moveidx(S, 1, 4)
        psi[i, j-1] = S
    end
end

function moveright!(psi::PEPS, i::Int, j::Int)
    if i <= N && i >= 1 && j >= 1 && j < N
        U, S, V = svd(psi[i, j], 4)
        psi[i, j] = U
        S = contract(S, V, 2, 1)
        S = contract(S, psi[i, j+1], 2, 1)
        psi[i, j+1] = S
    end
end


"""
function movedown!(psi::PEPS, i::Int, j::Int)
    if i < N && i >=1 && j >= 1 && j <= N
        Q, R = qr(psi[i, j], 3)
        psi[i, j] = Q
        R = contract(R, psi[i+1, j], 2, 2)
        R = moveidx(R, 1, 2)
        psi[i+1, j] = R
    end
end

function moveup!(psi::PEPS, i::Int, j::Int)
    if i <= N && i > 1 && j >= 1 && j <= N
        L, Q = lq(psi[i, j], 2)
        psi[i, j] = Q
        L = contract(L, psi[i-1, j], 2, 3)
        L = moveidx(L, 1, 3)
        psi[i-1, j] = L
    end
end

function moveleft!(psi::PEPS, i::Int, j::Int)
    if i <= N && i >= 1 && j > 1 && j <= N
        L, Q = lq(psi[i, j], 1)
        psi[i, j] = Q
        L = contract(L, psi[i, j-1], 2, 4)
        L = moveidx(L, 1, 4)
        psi[i, j-1] = L
    end
end

function moveright!(psi::PEPS, i::Int, j::Int)
    if i <= N && i >= 1 && j >= 1 && j < N
        Q, R = qr(psi[i, j], 4)
        psi[i, j] = Q
        R = contract(R, psi[i, j+1], 2, 1)
        psi[i, j+1] = R
    end
end
"""

### Reduced tensors
function reducedtensor(psi::PEPS, i::Int, j::Int, axis::Int)
    # Check it's a virutal axis
    (axis < 1 || axis > 4) && error("Axis must be between 1 and 4.")

    # Group relevent idx with physical axis
    A = psi[i, j]
    d = dim(psi)
    D = size(A)[axis]
    A, cmb = combineidxs(A, [axis, 5])

    ### CHANGE TO QR ONCE WE KNOW IT WORKS WITH SVD
    # Apply SVD to split to reduced tensor
    U, S, V = svd(A, 4)
    V = contract(S, V, 2, 1)

    # Organize the axes
    U = moveidx(U, 4, axis)
    V = reshape(V, (size(V)[1], D, d))
    if axis == 1 || axis == 2
        V = moveidx(V, 1, 2)
    end
    return U, V
end

function reducedtensor(A, axis::Int)
    # Check it's a virutal axis
    (axis < 1 || axis > 4) && error("Axis must be between 1 and 4.")

    # Group relevent idx with physical axis
    d = size(A)[5]
    D = size(A)[axis]
    A, cmb = combineidxs(A, [axis, 5])

    ### CHANGE TO QR ONCE WE KNOW IT WORKS WITH SVD
    # Apply SVD to split to reduced tensor
    #U, S, V = svd(A, 4)
    #V = contract(S, V, 2, 1)
    U, V = qr(A, 4)

    # Organize the axes
    U = moveidx(U, 4, axis)
    V = reshape(V, (size(V)[1], D, d))
    if axis == 1 || axis == 2
        V = moveidx(V, 1, 2)
    end
    return U, V
end

### Save and write
function HDF5.write(parent::Union{HDF5.File, HDF5.Group}, name::AbstractString,
                    psi::PEPS)
    g = create_group(parent, name)
    attributes(g)["type"] = "PEPS"
    attributes(g)["version"] = 1
    write(g, "height", size(psi)[1])
    write(g, "length", size(psi)[2])
    write(g, "dim", psi.dim)
    for i = 1:size(psi)[1]
        for j = 1:size(psi)[2]
            write(g, "PEPS[$(i), $(j)]", psi[i, j])
        end
    end
end

function HDF5.read(parent::Union{HDF5.File, HDF5.Group}, name::AbstractString,
                    ::Type{PEPS})
    g = open_group(parent, name)
    if read(attributes(g)["type"]) != "PEPS"
        error("HDF5 group of file does not contain PEPS data.")
    end
    height = read(g, "height")
    len = read(g, "length")
    dim = read(g, "dim")
    psi = PEPS(dim, height, len)
    for i = 1:height
        for j = 1:len
            psi[i, j] = read(g, "PEPS[$(i), $(j)]")
        end
    end
    return psi
end
