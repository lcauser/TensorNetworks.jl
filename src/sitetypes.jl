mutable struct Sitetypes
    dim::Int
    statenames::Vector{String}
    states::Vector{Array{Complex{Float64}, 1}}
    opnames::Vector{String}
    ops::Vector{Array{Complex{Float64}, 2}}
    opdags::Vector{String}
    temp::Int
end

"""
    sitetype(dim::Int)

Create a sitetype.
"""
function sitetype(dim::Int)
    return Sitetypes(dim, [], [], [], [], [], 0)
end


### Getters
"""
    state(st::Sitetypes, name::String)

Find the state vector in a sitetype.
"""
function state(st::Sitetypes, name::String)
    # Find the index of the state
    idx = 0
    for i = 1:length(st.states)
        idx = name == st.statenames[i] ? i : idx
        idx != 0 && break
    end

    idx == 0 && error("The state $(name) is undefined.")
    return st.states[idx]
end


"""
    op(st::Sitetypes, name::String)

Find the operator matrix in a sitetype.
"""
function op(st::Sitetypes, name::String)
    # Find the index of the state
    idx = 0
    for i = 1:length(st.ops)
        idx = name == st.opnames[i] ? i : idx
        idx != 0 && break
    end

    idx == 0 && error("The operator $(name) is undefined.")
    return st.ops[idx]
end

"""
    dag(st::Sitetypes, name::String)

Find the name of the hermitian conjuage of an operator.
"""
function dag(st::Sitetypes, name::String)
    # Find the dagger
    idx = 0
    for i = 1:length(st.ops)
        idx = name == st.opnames[i] ? i : idx
        idx != 0 && break
    end

    idx == 0 && error("The operator $(name) is undefined.")
    return st.opdags[idx]
end


### Adders
"""
    add!(st::Sitetypes, name::String, A::Array{Complex{Float64}, 1})

Add a state to the sitetype. Must give a 1D array with the correction phyiscal
dimension.
"""
function add!(st::Sitetypes, name::String, A::Array{Complex{Float64}, 1})
    # Check that the vector is the right size
    size(A)[1] != st.dim && error("The vector must be dimension $(st.dim).")

    # Check the name doesn't exist
    name in st.statenames && error("The state name $(name) already exists.")

    # Add to list
    push!(st.statenames, name)
    push!(st.states, A)
end

"""
    add!(st::Sitetypes, name::String, A::Array{Complex{Float64}, 1},
         dag::string)

Add an operator to the sitetype. Must give a 2d array with the correct physical
dimensions
"""
function add!(st::Sitetypes, name::String, A::Array{Complex{Float64}, 2},
              dag::String)
    # Check that the matrix is the right size
    if size(A) != (st.dim, st.dim)
        error("The matrix must be dimensions ($(st.dim), $(st.dim)).")
    end

    # Check the name doesn't exist
    name in st.opnames && error("The operator name $(name) already exists.")

    # Add to list
    push!(st.opnames, name)
    push!(st.ops, A)
    push!(st.opdags, dag)
end


"""
    add!(st::Sitetypes, name::String, A, dag::String = "None")

Function to add to a sitetype with the incorrect type for A.
Converts A to the correct type and decides which function to use.
"""
function add!(st::Sitetypes, name::String, A, dag::String = "None")
    if length(size(A)) == 1
        A = convert(Array{Complex{Float64}, 1}, A)
        add!(st, name, A)
    elseif length(size(A)) == 2
        A = convert(Array{Complex{Float64}, 2}, A)
        add!(st, name, A, dag)
    else
        error("A is unsupported.")
    end
end


### Determine the product of operators
"""
    opprod(st::Sitetypes, names::Vector{String})

Determine the name of a product of operators, or add it to the sitetype and
return the choosen name.
"""
function opprod(st::Sitetypes, names::Vector{String})
    # Loop through each name contracting the operators
    prod = 1
    for i = 1:length(names)
        prod = i == 1 ? op(st, names[i]) : contract(prod, op(st, names[i]), 2, 1)
    end

    # Check to see if the operator is in the operator list
    idx = 0
    for i = 1:length(st.ops)
        idx = sum(abs.(prod - st.ops[i])) < 1e-12 ? i : idx
        idx != 0 && return st.opnames[idx]
    end

    # Add the operator
    dag = conj(moveidx(prod, 2, 1))
    name = "temp$(st.temp)"
    dagname = "temp$(st.temp+1)"
    add!(st, name, prod, dagname)
    add!(st, dagname, dag, name)
    st.temp += 2
    return name
end


### Create a gate from operators
function creategate(st::Sitetypes, ops::Vector{Vector{String}}, coeffs::Vector{Number})
    gate = 0
    for i = 1:length(ops)
        prod = 0
        for j = 1:length(ops[i])
            if j == 1
                prod = op(st, ops[i][j])
            else
                prod = tensorproduct(prod, op(st, ops[i][j]))
            end
        end
        if i == 1
            gate = coeffs[i]*prod
        else
            gate += coeffs[i]*prod
        end
    end
    return gate
end
