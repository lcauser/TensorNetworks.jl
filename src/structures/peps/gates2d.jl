mutable struct gates2d
    length::Int
    gates::Vector{Array{ComplexF64}}
    sites::Vector{Vector{Int}}
    directions::Vector{Bool}
end

length(gl::gates2d) = length(gl.gates)

function trotterize(st::Sitetypes, oplist::OpList2d, dt::Float64)
    gates = []
    sites = []
    directions = []

    # Add horizontal gates
    for direction = [false, true]
        for i = 1:oplist.length
            for j = 1:oplist.length
                # Get the gate at the site
                gate = sitetensor(st, oplist, [i, j], direction)
                if gate != false
                    # Exponentiate the gate
                    gate = exp(dt*gate, [2, 4])

                    # Add to list
                    push!(gates, gate)
                    push!(sites, [i, j])
                    push!(directions, direction)
                end
            end
        end
    end

    return gates2d(oplist.length, gates, sites, directions)
end


function getgate(gl::gates2d, i::Int, j::Int, dir::Bool)
    # Find the indexs with the given sites
    idxs = []
    for k = 1:length(gl.sites)
        if gl.sites[k][1] == i && gl.sites[k][2] == j
            push!(idxs, k)
        end
    end
    # check right direction
    for idx = idxs
        gl.directions[idx] == dir && return gl.gates[idx]
    end

    return 0
end
