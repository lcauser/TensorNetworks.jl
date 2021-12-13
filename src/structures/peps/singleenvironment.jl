mutable struct SingleEnvironment
    psi::PEPS
    configuration::Array
    chi::Int
    cutoff::Float64
    upblocks::Vector
    downblocks::Vector
    center::Int
    blocks2::Vector
    center2::Int
end

function block(env::SingleEnvironment, idx::Int, up::Bool)
    if idx < 1 || idx > length(env.psi)
        return bMPO(length(env.psi))
    else
        if up
            return env.upblocks[idx]
        else
            return env.downblocks[idx]
        end
    end
end

function block2(env::SingleEnvironment, idx::Int)
    if idx < 1 || idx > length(env.psi)
        return ones(ComplexF64, 1, 1, 1)
    else
        return env.blocks2[idx]
    end
end

function findsite(env::Environment)
    site1 = env.center
    site2 = env.center2
    return site1, site2
end


### Building the environment
function buildup!(env::SingleEnvironment, idx::Int)
    # Fetch the upwards bond dimensions
    bonddims = []
    for i = 1:length(env.psi)
        push!(bonddims, size(env.psi[idx, i])[2])
    end

    # Create a random bMPS
    bMPS = randombMPS(length(env.psi), env.chi, bonddims)

    # Apply variational sweeps to limit the bond dimension
    bMPS = vbMPS(bMPS, env, true, idx; chi=env.chi, cutoff=env.cutoff)

    # Save the block
    env.upblocks[idx] = bMPS
end


function builddown!(env::Environment, idx::Int)
    # Fetch the upwards bond dimensions
    bonddims = []
    for i = 1:length(env.psi)
        push!(bonddims, size(env.psi[idx, i])[3])
    end

    # Create a random bMPO
    bMPS = randombMPS(length(env.psi), env.chi, bonddims)

    # Apply variational sweeps to limit the bond dimension
    bMPS = vbMPO(bMPS, env, false, idx; chi=env.chi, cutoff=env.cutoff)

    # Save the block
    env.downblocks[idx] = bMPS
end
