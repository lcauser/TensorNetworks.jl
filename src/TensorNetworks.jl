"""
TensorNetworks is a library which has efficient implementations for tensor
network architechures and algorithms
"""

module TensorNetworks

#####################################
# External packages
#
using MKL
using HDF5
using KrylovKit
using LinearAlgebra


#####################################
# Exports
include("exports.jl")


#####################################
# Imports
include("imports.jl")


#####################################
# Tensors
include("tensors.jl")

######################################
# Sitetypes
include("sitetypes.jl")
include("lattices/spinhalf.jl")

######################################
# Operator lists 
include("structures/mps/oplist.jl")
include("structures/mps/infiniteoplist.jl")

######################################
# Quantum states (exact...)
include("structures/qs/abstractqs.jl")
include("structures/qs/gqs.jl")
include("structures/qs/qs.jl")
include("structures/qs/qo.jl")

######################################
# MPS Structures
include("structures/mps/abstractmps.jl")
include("structures/mps/gmps.jl")
include("structures/mps/mps.jl")
include("structures/mps/mpo.jl")
include("structures/mps/projector.jl")
include("structures/mps/abstractprojmps.jl")
include("structures/mps/projmps.jl")
include("structures/mps/projmpssum.jl")
include("structures/mps/gatelist.jl")
include("structures/mps/umps.jl")


######################################
# MPS algorithms
include("algorithms/mps/vmps.jl")
include("algorithms/mps/dmrg.jl")
include("algorithms/mps/dmrgx.jl")
include("algorithms/mps/tebd.jl")
include("algorithms/mps/qjmc.jl")
include("lattices/qkcms.jl")


######################################
# Infinite MPS
include("structures/mps/igmps.jl")
include("algorithms/mps/itebd.jl")


######################################
# PEPS Structures
include("structures/peps/oplist2d.jl")
include("structures/peps/gates2d.jl")
include("structures/peps/abstractpeps.jl")
include("structures/peps/gpeps.jl")
include("structures/peps/peps.jl")
include("structures/peps/pepo.jl")
include("structures/peps/environment.jl")
#include("structures/peps/reducedenvironment.jl")
include("structures/peps/projbmps.jl")
#include("algorithms/peps/vbmpo.jl")

#####################################
# PEPS algorithms
include("algorithms/peps/simpleupdate.jl")
include("algorithms/peps/fullupdate.jl")
include("algorithms/peps/vpeps.jl")

end
