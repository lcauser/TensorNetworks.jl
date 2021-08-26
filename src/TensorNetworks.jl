"""
TensorNetworks is a library which has efficient implementations for tensor
network architechures and algorithms
"""

#module TensorNetworks

#####################################
# External packages
#
using HDF5
using KrylovKit
using LinearAlgebra


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
include("lattices/qkcms.jl")

######################################
# MPS Structures
include("structures/abstractmps.jl")
include("structures/mps.jl")
include("structures/mpo.jl")
include("structures/abstractprojmps.jl")
include("structures/projmps.jl")
include("structures/oplist.jl")
include("structures/projmpo.jl")
include("structures/projmpssum.jl")
include("structures/gatelist.jl")

######################################
# MPS algorithms
include("algorithms/vmps.jl")
include("algorithms/dmrg.jl")
include("algorithms/tebd.jl")
