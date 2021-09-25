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

######################################
# MPS Structures
include("structures/mps/abstractmps.jl")
include("structures/mps/mps.jl")
include("structures/mps/mpo.jl")
include("structures/mps/abstractprojmps.jl")
include("structures/mps/projmps.jl")
include("structures/mps/oplist.jl")
include("structures/mps/projmpo.jl")
include("structures/mps/projmpssum.jl")
include("structures/mps/gatelist.jl")

######################################
# MPS algorithms
include("algorithms/mps/vmps.jl")
include("algorithms/mps/dmrg.jl")
include("algorithms/mps/tebd.jl")
include("algorithms/mps/qjmc.jl")
include("lattices/qkcms.jl")


######################################
# PEPS Structures
include("structures/peps/abstractpeps.jl")
include("structures/peps/peps.jl")
include("structures/peps/projbmpo.jl")
include("algorithms/peps/vbmpo.jl")
include("structures/peps/oplist2d.jl")
include("structures/peps/environment.jl")

#####################################
# PEPS algorithms
include("algorithms/peps/simpleupdate.jl")
include("algorithms/peps/fullupdate.jl")
