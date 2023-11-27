# TensorNetworks.jl
Implements some basic algorithms for quantum many-body systems using Tensor network decompositions.

Matrix product states:
- Generalized framework to allow for higher ranked objects (e.g. matrices) to have a matrix product representation (e.g. MPO).
- High-level unified code for easy implementation. For example, writing Hamiltonians expliclity and using them to create MPOs (very much inspired by ITensors.jl!!!) and Trotterized time evolution.
- Variational MPS for targetting ground states (DMRG).
- Time-evolving block decimation (TEBD) for real- and imaginary-time dynamics.
- Thermal states using MPOs with TEBD.
- Quantum jump monte carlo (QJMC) with an MPS ansatz; time-evolution done with TEBD.

Projected entangled-pair states:
- Simple updates (SU) and full updates (FU) for targetting ground states via imaginary time evolution.
